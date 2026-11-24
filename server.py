import sys, os, subprocess, shutil, time, base64, threading, requests, json, traceback
from io import BytesIO
from urllib.parse import urlparse

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU is NOT active! Go to Runtime -> Change runtime type -> Select T4 GPU and run again.")

print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

needed = []
for pkg, mod in [
    ("diffusers", "diffusers"), ("transformers", "transformers"), ("accelerate", "accelerate"),
    ("safetensors", "safetensors"), ("peft", "peft"), ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"), ("ultralytics", "ultralytics"), ("spandrel", "spandrel"),
    ("python-multipart", "multipart")
]:
    try:
        __import__(mod)
    except ImportError:
        needed.append(pkg)

if needed:
    print(f"Installing missing packages: {', '.join(needed)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

import spandrel, uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, AutoPipelineForInpainting, EulerAncestralDiscreteScheduler
from ultralytics import YOLO
from google.colab import drive, output

UI_URL = "https://raw.githubusercontent.com/AmbiguityLabs/Colab-sdxl/main/index.html"

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

base = "/content/drive/MyDrive/SDXL_Setup"
lora_dir = f"{base}/loras"
local_model = "/content/model.safetensors"
os.makedirs(f"{base}/outputs", exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)

drive_model = f"{base}/models/waiIllustriousSDXL_v170.safetensors"
if not os.path.exists(local_model):
    if not os.path.exists(drive_model):
        raise FileNotFoundError(f"Base model not found at: {drive_model}")
    print("Copying base model to local SSD...")
    shutil.copy(drive_model, local_model)

print("Loading SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_single_file(local_model, torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
inpaint_pipe = AutoPipelineForInpainting.from_pipe(pipe)

face_yolo = YOLO(f"{base}/adetailer/face_yolov8n.pt") if os.path.exists(f"{base}/adetailer/face_yolov8n.pt") else None
hand_yolo = YOLO(f"{base}/adetailer/hand_yolov8n.pt") if os.path.exists(f"{base}/adetailer/hand_yolov8n.pt") else None
upscale_model = spandrel.ModelLoader().load_from_file(f"{base}/upscalers/RealESRGAN_x4plus_anime_6B.pth").eval().half().to("cuda") if os.path.exists(f"{base}/upscalers/RealESRGAN_x4plus_anime_6B.pth") else None

active_lora = {"name": None}
jobs = {}

def get_drive_loras():
    if not os.path.exists(lora_dir): return []
    return sorted([f for f in os.listdir(lora_dir) if f.endswith(".safetensors")])

def apply_lora(lora_name, weight, log_fn):
    if not lora_name or lora_name == "none":
        if active_lora["name"]:
            log_fn("Unloading active LoRA...")
            try: pipe.unload_lora_weights()
            except: pass
            active_lora["name"] = None
        return

    lid = f"{lora_dir}/{lora_name}"
    if not os.path.exists(lid):
        raise FileNotFoundError(f"LoRA not found in Drive: {lora_name}")

    if active_lora["name"] != lid:
        log_fn(f"Loading LoRA from Drive: {lora_name}")
        try: pipe.unload_lora_weights()
        except: pass
        pipe.load_lora_weights(lora_dir, weight_name=lora_name, adapter_name="lora")
        active_lora["name"] = lid
    pipe.set_adapters(["lora"], adapter_weights=[weight])
    log_fn(f"LoRA active: {lora_name} (weight={weight})")

RGB_MAT = torch.tensor([[0.3448,0.4168,0.4395],[-0.1953,-0.0290,0.0250],[0.1074,0.0886,-0.0163],[-0.3730,-0.2499,-0.2088]], dtype=torch.float16, device="cuda")

def to_b64(latents):
    with torch.no_grad():
        rgb = (torch.einsum("bchw,cd->bdhw", latents, RGB_MAT) / 2.0 + 0.5).clamp(0, 1)
        img = Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        buf = BytesIO(); img.save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def adetail(img, p, np_txt, steps, cfg, ff, fh, ad_str, ad_blur, log_fn):
    for name, yolo in ([("Face", face_yolo)] if ff and face_yolo else []) + ([("Hand", hand_yolo)] if fh and hand_yolo else []):
        boxes = yolo(img, conf=0.35, verbose=False)[0].boxes.xyxy.cpu().numpy()
        log_fn(f"ADetailer ({name}): detected {len(boxes)} region(s)")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).rectangle([max(0, x1-16), max(0, y1-16), min(img.width, x2+16), min(img.height, y2+16)], fill=255)
            img = inpaint_pipe(
                prompt=p, negative_prompt=np_txt, image=img,
                mask_image=mask.filter(ImageFilter.GaussianBlur(ad_blur)),
                width=img.width, height=img.height,
                num_inference_steps=max(15, int(steps*0.7)),
                guidance_scale=cfg, strength=ad_str
            ).images[0]
    return img

app = FastAPI()

@app.get("/list_loras")
def list_loras():
    return {"loras": get_drive_loras()}

@app.post("/upload_lora")
async def upload_lora(file: UploadFile = File(...)):
    try:
        fname = os.path.basename(file.filename)
        if not fname.endswith(".safetensors"):
            return JSONResponse({"ok": False, "error": "Only .safetensors files are supported."}, status_code=400)
        dest = os.path.join(lora_dir, fname)
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        return {"ok": True, "filename": fname, "loras": get_drive_loras()}
    except Exception:
        return JSONResponse({"ok": False, "error": traceback.format_exc()}, status_code=500)

class UrlReq(BaseModel):
    url: str

@app.post("/download_lora")
def download_lora(req: UrlReq):
    try:
        u = req.url.strip()
        if not u:
            return JSONResponse({"ok": False, "error": "URL is empty."}, status_code=400)
        fname = os.path.basename(urlparse(u).path)
        if not fname.endswith(".safetensors"):
            fname = f"lora_{abs(hash(u))}.safetensors"
        dest = os.path.join(lora_dir, fname)
        if not os.path.exists(dest):
            with requests.get(u, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for c in r.iter_content(8192): f.write(c)
        return {"ok": True, "filename": fname, "loras": get_drive_loras()}
    except Exception:
        return JSONResponse({"ok": False, "error": traceback.format_exc()}, status_code=500)

class Req(BaseModel):
    prompt: str; neg_prompt: str; width: int; height: int; steps: int; cfg: float
    batch_count: int = 1; lora_name: str = "none"; lora_weight: float = 0.8
    fix_face: bool = False; fix_hand: bool = False; ad_str: float = 0.4; ad_blur: int = 4; upscale: bool = False

@app.post("/start_gen")
def start_gen(req: Req):
    job_id = str(int(time.time() * 1000))
    jobs[job_id] = {
        "status": "running", "logs": ["Starting generation pipeline..."],
        "previews": {}, "finals": {}, "steps": {}, "badges": {}, "error": None
    }
    job = jobs[job_id]

    def log(msg):
        job["logs"].append(msg)

    def work():
        try:
            apply_lora(req.lora_name, req.lora_weight, log)
            for i in range(max(1, min(req.batch_count, 8))):
                log(f"Generating image {i+1}/{req.batch_count} ({req.width}x{req.height}, steps={req.steps})")
                def cb(p, s, ts, kw):
                    job["previews"][i] = to_b64(kw["latents"])
                    job["steps"][i] = [s + 1, req.steps]
                    return kw
                img = pipe(prompt=req.prompt, negative_prompt=req.neg_prompt, width=req.width, height=req.height, num_inference_steps=req.steps, guidance_scale=req.cfg, callback_on_step_end=cb).images[0]

                if req.fix_face or req.fix_hand:
                    job["badges"][i] = "Detailer"
                    img = adetail(img, req.prompt, req.neg_prompt, req.steps, req.cfg, req.fix_face, req.fix_hand, req.ad_str, req.ad_blur, log)

                if req.upscale and upscale_model:
                    job["badges"][i] = "Upscaling 4x"
                    log("Running RealESRGAN 4x Anime6B upscaler...")
                    with torch.no_grad():
                        t = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).half().to("cuda")/255.0
                        img = Image.fromarray((upscale_model(t).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                        torch.cuda.empty_cache()

                out_path = f"{base}/outputs/img_{int(time.time())}_{i+1}.png"
                img.save(out_path)
                log(f"Saved: {out_path}")
                buf = BytesIO(); img.save(buf, format="PNG")
                job["finals"][i] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            job["status"] = "done"
        except Exception:
            job["status"] = "error"
            job["error"] = traceback.format_exc()

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}

@app.get("/job_status/{job_id}")
def job_status(job_id: str):
    return jobs.get(job_id, {"status": "error", "error": "Job not found"})

@app.get("/", response_class=HTMLResponse)
def index():
    return requests.get(UI_URL, timeout=15).text

threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error"), daemon=True).start()
time.sleep(2)
print("Studio ready.")
output.serve_kernel_port_as_iframe(8000, height=850)
