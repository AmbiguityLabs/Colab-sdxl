import subprocess, sys, os, shutil, time, base64, threading, requests, json, queue, traceback
from io import BytesIO
from urllib.parse import urlparse

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "safetensors", "peft", "torch", "fastapi", "uvicorn", "requests", "ultralytics", "spandrel", "python-multipart"])

import torch, spandrel, uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, AutoPipelineForInpainting, EulerAncestralDiscreteScheduler
from ultralytics import YOLO
from google.colab import drive, output

UI_URL = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/index.html"

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

base = "/content/drive/MyDrive/SDXL_Setup"
local_model = "/content/model.safetensors"
upload_lora_dir = "/content/uploaded_loras"
os.makedirs(f"{base}/outputs", exist_ok=True)
os.makedirs(upload_lora_dir, exist_ok=True)

if not os.path.exists(local_model):
    shutil.copy(f"{base}/models/waiIllustriousSDXL_v170.safetensors", local_model)

pipe = StableDiffusionXLPipeline.from_single_file(local_model, torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
inpaint_pipe = AutoPipelineForInpainting.from_pipe(pipe)

face_yolo = YOLO(f"{base}/adetailer/face_yolov8n.pt") if os.path.exists(f"{base}/adetailer/face_yolov8n.pt") else None
hand_yolo = YOLO(f"{base}/adetailer/hand_yolov8n.pt") if os.path.exists(f"{base}/adetailer/hand_yolov8n.pt") else None
upscale_model = spandrel.ModelLoader().load_from_file(f"{base}/upscalers/RealESRGAN_x4plus_anime_6B.pth").eval().half().to("cuda") if os.path.exists(f"{base}/upscalers/RealESRGAN_x4plus_anime_6B.pth") else None

active_lora = {"name": None}

def apply_lora(mode, local_name, remote_url, uploaded_name, weight, log_fn):
    if mode == "none":
        if active_lora["name"]:
            log_fn("Unloading active LoRA...")
            try: pipe.unload_lora_weights()
            except: pass
            active_lora["name"] = None
        return

    if mode == "local":
        if not local_name: raise ValueError("No Drive LoRA selected.")
        t_dir, t_file = f"{base}/loras", local_name
    elif mode == "upload":
        if not uploaded_name: raise ValueError("No uploaded LoRA file found. Please upload a .safetensors file first.")
        t_dir, t_file = upload_lora_dir, uploaded_name
    elif mode == "remote":
        if not remote_url.strip(): raise ValueError("Remote LoRA URL is empty.")
        fname = os.path.basename(urlparse(remote_url.strip()).path)
        if not fname.endswith(".safetensors"): fname = f"lora_{abs(hash(remote_url))}.safetensors"
        dest = f"{upload_lora_dir}/{fname}"
        if not os.path.exists(dest):
            log_fn(f"Downloading LoRA from URL: {remote_url.strip()}")
            with requests.get(remote_url.strip(), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for c in r.iter_content(8192): f.write(c)
            log_fn(f"Downloaded LoRA: {fname}")
        else:
            log_fn(f"Using cached LoRA: {fname}")
        t_dir, t_file = upload_lora_dir, fname
    else:
        return

    lid = f"{t_dir}/{t_file}"
    if active_lora["name"] != lid:
        log_fn(f"Loading LoRA weights: {t_file} (weight={weight})")
        try: pipe.unload_lora_weights()
        except: pass
        pipe.load_lora_weights(t_dir, weight_name=t_file, adapter_name="lora")
        active_lora["name"] = lid
    pipe.set_adapters(["lora"], adapter_weights=[weight])
    log_fn(f"LoRA active: {t_file} at weight {weight}")

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
        for idx, box in enumerate(boxes):
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

@app.post("/upload_lora")
async def upload_lora(file: UploadFile = File(...)):
    try:
        fname = os.path.basename(file.filename)
        if not fname.endswith(".safetensors"):
            return JSONResponse({"ok": False, "error": "Only .safetensors files are supported."}, status_code=400)
        dest = os.path.join(upload_lora_dir, fname)
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        return {"ok": True, "filename": fname, "size_mb": round(os.path.getsize(dest) / (1024*1024), 2)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": traceback.format_exc()}, status_code=500)

class Req(BaseModel):
    prompt: str; neg_prompt: str; width: int; height: int; steps: int; cfg: float
    batch_count: int = 1
    lora_mode: str = "none"; local_lora: str = ""; remote_url: str = ""; uploaded_lora: str = ""; lora_weight: float = 0.8
    fix_face: bool = False; fix_hand: bool = False; ad_str: float = 0.4; ad_blur: int = 4; upscale: bool = False

@app.post("/generate_stream")
def gen(req: Req):
    q = queue.Queue()
    def log(msg):
        print(f"[STUDIO] {msg}")
        q.put(json.dumps({"type": "log", "message": msg}))

    def work():
        try:
            log("Starting generation pipeline...")
            apply_lora(req.lora_mode, req.local_lora, req.remote_url, req.uploaded_lora, req.lora_weight, log)
            for i in range(max(1, min(req.batch_count, 8))):
                log(f"Generating image {i+1}/{req.batch_count} ({req.width}x{req.height}, steps={req.steps})")
                cb = lambda p, s, ts, kw: (q.put(json.dumps({"type":"preview","index":i,"step":s+1,"total_steps":req.steps,"image":to_b64(kw["latents"])})), kw)[1]
                img = pipe(prompt=req.prompt, negative_prompt=req.neg_prompt, width=req.width, height=req.height, num_inference_steps=req.steps, guidance_scale=req.cfg, callback_on_step_end=cb).images[0]

                if req.fix_face or req.fix_hand:
                    q.put(json.dumps({"type":"status","index":i,"text":"Detailer"}))
                    img = adetail(img, req.prompt, req.neg_prompt, req.steps, req.cfg, req.fix_face, req.fix_hand, req.ad_str, req.ad_blur, log)

                if req.upscale and upscale_model:
                    q.put(json.dumps({"type":"status","index":i,"text":"Upscaling 4x"}))
                    log("Running RealESRGAN 4x Anime6B upscaler...")
                    with torch.no_grad():
                        t = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).half().to("cuda")/255.0
                        img = Image.fromarray((upscale_model(t).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                        torch.cuda.empty_cache()

                out_path = f"{base}/outputs/img_{int(time.time())}_{i+1}.png"
                img.save(out_path)
                log(f"Saved: {out_path}")
                buf = BytesIO(); img.save(buf, format="PNG")
                q.put(json.dumps({"type":"final","index":i,"image":"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()}))
            q.put(json.dumps({"type":"done"}))
        except Exception:
            err_trace = traceback.format_exc()
            print(err_trace)
            q.put(json.dumps({"type":"error","message":err_trace}))
        finally:
            q.put(None)
    threading.Thread(target=work, daemon=True).start()
    return StreamingResponse((f"data: {x}\n\n" for x in iter(q.get, None)), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
def index():
    html = requests.get(UI_URL, timeout=15).text
    loras = [f for f in os.listdir(f"{base}/loras") if f.endswith(".safetensors")] if os.path.exists(f"{base}/loras") else []
    return html.replace("__LORA_OPTIONS__", "".join([f'<option value="{f}">{f}</option>' for f in loras]))

threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error"), daemon=True).start()
time.sleep(2)
output.serve_kernel_port_as_window(8000)
