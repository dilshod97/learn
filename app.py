import os
import json
import asyncio
import threading
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trainer import Trainer

# ── State ─────────────────────────────────────────────────────────────────────
LORA_DIR = Path("./uzbek-gpt-lora")

def _init_state() -> dict:
    base = {
        "status"          : "idle",
        "progress"        : 0,
        "logs"            : [],
        "uploaded_files"  : [],
        "ollama_model"    : None,
        "recent_requests" : [],   # [{time, model, question, duration_ms}]
    }
    if LORA_DIR.exists() and (LORA_DIR / "adapter_config.json").exists():
        base.update({
            "status"      : "ready",
            "progress"    : 100,
            "logs"        : ["✅ Mavjud LoRA adapter topildi"],
            "ollama_model": "uzbek-gpt-local",
        })
    return base

state: dict = _init_state()
trainer = Trainer(state)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11435"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="O'zbek GPT Fine-Tuning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text(encoding="utf-8")


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append(str(dest))
    state["uploaded_files"].extend(saved)
    return {"uploaded": [Path(p).name for p in saved], "total": len(state["uploaded_files"])}


@app.delete("/api/upload")
async def clear_uploads():
    state["uploaded_files"] = []
    return {"ok": True}


# ── Training ──────────────────────────────────────────────────────────────────
@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks):
    if state["status"] in ("preparing", "training", "converting"):
        raise HTTPException(400, "Training hozir ham davom etmoqda")
    if not state["uploaded_files"]:
        raise HTTPException(400, "Avval fayl yuklang")

    background_tasks.add_task(_run_training)
    return {"message": "Training boshlandi"}


def _run_training():
    trainer.run()


@app.get("/api/status")
async def get_status():
    return {
        "status"  : state["status"],
        "progress": state["progress"],
        "model"   : state.get("ollama_model"),
    }


@app.get("/api/logs")
async def stream_logs():
    """Server-Sent Events: real-time training logs."""
    async def generate():
        idx = 0
        while True:
            logs = state.get("logs", [])
            while idx < len(logs):
                yield f"data: {json.dumps({'log': logs[idx]})}\n\n"
                idx += 1

            if state["status"] in ("ready", "error"):
                yield f"data: {json.dumps({'done': True, 'status': state['status']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []
    model: str = "uzbek-gpt"


def _log_request(model: str, question: str, duration_ms: int, status: str = "ok"):
    from datetime import datetime
    state["recent_requests"].append({
        "time"       : datetime.now().strftime("%H:%M:%S"),
        "model"      : model,
        "question"   : (question[:80] + "…") if len(question) > 80 else question,
        "duration_ms": duration_ms,
        "status"     : status,
    })
    state["recent_requests"] = state["recent_requests"][-30:]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    import time
    t0 = time.time()

    requested = req.model or state.get("ollama_model") or "gpt-oss:20b"

    # uzbek-gpt-local → Ollama da gpt-oss:20b ni o'zbek system prompt bilan
    if requested == "uzbek-gpt-local":
        actual_model = "gpt-oss:20b"
        system = (
            "Siz o'zbek tilidagi AI yordamchisiz. Foydalanuvchiga doim o'zbek tilida "
            "aniq va foydali javob bering. Qisqa savolga qisqa, batafsil savolga batafsil."
        )
    else:
        actual_model = requested
        system = "Siz foydali AI yordamchisiz."

    messages = [{"role": "system", "content": system}]
    for h in req.history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": actual_model, "messages": messages, "stream": False},
            timeout=300,
        )
        ms = int((time.time() - t0) * 1000)

        if resp.status_code == 404:
            _log_request(requested, req.message, ms, "model-not-found")
            raise HTTPException(404, f"Model '{actual_model}' Ollama da topilmadi. "
                                     f"O'rnatish: ollama pull {actual_model}")

        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
        _log_request(requested, req.message, ms, "ok")
        return {"response": answer}

    except requests.ConnectionError:
        _log_request(requested, req.message, int((time.time()-t0)*1000), "ollama-down")
        raise HTTPException(503, "Ollama ishlamayapti. 'ollama serve' ni ishga tushiring.")
    except HTTPException:
        raise
    except Exception as e:
        _log_request(requested, req.message, int((time.time()-t0)*1000), "error")
        raise HTTPException(500, str(e))


# ── Ollama helpers ────────────────────────────────────────────────────────────
@app.get("/api/ollama/models")
async def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        return r.json()
    except Exception:
        return {"models": []}


@app.get("/api/ollama/status")
async def ollama_status():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return {"running": True}
    except Exception:
        return {"running": False}


# ── Monitoring ────────────────────────────────────────────────────────────────
import subprocess
import psutil

def _gpu_info() -> list:
    """nvidia-smi orqali GPU ma'lumotlari."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=4)
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            gpus.append({
                "index"       : int(p[0]),
                "name"        : p[1],
                "utilization" : int(p[2]),
                "mem_used"    : int(p[3]),
                "mem_total"   : int(p[4]),
                "temperature" : int(p[5]),
                "power_draw"  : float(p[6]) if p[6] != "N/A" else None,
                "power_limit" : float(p[7]) if p[7] != "N/A" else None,
            })
        return gpus
    except Exception:
        pass

    # PyTorch MPS/CUDA fallback
    try:
        import torch
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index"       : i,
                    "name"        : props.name,
                    "utilization" : None,
                    "mem_used"    : torch.cuda.memory_allocated(i) // 1024**2,
                    "mem_total"   : props.total_memory // 1024**2,
                    "temperature" : None,
                    "power_draw"  : None,
                    "power_limit" : None,
                })
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpus.append({
                "index": 0, "name": "Apple MPS",
                "utilization": None, "mem_used": None, "mem_total": None,
                "temperature": None, "power_draw": None, "power_limit": None,
            })
        return gpus
    except Exception:
        return []


def _ollama_ps() -> list:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        models = r.json().get("models", [])
        result = []
        for m in models:
            result.append({
                "name"      : m.get("name", ""),
                "size_vram" : round(m.get("size_vram", 0) / 1024**3, 2),
                "size"      : round(m.get("size", 0) / 1024**3, 2),
                "expires_at": m.get("expires_at", ""),
            })
        return result
    except Exception:
        return []


@app.get("/api/monitor")
async def monitor():
    vm  = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)
    return {
        "gpu": _gpu_info(),
        "ram": {
            "used"   : round(vm.used  / 1024**3, 1),
            "total"  : round(vm.total / 1024**3, 1),
            "percent": vm.percent,
        },
        "cpu": {
            "percent": cpu,
            "cores"  : psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        },
        "ollama_models": _ollama_ps(),
        "training_status": state["status"],
        "recent_requests": list(reversed(state.get("recent_requests", []))),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8088, reload=False)
