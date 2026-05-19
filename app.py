import os
import json
import time
import asyncio
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import uvicorn
import psutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Header, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
from rag import RAGIndex, is_embed_available, EMBED_MODEL
from trainer import Trainer
from file_reader import ALLOWED_EXT

# ── Init ──────────────────────────────────────────────────────────────────────
db.init_db()
db.cleanup_expired_sessions()

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11435"

# Per-user holatlar (xotirada)
user_states: dict[int, dict] = {}
user_trainers: dict[int, Trainer] = {}

# ── Global (umumiy) RAG — FAISS asosida ───────────────────────────────────────
global_rag = RAGIndex()
global_rag.load()  # rag_index.json + rag_index.faiss
global_rag_state = {"building": False, "logs": []}


def _ensure_user_state(uid: int) -> dict:
    if uid not in user_states:
        user_states[uid] = {
            "training": {"status": "idle", "progress": 0, "logs": []},
        }
    return user_states[uid]


def _user_trainer(uid: int) -> Trainer:
    if uid not in user_trainers:
        st = _ensure_user_state(uid)
        user_trainers[uid] = Trainer(st["training"])
        user_trainers[uid].cfg["output_dir"]   = f"./uzbek-gpt-lora-{uid}"
        user_trainers[uid].cfg["merged_dir"]   = f"./uzbek-gpt-merged-{uid}"
        user_trainers[uid].cfg["dataset_file"] = f"uzbek_dataset_{uid}.json"
    return user_trainers[uid]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="HP-AI Audit Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def current_user(token: Optional[str] = Cookie(None, alias="session"),
                 authorization: Optional[str] = Header(None)) -> dict:
    """Cookie yoki Authorization: Bearer ... orqali user oladi."""
    sess = token
    if not sess and authorization and authorization.lower().startswith("bearer "):
        sess = authorization.split(None, 1)[1].strip()
    user = db.get_session_user(sess) if sess else None
    if not user:
        raise HTTPException(401, "Avtorizatsiya kerak")
    return user


def admin_required(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Faqat admin")
    return user


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text(encoding="utf-8")


# ── Auth API ──────────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(req: LoginReq, response: Response):
    user = db.get_user_by_name(req.username)
    if not user or not user["is_active"] or not db.verify_password(req.password, user["salt"], user["password_hash"]):
        raise HTTPException(401, "Login yoki parol noto'g'ri")
    token = db.create_session(user["id"])
    response.set_cookie("session", token, max_age=7*24*3600, httponly=True, samesite="lax")
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "role": user["role"]}}


@app.post("/api/auth/logout")
async def logout(response: Response, token: Optional[str] = Cookie(None, alias="session")):
    if token:
        db.delete_session(token)
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: dict = Depends(current_user)):
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


# ── Upload ────────────────────────────────────────────────────────────────────
def _user_upload_dir(uid: int) -> Path:
    p = UPLOAD_ROOT / str(uid)
    p.mkdir(exist_ok=True, parents=True)
    return p


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...), user: dict = Depends(current_user)):
    uid = user["id"]
    udir = _user_upload_dir(uid)

    results = []
    for f in files:
        # Faqat .txt va .docx ruxsat etiladi
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXT:
            results.append({"name": f.filename, "status": "bad-format"})
            continue

        content = await f.read()
        if not content:
            results.append({"name": f.filename, "status": "empty"})
            continue

        file_hash = hashlib.sha256(content).hexdigest()
        if db.file_exists_for_user(uid, file_hash):
            results.append({"name": f.filename, "status": "duplicate"})
            continue

        dest = udir / f.filename
        # nom konflikti bo'lsa, suffix qo'shamiz
        if dest.exists():
            base, ext = os.path.splitext(f.filename)
            i = 1
            while (udir / f"{base}_{i}{ext}").exists():
                i += 1
            dest = udir / f"{base}_{i}{ext}"
        dest.write_bytes(content)

        db.add_file(uid, dest.name, str(dest), file_hash, len(content))
        results.append({"name": dest.name, "status": "ok", "size": len(content)})

    return {"results": results}


@app.get("/api/files")
async def list_files(user: dict = Depends(current_user)):
    """Foydalanuvchi o'z fayllarini ko'radi."""
    files = db.list_files(user["id"])
    return {"files": files}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int, user: dict = Depends(current_user)):
    f = db.get_file(file_id)
    if not f:
        raise HTTPException(404, "Fayl topilmadi")
    if f["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(403, "Boshqa user fayli")
    try:
        Path(f["path"]).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete_file(file_id)
    return {"ok": True}


@app.delete("/api/files")
async def clear_files(user: dict = Depends(current_user)):
    """Hozirgi userning hamma fayllarini o'chir."""
    for f in db.list_files(user["id"]):
        try:
            Path(f["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    db.clear_user_files(user["id"])
    return {"ok": True}


# ── Global RAG ────────────────────────────────────────────────────────────────
def _build_global_rag_bg(builder_uid: int, file_specs: list[dict]):
    job_id = db.add_training_job(builder_uid, "rag", "running")
    try:
        global_rag_state["building"] = True
        global_rag_state["logs"]     = []
        global_rag_state["logs"].append(
            f"📚 Umumiy bilim bazasi qurilmoqda ({len(file_specs)} ta fayl)..."
        )
        global_rag.build(file_specs, log=lambda m: global_rag_state["logs"].append(m))
        global_rag.save()
        db.update_training_job(job_id, "done")
    except Exception as e:
        global_rag_state["logs"].append(f"❌ Xato: {e}")
        db.update_training_job(job_id, "error", str(e))
    finally:
        global_rag_state["building"] = False


@app.post("/api/rag/build")
async def rag_build(background_tasks: BackgroundTasks, user: dict = Depends(current_user)):
    if global_rag_state["building"]:
        raise HTTPException(400, "RAG hozir qurilmoqda (boshqa user tomonidan)")

    # Barcha userlarning barcha fayllari
    all_files = db.list_files()
    if not all_files:
        raise HTTPException(400, "Hech qanday fayl yo'q")
    if not is_embed_available():
        raise HTTPException(400, f"{EMBED_MODEL} Ollama'da topilmadi")

    specs = [{
        "path"       : f["path"],
        "source"     : f["filename"],
        "uploaded_by": f["username"],
    } for f in all_files]

    background_tasks.add_task(_build_global_rag_bg, user["id"], specs)
    return {"message": "Umumiy RAG quriliyapti"}


@app.get("/api/rag/status")
async def rag_status(user: dict = Depends(current_user)):
    return {
        "ready"   : global_rag.ready,
        "building": global_rag_state["building"],
        "stats"   : global_rag.stats(),
        "logs"    : global_rag_state["logs"][-30:],
    }


@app.delete("/api/rag")
async def rag_clear(admin: dict = Depends(admin_required)):
    """Faqat admin RAG'ni tozalashi mumkin."""
    global_rag.clear()
    return {"ok": True}


# ── Training ──────────────────────────────────────────────────────────────────
@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, user: dict = Depends(current_user)):
    uid = user["id"]
    st  = _ensure_user_state(uid)["training"]
    if st["status"] in ("preparing", "training", "converting"):
        raise HTTPException(400, "Training davom etmoqda")

    files = db.list_files(uid)
    if not files:
        raise HTTPException(400, "Avval fayl yuklang")

    trainer = _user_trainer(uid)
    trainer.state["uploaded_files"] = [f["path"] for f in files]

    job_id = db.add_training_job(uid, "finetune", "running")

    def run():
        try:
            trainer.run()
            db.update_training_job(job_id, "done")
        except Exception as e:
            db.update_training_job(job_id, "error", str(e))

    background_tasks.add_task(run)
    return {"message": "Training boshlandi"}


@app.get("/api/status")
async def get_status(user: dict = Depends(current_user)):
    uid = user["id"]
    st  = _ensure_user_state(uid)["training"]
    return {"status": st["status"], "progress": st["progress"]}


@app.get("/api/logs")
async def stream_logs(user: dict = Depends(current_user)):
    uid = user["id"]
    st  = _ensure_user_state(uid)["training"]

    async def generate():
        idx = 0
        while True:
            logs = st.get("logs", [])
            while idx < len(logs):
                yield f"data: {json.dumps({'log': logs[idx]})}\n\n"
                idx += 1
            if st["status"] in ("ready", "error"):
                yield f"data: {json.dumps({'done': True, 'status': st['status']})}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    message: str
    history: list = []
    model  : str = "gpt-oss:20b"
    use_rag: bool = True


@app.post("/api/chat")
async def chat(req: ChatReq, user: dict = Depends(current_user)):
    uid = user["id"]
    t0  = time.time()
    actual = req.model or "gpt-oss:20b"
    if actual == "uzbek-gpt-local":
        actual = "gpt-oss:20b"

    # User saqlagan custom prompt — bo'lmasa default
    settings = db.get_user_settings(uid)
    base_prompt = settings.get("system_prompt", "").strip()
    if not base_prompt:
        base_prompt = (
            "Siz HP-AI Audit Assistant — o'zbek tilidagi AI yordamchisiz. "
            "Foydalanuvchiga doim o'zbek tilida aniq va foydali javob bering. "
            "JAVOBNI FORMATLASH QOIDALARI:\n"
            "1) Jadvallar uchun har doim markdown jadval ishlating: | ustun | ustun |\n"
            "2) Sarlavhalar uchun ##, ro'yxat uchun -, muhim so'zlar uchun **bold**.\n"
            "3) Agar javobda yillar/davrlar bo'yicha sonlar bo'lsa (masalan daromad, "
            "xarajat, foiz), markdown jadvaldan TASHQARI quyidagi formatda diagramma ham bering:\n"
            "```chart\n{\n  \"type\": \"line\",\n  \"data\": {\n    \"labels\": [\"2017\",\"2018\",\"2019\"],\n"
            "    \"datasets\": [{\"label\": \"Daromad\",\"data\": [100,200,300],\"borderColor\":\"#3b82f6\",\"backgroundColor\":\"#3b82f655\"}]\n"
            "  }\n}\n```\n"
            "Chart turlari: 'line' (vaqt bo'yicha trend), 'bar' (taqqoslash), 'pie' (ulush), 'doughnut'.\n"
            "Faqat raqamli ma'lumotlar bo'lsa chart bering — matnli javoblar uchun shart emas."
        )

    system  = base_prompt
    sources = []

    if req.use_rag and global_rag.ready:
        try:
            hits = global_rag.search(req.message, top_k=5, threshold=0.3)
            if hits:
                ctx_parts = []
                for i, h in enumerate(hits, 1):
                    ctx_parts.append(f"[Manba {i}: {h['source']} — yuklagan: {h['uploaded_by']}]\n{h['text']}")
                    sources.append({
                        "source"     : h["source"],
                        "uploaded_by": h["uploaded_by"],
                        "score"      : round(h["score"], 3),
                    })
                ctx = "\n\n".join(ctx_parts)
                system = (f"{system}\n\nQuyidagi hujjatlardan foydalaning. Faqat ulardan oling, "
                          f"to'qib chiqarmang. Topilmasa 'Hujjatlarda topilmadi' deb yozing.\n\n"
                          f"=== HUJJATLAR ===\n{ctx}\n=== TUGADI ===")
        except Exception as e:
            print(f"RAG search xato: {e}", flush=True)

    messages = [{"role": "system", "content": system}]
    for h in req.history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": actual, "messages": messages, "stream": False},
                          timeout=300)
        ms = int((time.time() - t0) * 1000)

        if r.status_code == 404:
            db.add_chat_log(uid, actual, req.message, "", bool(sources), ms, "model-not-found")
            raise HTTPException(404, f"Model '{actual}' Ollama'da yo'q")

        r.raise_for_status()
        answer = r.json()["message"]["content"]
        db.add_chat_log(uid, actual, req.message, answer, bool(sources), ms, "ok")
        return {
            "response"   : answer,
            "sources"    : sources,
            "rag_used"   : bool(sources),
            "duration_ms": ms,
        }

    except requests.ConnectionError:
        db.add_chat_log(uid, actual, req.message, "", bool(sources), int((time.time()-t0)*1000), "ollama-down")
        raise HTTPException(503, "Ollama ishlamayapti")
    except HTTPException:
        raise
    except Exception as e:
        db.add_chat_log(uid, actual, req.message, "", bool(sources), int((time.time()-t0)*1000), "error")
        raise HTTPException(500, str(e))


# ── Ollama helpers ────────────────────────────────────────────────────────────
@app.get("/api/ollama/models")
async def list_ollama_models(user: dict = Depends(current_user)):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        return r.json()
    except Exception:
        return {"models": []}


@app.get("/api/ollama/status")
async def ollama_status(user: dict = Depends(current_user)):
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return {"running": True}
    except Exception:
        return {"running": False}


# ── Activity (user o'zining) ──────────────────────────────────────────────────
@app.get("/api/activity")
async def my_activity(user: dict = Depends(current_user)):
    return {
        "chats"   : db.list_chat_logs(user["id"], limit=30),
        "trainings": db.list_training_jobs(user["id"], limit=20),
    }


class SettingsReq(BaseModel):
    system_prompt: str = ""


@app.get("/api/settings")
async def get_settings(user: dict = Depends(current_user)):
    return db.get_user_settings(user["id"])


@app.patch("/api/settings")
async def update_settings(req: SettingsReq, user: dict = Depends(current_user)):
    db.set_user_settings(user["id"], req.system_prompt.strip())
    return {"ok": True}


@app.get("/api/chat/history")
async def chat_history(user: dict = Depends(current_user), limit: int = 100):
    """Hozirgi user o'zining suhbat tarixini chronological tartibda oladi."""
    rows = db.list_chat_logs(user["id"], limit=limit)
    # DB DESC qaytaradi — eskidan yangiga
    rows = list(reversed(rows))
    history = []
    for r in rows:
        history.append({
            "question": r["question"] or "",
            "answer"  : r["answer"] or "",
            "rag_used": bool(r["rag_used"]),
            "model"   : r["model"] or "",
        })
    return {"history": history}


@app.delete("/api/chat/history")
async def chat_history_clear(user: dict = Depends(current_user)):
    """User o'z suhbat tarixini tozalaydi."""
    with db.db() as c:
        c.execute("DELETE FROM chat_logs WHERE user_id = ?", (user["id"],))
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(admin_required)):
    return {"users": db.list_users()}


class NewUserReq(BaseModel):
    username: str
    password: str
    role: str = "user"


@app.post("/api/admin/users")
async def admin_create_user(req: NewUserReq, admin: dict = Depends(admin_required)):
    if req.role not in ("user", "admin"):
        raise HTTPException(400, "role: user yoki admin")
    uid = db.create_user(req.username, req.password, req.role)
    if uid is None:
        raise HTTPException(400, "Bu nom band")
    return {"id": uid}


class UpdateUserReq(BaseModel):
    role     : Optional[str] = None
    is_active: Optional[bool] = None
    password : Optional[str] = None


@app.patch("/api/admin/users/{uid}")
async def admin_update_user(uid: int, req: UpdateUserReq, admin: dict = Depends(admin_required)):
    if req.role: db.set_user_role(uid, req.role)
    if req.is_active is not None: db.set_user_active(uid, req.is_active)
    if req.password: db.reset_password(uid, req.password)
    return {"ok": True}


@app.delete("/api/admin/users/{uid}")
async def admin_delete_user(uid: int, admin: dict = Depends(admin_required)):
    if uid == admin["id"]:
        raise HTTPException(400, "O'zingizni o'chira olmaysiz")
    db.delete_user(uid)
    return {"ok": True}


@app.get("/api/admin/files")
async def admin_all_files(admin: dict = Depends(admin_required)):
    return {"files": db.list_files()}


@app.get("/api/files/{file_id}/download")
async def download_file(file_id: int, user: dict = Depends(current_user)):
    """User o'z faylini yuklab oladi, admin esa hammasini."""
    f = db.get_file(file_id)
    if not f:
        raise HTTPException(404, "Fayl topilmadi")
    if f["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(403, "Boshqa user fayli")
    path = Path(f["path"])
    if not path.exists():
        raise HTTPException(404, "Fayl diskda yo'q")
    return FileResponse(str(path), filename=f["filename"], media_type="application/octet-stream")


@app.get("/api/admin/chats")
async def admin_all_chats(admin: dict = Depends(admin_required)):
    return {"chats": db.list_chat_logs(limit=200)}


@app.get("/api/admin/trainings")
async def admin_all_trainings(admin: dict = Depends(admin_required)):
    return {"trainings": db.list_training_jobs(limit=100)}


# ── Monitoring (umumiy — admin ham, user ham ko'radi) ─────────────────────────
def _gpu_info() -> list:
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
                "index": int(p[0]), "name": p[1],
                "utilization": int(p[2]),
                "mem_used": int(p[3]), "mem_total": int(p[4]),
                "temperature": int(p[5]),
                "power_draw": float(p[6]) if p[6] != "N/A" else None,
                "power_limit": float(p[7]) if p[7] != "N/A" else None,
            })
        return gpus
    except Exception:
        return []


def _ollama_ps() -> list:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        models = r.json().get("models", [])
        return [{
            "name"     : m.get("name", ""),
            "size_vram": round(m.get("size_vram", 0) / 1024**3, 2),
            "size"     : round(m.get("size", 0) / 1024**3, 2),
        } for m in models]
    except Exception:
        return []


@app.get("/api/monitor")
async def monitor(user: dict = Depends(current_user)):
    vm  = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)

    # User uchun: o'z chat log lari
    # Admin uchun: hammasi
    if user["role"] == "admin":
        recent = db.list_chat_logs(limit=30)
    else:
        recent = db.list_chat_logs(user["id"], limit=30)

    def _fmt_time(v) -> str:
        if not v: return ""
        if hasattr(v, "strftime"):
            return v.strftime("%H:%M:%S")
        s = str(v)
        return s[11:19] if len(s) >= 19 else s

    recent_formatted = []
    for r in recent:
        q = r["question"] or ""
        recent_formatted.append({
            "time"       : _fmt_time(r["created_at"]),
            "user"       : r["username"] if "username" in r.keys() else "",
            "model"      : r["model"] or "",
            "question"   : (q[:80] + "…") if len(q) > 80 else q,
            "duration_ms": r["duration_ms"] or 0,
            "status"     : r["status"] or "ok",
            "rag"        : bool(r["rag_used"]),
        })

    return {
        "gpu": _gpu_info(),
        "ram": {"used": round(vm.used/1024**3,1), "total": round(vm.total/1024**3,1), "percent": vm.percent},
        "cpu": {"percent": cpu, "cores": psutil.cpu_count(False), "threads": psutil.cpu_count(True)},
        "ollama_models": _ollama_ps(),
        "recent_requests": recent_formatted,
        "is_admin": user["role"] == "admin",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8088, reload=False)
