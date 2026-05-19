"""
RAG (Retrieval-Augmented Generation) — FAISS bilan
Yuklangan fayllarni chunk + embedding qiladi.
FAISS IndexFlatIP (normalized) = cosine similarity, lekin C++ tezligida.
"""

import re
import json
import numpy as np
import requests
import faiss
from pathlib import Path
from file_reader import read_file_text

OLLAMA_URL  = "http://localhost:11435"
EMBED_MODEL = "qwen3-embedding:8b"
INDEX_FILE  = "rag_index.json"   # metadata (text, source, uploaded_by)
FAISS_FILE  = "rag_index.faiss"  # vector index


def _embed(text: str) -> list:
    # Yangi API (qwen3, bge)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            if "embeddings" in data: return data["embeddings"][0]
            if "embedding"  in data: return data["embedding"]
    except Exception:
        pass
    # Eski API (nomic-embed-text)
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _chunk_text(text: str, size: int = 200) -> list[str]:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    sentences = re.split(r'(?<=[.!?…])\s+|\n\n', text)
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        words = sent.split()
        if cur_len + len(words) > size and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = words, len(words)
        else:
            cur.extend(words)
            cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 10]


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 normalization → IndexFlatIP'da inner product = cosine similarity."""
    vecs = vecs.astype(np.float32, copy=False)
    faiss.normalize_L2(vecs)
    return vecs


class RAGIndex:
    def __init__(self):
        self.chunks: list[dict] = []   # {text, source, uploaded_by} — emb yo'q (FAISS'da)
        self.index : faiss.Index | None = None
        self.dim   : int | None = None

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.index is not None

    def stats(self) -> dict:
        sources, contributors = {}, {}
        for c in self.chunks:
            sources[c["source"]]                    = sources.get(c["source"], 0) + 1
            contributors[c.get("uploaded_by", "—")] = contributors.get(c.get("uploaded_by", "—"), 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "sources"     : sources,
            "contributors": contributors,
            "backend"     : "FAISS" + (f" (dim={self.dim})" if self.dim else ""),
        }

    # ── Build ──────────────────────────────────────────────────────────────
    def build(self, files: list, log=None) -> int:
        self.chunks = []
        self.index  = None

        # Fayl ma'lumotlarini normallashtirish
        norm = []
        for f in files:
            if isinstance(f, str):
                norm.append({"path": f, "source": Path(f).name, "uploaded_by": "—"})
            else:
                norm.append({
                    "path"       : f["path"],
                    "source"     : f.get("source") or Path(f["path"]).name,
                    "uploaded_by": f.get("uploaded_by", "—"),
                })

        # Chunklarga ajratish
        for item in norm:
            try:
                raw   = read_file_text(item["path"])
                parts = _chunk_text(raw)
                if log:
                    log(f"   📄 {item['source']} (👤 {item['uploaded_by']}) → {len(parts)} ta chunk")
                for chunk in parts:
                    self.chunks.append({
                        "text"       : chunk,
                        "source"     : item["source"],
                        "uploaded_by": item["uploaded_by"],
                    })
            except Exception as e:
                if log:
                    log(f"   ✗ {item['source']}: {e}")

        if not self.chunks:
            raise ValueError("Fayllardan chunk ajratib bo'lmadi")

        # Embeddinglarni hisoblash
        total = len(self.chunks)
        if log:
            log(f"⚡ {total} ta chunk uchun embedding hisoblanmoqda (FAISS)...")

        embs = []
        for i, item in enumerate(self.chunks):
            embs.append(_embed(item["text"]))
            if log and (i + 1) % 10 == 0:
                log(f"   {i+1}/{total} tayyor...")

        # FAISS index quramiz
        mat = np.array(embs, dtype=np.float32)
        self.dim = mat.shape[1]
        mat = _normalize(mat)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(mat)

        self.save()
        if log:
            log(f"✅ FAISS index tayyor: {total} chunk, dim={self.dim}")
        return total

    # ── Search ─────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
        if not self.ready:
            return []
        q = np.array([_embed(query)], dtype=np.float32)
        q = _normalize(q)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            c = self.chunks[idx]
            results.append({
                "text"       : c["text"],
                "source"     : c["source"],
                "uploaded_by": c.get("uploaded_by", "—"),
                "score"      : float(score),
            })
        return results

    # ── Persist ────────────────────────────────────────────────────────────
    def save(self, meta_path: str = INDEX_FILE, faiss_path: str = FAISS_FILE):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        if self.index is not None:
            faiss.write_index(self.index, faiss_path)

    def load(self, meta_path: str = INDEX_FILE, faiss_path: str = FAISS_FILE) -> bool:
        if not Path(meta_path).exists():
            return False
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            # Eski format (emb ichida bo'lsa) → tashlab yuboramiz
            self.chunks = [
                {"text": c["text"], "source": c["source"], "uploaded_by": c.get("uploaded_by", "—")}
                for c in data
            ]
            if Path(faiss_path).exists():
                self.index = faiss.read_index(faiss_path)
                self.dim   = self.index.d
            return self.ready
        except Exception as e:
            print(f"RAG load xato: {e}", flush=True)
            return False

    def clear(self):
        self.chunks = []
        self.index  = None
        self.dim    = None
        for p in (INDEX_FILE, FAISS_FILE):
            if Path(p).exists():
                Path(p).unlink()


def is_embed_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = r.json().get("models", [])
        names  = [m.get("name", "") for m in models]
        base   = EMBED_MODEL.split(":")[0]
        return any(EMBED_MODEL == n or base in n for n in names)
    except Exception:
        return False
