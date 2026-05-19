"""
RAG (Retrieval-Augmented Generation)
Yuklangan fayllarni chunk + embedding qiladi.
Savol kelganda mos qismlarni topib, modelga kontekst sifatida beradi.
Embedding model: Ollama'ning nomic-embed-text.
"""

import re
import json
import numpy as np
import requests
from pathlib import Path
from file_reader import read_file_text

OLLAMA_URL  = "http://localhost:11435"
EMBED_MODEL = "qwen3-embedding:8b"
INDEX_FILE  = "rag_index.json"


def _embed(text: str) -> list:
    # Yangi API: /api/embed (qwen3, bge, va boshqalar uchun)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            if "embeddings" in data:
                return data["embeddings"][0]
            if "embedding" in data:
                return data["embedding"]
    except Exception:
        pass

    # Eski API: /api/embeddings (nomic-embed-text uchun)
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _cosine(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _chunk_text(text: str, size: int = 200) -> list[str]:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # Sentence-aware chunking
    sentences = re.split(r'(?<=[.!?…])\s+|\n\n', text)
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
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


class RAGIndex:
    def __init__(self):
        self.chunks: list[dict] = []  # {text, emb, source}

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def stats(self) -> dict:
        sources = {}
        contributors = {}
        for c in self.chunks:
            sources[c["source"]] = sources.get(c["source"], 0) + 1
            u = c.get("uploaded_by", "—")
            contributors[u] = contributors.get(u, 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "sources"     : sources,
            "contributors": contributors,
        }

    # ── Build ──────────────────────────────────────────────────────────────
    def build(self, files: list, log=None) -> int:
        """
        files: list of dict {path, source, uploaded_by} yoki oddiy string list.
        """
        self.chunks = []

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

        for item in norm:
            try:
                raw   = read_file_text(item["path"])
                parts = _chunk_text(raw)
                if log:
                    log(f"   📄 {item['source']} (👤 {item['uploaded_by']}) → {len(parts)} ta chunk")
                for chunk in parts:
                    self.chunks.append({
                        "text"       : chunk,
                        "emb"        : None,
                        "source"     : item["source"],
                        "uploaded_by": item["uploaded_by"],
                    })
            except Exception as e:
                if log:
                    log(f"   ✗ {item['source']}: {e}")

        if not self.chunks:
            raise ValueError("Fayllardan chunk ajratib bo'lmadi")

        total = len(self.chunks)
        if log:
            log(f"⚡ {total} ta chunk uchun embedding yasalmoqda...")

        for i, item in enumerate(self.chunks):
            item["emb"] = _embed(item["text"])
            if log and (i + 1) % 5 == 0:
                log(f"   {i+1}/{total} tayyor...")

        self.save()
        if log:
            log(f"✅ RAG index tayyor: {total} ta chunk")
        return total

    # ── Search ─────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 4, threshold: float = 0.3) -> list[dict]:
        if not self.chunks:
            return []
        q_emb = _embed(query)
        scored = [
            {"text": item["text"],
             "source": item["source"],
             "uploaded_by": item.get("uploaded_by", "—"),
             "score": _cosine(q_emb, item["emb"])}
            for item in self.chunks
        ]
        scored.sort(key=lambda x: -x["score"])
        return [r for r in scored[:top_k] if r["score"] >= threshold]

    # ── Persist ────────────────────────────────────────────────────────────
    def save(self, path: str = INDEX_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self, path: str = INDEX_FILE) -> bool:
        if not Path(path).exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                self.chunks = json.load(f)
            return self.ready
        except Exception:
            return False

    def clear(self):
        self.chunks = []
        if Path(INDEX_FILE).exists():
            Path(INDEX_FILE).unlink()


def is_embed_available() -> bool:
    """EMBED_MODEL Ollama'da bormi tekshirish."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = r.json().get("models", [])
        names  = [m.get("name", "") for m in models]
        # To'liq nom yoki nom asosi mosligi
        base = EMBED_MODEL.split(":")[0]
        return any(EMBED_MODEL == n or base in n for n in names)
    except Exception:
        return False
