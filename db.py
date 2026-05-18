"""SQLite baza — barcha user data shu yerda."""
import sqlite3
import secrets
import hashlib
import hmac
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path("app.db")


def _conn():
    c = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


@contextmanager
def db():
    c = _conn()
    try:
        yield c
        c.commit()
    finally:
        c.close()


# ── Schema ───────────────────────────────────────────────────────────────────
def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',  -- user | admin
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active     INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at  TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename    TEXT NOT NULL,
            path        TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            size_bytes  INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, file_hash)
        );

        CREATE TABLE IF NOT EXISTS chat_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            model       TEXT NOT NULL,
            question    TEXT NOT NULL,
            answer      TEXT,
            rag_used    INTEGER DEFAULT 0,
            duration_ms INTEGER,
            status      TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS training_jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind         TEXT NOT NULL,        -- finetune | rag
            status       TEXT NOT NULL,        -- preparing | running | done | error
            started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at  TIMESTAMP,
            error        TEXT
        );

        CREATE TABLE IF NOT EXISTS user_settings (
            user_id        INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            system_prompt  TEXT DEFAULT '',
            updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_files_user      ON files(user_id);
        CREATE INDEX IF NOT EXISTS idx_chat_user_time  ON chat_logs(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_train_user_time ON training_jobs(user_id, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_session_expires ON sessions(expires_at);
        """)

    # Default admin yaratish
    if not get_user_by_name("admin"):
        create_user("admin", "admin", role="admin")
        print("✅ Default admin yaratildi: admin / admin")


# ── Parol ────────────────────────────────────────────────────────────────────
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def verify_password(password: str, salt: str, hash_: str) -> bool:
    return hmac.compare_digest(_hash_password(password, salt), hash_)


# ── Users ────────────────────────────────────────────────────────────────────
def create_user(username: str, password: str, role: str = "user") -> Optional[int]:
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, salt, role) VALUES (?,?,?,?)",
                (username, pw_hash, salt, role),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_name(username: str) -> Optional[dict]:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users() -> list[dict]:
    with db() as c:
        rows = c.execute(
            "SELECT id, username, role, created_at, is_active FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def set_user_active(user_id: int, active: bool):
    with db() as c:
        c.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))


def set_user_role(user_id: int, role: str):
    with db() as c:
        c.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def reset_password(user_id: int, new_password: str):
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(new_password, salt)
    with db() as c:
        c.execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                  (pw_hash, salt, user_id))


def delete_user(user_id: int):
    with db() as c:
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ── Sessions ─────────────────────────────────────────────────────────────────
def create_session(user_id: int, days: int = 7) -> str:
    from datetime import timedelta
    token = secrets.token_urlsafe(32)
    exp   = datetime.now() + timedelta(days=days)
    with db() as c:
        c.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)",
                  (token, user_id, exp))
    return token


def get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    with db() as c:
        row = c.execute("""
            SELECT u.* FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
        """, (token,)).fetchone()
        return dict(row) if row else None


def delete_session(token: str):
    with db() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def cleanup_expired_sessions():
    with db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")


# ── Files ────────────────────────────────────────────────────────────────────
def file_exists_for_user(user_id: int, file_hash: str) -> bool:
    with db() as c:
        row = c.execute(
            "SELECT 1 FROM files WHERE user_id = ? AND file_hash = ?",
            (user_id, file_hash),
        ).fetchone()
        return bool(row)


def add_file(user_id: int, filename: str, path: str, file_hash: str, size: int) -> Optional[int]:
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO files (user_id, filename, path, file_hash, size_bytes) VALUES (?,?,?,?,?)",
                (user_id, filename, path, file_hash, size),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # Duplicate


def list_files(user_id: Optional[int] = None) -> list[dict]:
    with db() as c:
        if user_id is None:
            rows = c.execute("""
                SELECT f.*, u.username FROM files f
                JOIN users u ON u.id = f.user_id
                ORDER BY f.uploaded_at DESC
            """).fetchall()
        else:
            rows = c.execute("""
                SELECT f.*, u.username FROM files f
                JOIN users u ON u.id = f.user_id
                WHERE f.user_id = ?
                ORDER BY f.uploaded_at DESC
            """, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def get_file(file_id: int) -> Optional[dict]:
    with db() as c:
        row = c.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None


def delete_file(file_id: int):
    with db() as c:
        c.execute("DELETE FROM files WHERE id = ?", (file_id,))


def clear_user_files(user_id: int):
    with db() as c:
        c.execute("DELETE FROM files WHERE user_id = ?", (user_id,))


# ── Chat logs ────────────────────────────────────────────────────────────────
def add_chat_log(user_id: int, model: str, question: str, answer: str,
                 rag_used: bool, duration_ms: int, status: str):
    with db() as c:
        c.execute("""
            INSERT INTO chat_logs (user_id, model, question, answer, rag_used, duration_ms, status)
            VALUES (?,?,?,?,?,?,?)
        """, (user_id, model, question, answer, 1 if rag_used else 0, duration_ms, status))


def list_chat_logs(user_id: Optional[int] = None, limit: int = 100) -> list[dict]:
    with db() as c:
        if user_id is None:
            rows = c.execute("""
                SELECT cl.*, u.username FROM chat_logs cl
                JOIN users u ON u.id = cl.user_id
                ORDER BY cl.created_at DESC LIMIT ?
            """, (limit,)).fetchall()
        else:
            rows = c.execute("""
                SELECT cl.*, u.username FROM chat_logs cl
                JOIN users u ON u.id = cl.user_id
                WHERE cl.user_id = ?
                ORDER BY cl.created_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


# ── Training jobs ────────────────────────────────────────────────────────────
def add_training_job(user_id: int, kind: str, status: str = "preparing") -> int:
    with db() as c:
        cur = c.execute(
            "INSERT INTO training_jobs (user_id, kind, status) VALUES (?,?,?)",
            (user_id, kind, status),
        )
        return cur.lastrowid


def update_training_job(job_id: int, status: str, error: Optional[str] = None):
    finished = "CURRENT_TIMESTAMP" if status in ("done", "error") else "NULL"
    with db() as c:
        c.execute(f"""
            UPDATE training_jobs
            SET status = ?, error = ?, finished_at = {finished}
            WHERE id = ?
        """, (status, error, job_id))


def get_user_settings(user_id: int) -> dict:
    with db() as c:
        row = c.execute(
            "SELECT system_prompt FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return {"system_prompt": row["system_prompt"] or ""}
        return {"system_prompt": ""}


def set_user_settings(user_id: int, system_prompt: str):
    with db() as c:
        c.execute("""
            INSERT INTO user_settings (user_id, system_prompt, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                system_prompt = excluded.system_prompt,
                updated_at    = CURRENT_TIMESTAMP
        """, (user_id, system_prompt))


def list_training_jobs(user_id: Optional[int] = None, limit: int = 50) -> list[dict]:
    with db() as c:
        if user_id is None:
            rows = c.execute("""
                SELECT tj.*, u.username FROM training_jobs tj
                JOIN users u ON u.id = tj.user_id
                ORDER BY tj.started_at DESC LIMIT ?
            """, (limit,)).fetchall()
        else:
            rows = c.execute("""
                SELECT tj.*, u.username FROM training_jobs tj
                JOIN users u ON u.id = tj.user_id
                WHERE tj.user_id = ?
                ORDER BY tj.started_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        return [dict(r) for r in rows]
