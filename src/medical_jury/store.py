from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain import canonical, digest


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Cancelled(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "data/jury.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
              provider TEXT NOT NULL, mode TEXT NOT NULL, request_json TEXT NOT NULL,
              snapshot_json TEXT, report_json TEXT, error_code TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, event TEXT NOT NULL,
              detail_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS costs(
              id TEXT PRIMARY KEY, job_id TEXT NOT NULL, reserved REAL NOT NULL,
              accounted REAL NOT NULL, state TEXT NOT NULL, usage_json TEXT,
              created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rubrics(
              id TEXT PRIMARY KEY, text TEXT NOT NULL, level TEXT NOT NULL, task_type TEXT NOT NULL,
              case_id TEXT NOT NULL, group_id TEXT NOT NULL, split TEXT NOT NULL,
              approved INTEGER NOT NULL, mandatory INTEGER NOT NULL, data_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS idempotency(
              key_hash TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, job_id TEXT NOT NULL);
            """)

    def recover_interrupted(self):
        # Only call while holding the exclusive runner lock, never on read-only commands.
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='Interrupted',stage='interrupted',updated_at=? WHERE status IN ('Running','Pending')", (timestamp(),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_job(self, request: Any) -> str:
        return self.create_once(request, None)[0]

    def create_once(self, request: Any, idempotency_key: str | None) -> tuple[str, bool]:
        job_id = "JURY-" + uuid4().hex[:16]
        now = timestamp()
        payload_hash = digest(request.model_dump())
        key_hash = digest(idempotency_key) if idempotency_key else None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key_hash:
                old = db.execute("SELECT payload_hash,job_id FROM idempotency WHERE key_hash=?", (key_hash,)).fetchone()
                if old:
                    if old["payload_hash"] != payload_hash:
                        raise ValueError("Idempotency key was reused for a different request.")
                    return old["job_id"], False
            db.execute("INSERT INTO jobs(id,status,stage,provider,mode,request_json,created_at,updated_at) VALUES(?,'Pending','queued',?,?,?,?,?)",
                       (job_id, request.provider, request.mode, canonical(request.model_dump()), now, now))
            if key_hash:
                db.execute("INSERT INTO idempotency VALUES(?,?,?)", (key_hash, payload_hash, job_id))
        self.audit(job_id, "queued", {})
        return job_id, True

    def get_job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        item = dict(row)
        for key in ("request_json", "snapshot_json", "report_json"):
            raw = item.pop(key)
            item[key.removesuffix("_json")] = json.loads(raw) if raw else None
        return item

    def update(self, job_id: str, **values):
        allowed = {"status", "stage", "snapshot_json", "report_json", "error_code", "cancel_requested"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid job state fields.")
        values["updated_at"] = timestamp()
        sql = ",".join(f"{k}=?" for k in values)
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {sql} WHERE id=?", (*values.values(), job_id))

    def check_cancelled(self, job_id: str):
        if self.get_job(job_id)["cancel_requested"]:
            raise Cancelled("Cancelled by local operator.")

    def cancel(self, job_id: str):
        job = self.get_job(job_id)
        if job["status"] in {"Completed", "Failed", "Cancelled", "Interrupted"}:
            return False
        self.update(job_id, cancel_requested=1)
        self.audit(job_id, "cancel_requested", {})
        return True

    def audit(self, job_id: str, event: str, details: dict):
        with self.connect() as db:
            db.execute("INSERT INTO audit(job_id,event,detail_json,created_at) VALUES(?,?,?,?)",
                       (job_id, event, canonical(details), timestamp()))

    def events(self, job_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT seq,event,detail_json,created_at FROM audit WHERE job_id=? ORDER BY seq", (job_id,)).fetchall()
        return [dict(row) for row in rows]

    def admin_token(self) -> str:
        file = self.root / ".runtime/admin-token"
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            try:
                with file.open("x", encoding="utf-8") as stream:
                    stream.write(secrets.token_urlsafe(32))
            except FileExistsError:
                pass
        return file.read_text(encoding="utf-8").strip()
