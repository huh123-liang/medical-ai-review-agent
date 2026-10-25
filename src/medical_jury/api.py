from __future__ import annotations

import hmac
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from .budget import Budget
from .config import Settings
from .domain import EvaluationRequest
from .store import Store
from .workflow import Engine


def create_app(settings: Settings | None = None, store: Store | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.prepare()
    store = store or Store(settings.root)
    engine = engine or Engine(settings, store)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jury-single-worker")
    token = store.admin_token()
    @asynccontextmanager
    async def lifespan(app):
        yield
        with store.connect() as db:
            unfinished = db.execute("SELECT id FROM jobs WHERE status IN ('Pending','Running')").fetchall()
        for row in unfinished:
            store.cancel(row["id"])
        pool.shutdown(wait=True, cancel_futures=True)
        with store.connect() as db:
            db.execute("UPDATE jobs SET status='Cancelled',stage='cancelled' WHERE status='Pending' AND cancel_requested=1")

    app = FastAPI(title="Medical AI Jury — independent research API", version="0.1.0", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    instance_id = uuid4().hex

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(request, exception):
        # Pydantic defaults can echo rejected input values, including accidentally pasted secrets.
        return JSONResponse({"error": "Invalid evaluation schema", "fields": [
            {"location": list(item["loc"]), "type": item["type"]} for item in exception.errors()
        ]}, status_code=422)

    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        # Loopback-only runtime; no CORS and no external access by default.
        origin = request.headers.get("origin")
        if origin:
            return JSONResponse({"error": "Browser cross-origin calls disabled; use platform backend."}, status_code=403)
        try:
            if int(request.headers.get("content-length", "0")) > 140000:
                return JSONResponse({"error": "Payload too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "Invalid length"}, status_code=400)
        chunks, length = [], 0
        async for chunk in request.stream():
            length += len(chunk)
            if length > 140000:
                return JSONResponse({"error": "Payload too large"}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    def admin(request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + token):
            raise HTTPException(status_code=401, detail="Local Admin token required.")

    @app.get("/health")
    def health():
        return {"status": "ok", "role": "standalone-single-agent", "clinical_validation": "NOT_VALIDATED",
                "instance_id": instance_id, "pid": os.getpid(), "parent_pid": os.getppid(), "project_root": str(settings.root)}

    @app.get("/v1/budget", dependencies=[Depends(admin)])
    def budget():
        return Budget(settings, store).total()

    @app.post("/v1/evaluations", dependencies=[Depends(admin)], status_code=202)
    def start(body: EvaluationRequest, request: Request):
        with store.connect() as db:
            queued = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('Pending','Running')").fetchone()[0]
        if queued >= 20:
            raise HTTPException(status_code=429, detail="Local queue is full.")
        if body.provider == "deepseek":
            try:
                settings.validate_paid()
            except ValueError:
                raise HTTPException(status_code=422, detail="Local credential/model/price configuration must be reviewed.") from None
        key = request.headers.get("idempotency-key")
        if key and (len(key) > 128 or not key.isascii()):
            raise HTTPException(status_code=422, detail="Invalid idempotency key.")
        try:
            job_id, created = store.create_once(body, key)
        except ValueError:
            raise HTTPException(status_code=409, detail="Idempotency key conflicts with previous request.") from None
        if created:
            pool.submit(engine.run, body, job_id)
        return {"job_id": job_id, "status": store.get_job(job_id)["status"], "created": created}

    def job_or_404(job_id):
        try:
            return store.get_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown job") from None

    @app.get("/v1/evaluations/{job_id}", dependencies=[Depends(admin)])
    def status(job_id: str):
        result = job_or_404(job_id)
        return {key: result[key] for key in ("id", "status", "stage", "report", "error_code", "created_at", "updated_at")}

    @app.post("/v1/evaluations/{job_id}/cancel", dependencies=[Depends(admin)])
    def cancel(job_id: str):
        job_or_404(job_id)
        return {"cancel_requested": store.cancel(job_id)}

    @app.post("/v1/evaluations/{job_id}/retry", dependencies=[Depends(admin)], status_code=202)
    def retry(job_id: str):
        previous = job_or_404(job_id)
        if previous["status"] not in {"Failed", "Cancelled", "Interrupted"}:
            raise HTTPException(status_code=409, detail="Only terminal unsuccessful jobs can be retried.")
        request = EvaluationRequest.model_validate(previous["request"])
        new_id = store.create_job(request)
        store.audit(new_id, "retry_of", {"original_job_id": job_id})
        pool.submit(engine.run, request, new_id)
        return {"job_id": new_id, "original_job_id": job_id}

    @app.get("/v1/evaluations/{job_id}/audit", dependencies=[Depends(admin)])
    def audit(job_id: str):
        job_or_404(job_id)
        return {"events": store.events(job_id)}

    app.state.jury_pool = pool
    app.state.jury_store = store
    return app
