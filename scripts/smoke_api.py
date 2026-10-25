"""Real localhost API smoke; child is hidden, owned, checked and stopped. No paid calls."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / ".runtime"
runtime.mkdir(exist_ok=True)
with socket.socket() as socket_probe:
    socket_probe.bind(("127.0.0.1", 0))
    port = socket_probe.getsockname()[1]
log = (runtime / "api-smoke.log").open("w", encoding="utf-8")
process = subprocess.Popen([sys.executable, "-m", "medical_jury.cli", "serve", "--port", str(port)],
    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
try:
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=5, trust_env=False) as client:
        health = {}
        for _ in range(200):
            if process.poll() is not None:
                raise RuntimeError("Owned API child exited; inspect local smoke log.")
            try:
                health = client.get(base + "/health").json()
                # Windows venv executables may launch the base Python as their child.
                owned_pid = process.pid in (health.get("pid"), health.get("parent_pid"))
                owned_root = Path(health.get("project_root", "")).resolve() == ROOT
                if owned_pid and owned_root and health.get("role") == "standalone-single-agent":
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Owned local API did not become ready: expected launcher={process.pid}; health={health}")
        headers = {"Authorization": "Bearer " + (runtime / "admin-token").read_text().strip(), "Idempotency-Key": "smoke-" + health["instance_id"]}
        before = client.get(base + '/v1/budget', headers=headers).json()
        assert client.get(base + "/v1/budget").status_code == 401
        body = json.loads((ROOT / "examples/synthetic-request.json").read_text())
        response = client.post(base + "/v1/evaluations", json=body, headers=headers)
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        for _ in range(100):
            result = client.get(base + "/v1/evaluations/" + job_id, headers=headers).json()
            if result["status"] in ("Completed", "Failed"):
                break
            time.sleep(0.05)
        assert result["status"] == "Completed"
        assert result["report"]["clinical_validation"] == "OFFLINE_MOCK_NOT_CLINICAL"
        repeat = client.post(base + "/v1/evaluations", json=body, headers=headers).json()
        assert repeat["job_id"] == job_id and repeat["created"] is False
        budget = client.get(base + "/v1/budget", headers=headers).json()
        assert budget['paid_attempts'] == before['paid_attempts']
        assert budget['accounted_cny'] == before['accounted_cny']
        summary = {"http_api": "PASS", "auth": "PASS", "idempotency": "PASS", "mock_job": job_id,
                   "paid_attempts": 0, "accounted_cny": 0,
                   "project_paid_attempts_unchanged": budget['paid_attempts'],
                   "project_accounted_cny_unchanged": budget['accounted_cny']}
        folder = ROOT / "reports"
        folder.mkdir(exist_ok=True)
        (folder / "api-smoke-verification.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log.close()
