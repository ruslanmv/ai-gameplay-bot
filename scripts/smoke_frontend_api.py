"""
Frontend API Smoke Test (production-safe)

What it tests (aligned to the Enterprise Gameplay Bot frontend contract):

System & Orchestration
- GET  /health
- GET  /api/health
- GET  /api/status
- POST /api/set_active_model
- POST /api/start_nn
- POST /api/stop_nn
- POST /api/start_transformer
- POST /api/stop_transformer

Inference Lab
- POST /api/test_predict (nn + transformer)
- POST /api/predict (routes to active model)

Model Repository
- GET    /api/models
- POST   /api/upload_model  (multipart with .pth + optional meta)
- POST   /api/load_model    (activate uploaded weights)
- DELETE /api/delete_model  (delete uploaded weights, not active)

Dojo (Capture & Training)
- POST /api/start_capture
- POST /api/ingest_frame
- POST /api/ingest_input
- POST /api/stop_capture
- POST /api/train_offline
- GET  /api/train_status/<job_id>

Analytics & Logs
- GET /api/metrics
- GET /api/service_log/<svc>

It also:
- Quick-trains NN + Transformer weights (fast)
- Spawns deploy_nn.py, deploy_transformer.py, control_backend.py
- Cleans up all processes at the end

Run:
  uv run python scripts/smoke_frontend_api.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

NN_WEIGHTS = PROJECT / "models/neural_network/nn_model_finetuned.pth"
TR_WEIGHTS = PROJECT / "models/transformer/transformer_model_finetuned.pth"

NN_URL = "http://127.0.0.1:5000"
TR_URL = "http://127.0.0.1:5001"
CB_URL = "http://127.0.0.1:8000"

TIMEOUT_START = 12.0


# -----------------------------
# Helpers
# -----------------------------
def _wait_health(url: str, timeout: float = TIMEOUT_START) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _terminate(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    finally:
        print(f"[CLEANUP] {name} stopped")


def _spawn(script: Path, env: dict) -> subprocess.Popen:
    preexec = os.setsid if os.name != "nt" else None
    return subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(PROJECT),
        env=env,
        preexec_fn=preexec,
    )


def _assert_status(resp: requests.Response, expected: int, label: str) -> Dict[str, Any]:
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text}

    if resp.status_code != expected:
        raise RuntimeError(
            f"{label} expected {expected} but got {resp.status_code}. Response: {body}"
        )
    return body


def _tiny_png_data_url(w: int = 16, h: int = 8, rgb: Tuple[int, int, int] = (0, 0, 0)) -> str:
    img = Image.new("RGB", (w, h), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _post_json(path: str, payload: dict, timeout: float = 8) -> requests.Response:
    return requests.post(f"{CB_URL}{path}", json=payload, timeout=timeout)


def _get(path: str, timeout: float = 8) -> requests.Response:
    return requests.get(f"{CB_URL}{path}", timeout=timeout)


# -----------------------------
# Quick training
# -----------------------------
def quick_train_nn(seconds: float = 2.0) -> None:
    from models.neural_network.nn_model import GameplayNN

    model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    steps = 0
    while time.time() - start < seconds:
        x = torch.randn(64, 128)
        y = torch.randint(0, 10, (64,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        steps += 1

    NN_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), NN_WEIGHTS)
    print(f"[OK] NN trained {steps} steps, saved -> {NN_WEIGHTS}")


def quick_train_transformer(seconds: float = 2.0) -> None:
    from models.transformer.transformer_model import GameplayTransformer

    model = GameplayTransformer(
        input_size=128,
        num_heads=4,
        hidden_size=64,
        num_layers=2,
        output_size=10,
    )
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    steps = 0
    while time.time() - start < seconds:
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        steps += 1

    TR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TR_WEIGHTS)
    print(f"[OK] Transformer trained {steps} steps, saved -> {TR_WEIGHTS}")


# -----------------------------
# API tests
# -----------------------------
def test_system_and_orchestration() -> None:
    print("\n== System & Orchestration ==")

    # legacy health
    r = _get("/health")
    _assert_status(r, 200, "GET /health")
    print("[OK] GET /health")

    r = _get("/api/health")
    _assert_status(r, 200, "GET /api/health")
    print("[OK] GET /api/health")

    r = _get("/api/status")
    body = _assert_status(r, 200, "GET /api/status")
    print("[OK] GET /api/status ->", body)

    # set active model
    r = _post_json("/api/set_active_model", {"model": "nn"})
    _assert_status(r, 200, "POST /api/set_active_model nn")
    print("[OK] POST /api/set_active_model nn")

    r = _post_json("/api/set_active_model", {"model": "transformer"})
    _assert_status(r, 200, "POST /api/set_active_model transformer")
    print("[OK] POST /api/set_active_model transformer")

    # start/stop endpoints should respond 200 (even if already running)
    r = _post_json("/api/start_nn", {})
    _assert_status(r, 200, "POST /api/start_nn")
    print("[OK] POST /api/start_nn")

    r = _post_json("/api/start_transformer", {})
    _assert_status(r, 200, "POST /api/start_transformer")
    print("[OK] POST /api/start_transformer")

    r = _post_json("/api/stop_transformer", {})
    _assert_status(r, 200, "POST /api/stop_transformer")
    print("[OK] POST /api/stop_transformer")

    r = _post_json("/api/start_transformer", {})
    _assert_status(r, 200, "POST /api/start_transformer (restart)")
    print("[OK] POST /api/start_transformer (restart)")

    r = _post_json("/api/stop_nn", {})
    _assert_status(r, 200, "POST /api/stop_nn")
    print("[OK] POST /api/stop_nn")

    r = _post_json("/api/start_nn", {})
    _assert_status(r, 200, "POST /api/start_nn (restart)")
    print("[OK] POST /api/start_nn (restart)")


def test_inference_lab() -> None:
    print("\n== Inference Lab ==")

    # /api/test_predict for both models
    r = _post_json("/api/test_predict", {"model": "nn"}, timeout=12)
    body = _assert_status(r, 200, "POST /api/test_predict nn")
    print("[OK] /api/test_predict nn ->", body)

    r = _post_json("/api/test_predict", {"model": "transformer"}, timeout=12)
    body = _assert_status(r, 200, "POST /api/test_predict transformer")
    print("[OK] /api/test_predict transformer ->", body)

    # /api/predict routed by active model
    # Set active nn and predict
    _assert_status(_post_json("/api/set_active_model", {"model": "nn"}), 200, "set_active nn")
    dummy = {"image": _tiny_png_data_url(16, 8, (10, 20, 30))}
    r = _post_json("/api/predict", dummy, timeout=12)
    body = _assert_status(r, 200, "POST /api/predict (active nn)")
    if "action" not in body:
        raise RuntimeError(f"/api/predict nn missing 'action': {body}")
    print("[OK] /api/predict (active nn) -> action:", body.get("action"))

    # Set active transformer and predict
    _assert_status(_post_json("/api/set_active_model", {"model": "transformer"}), 200, "set_active transformer")
    dummy = {"image": _tiny_png_data_url(16, 8, (40, 50, 60))}
    r = _post_json("/api/predict", dummy, timeout=12)
    body = _assert_status(r, 200, "POST /api/predict (active transformer)")
    if "action" not in body:
        raise RuntimeError(f"/api/predict transformer missing 'action': {body}")
    print("[OK] /api/predict (active transformer) -> action:", body.get("action"))


def test_model_repository() -> None:
    print("\n== Model Repository ==")

    # list models
    r = _get("/api/models")
    body = _assert_status(r, 200, "GET /api/models")
    models = body.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError(f"/api/models returned non-list: {body}")
    print(f"[OK] /api/models -> {len(models)} models")

    # upload a model (use NN weights as a test artifact, but upload as nn)
    if not NN_WEIGHTS.exists():
        raise RuntimeError(f"Missing weights to upload: {NN_WEIGHTS}")

    meta = {"uploaded_by": "smoke_frontend_api", "ts": int(time.time())}
    files = {
        "file": (f"smoke_upload_nn_{int(time.time())}.pth", NN_WEIGHTS.read_bytes(), "application/octet-stream"),
        # optional meta file (frontend may call it "meta")
        "meta": ("meta.json", json.dumps(meta).encode("utf-8"), "application/json"),
    }
    data = {"model_type": "nn"}

    r = requests.post(f"{CB_URL}/api/upload_model", files=files, data=data, timeout=20)
    body = _assert_status(r, 200, "POST /api/upload_model")
    uploaded_path = body.get("path")
    if not uploaded_path:
        raise RuntimeError(f"upload response missing path: {body}")
    uploaded_filename = Path(uploaded_path).name
    print("[OK] /api/upload_model ->", uploaded_path)

    # load the uploaded model (activate)
    r = _post_json("/api/load_model", {"model_type": "nn", "filename": uploaded_filename}, timeout=20)
    body = _assert_status(r, 200, "POST /api/load_model (nn)")
    print("[OK] /api/load_model (nn) ->", body)

    # verify nn still predicts after reload
    r = _post_json("/api/test_predict", {"model": "nn"}, timeout=15)
    _assert_status(r, 200, "POST /api/test_predict nn after load_model")
    print("[OK] /api/test_predict nn after load_model")

    # delete the uploaded file (should not be the active weights file; backend blocks deleting active)
    r = requests.delete(
        f"{CB_URL}/api/delete_model",
        json={"type": "nn", "filename": uploaded_filename},
        timeout=10,
    )
    # It might fail with 400 if you just activated that exact file as active and backend maps it to active path.
    # So we accept 200 OR 400 with "Refusing to delete active weights file".
    if r.status_code == 200:
        print("[OK] /api/delete_model uploaded file deleted")
    elif r.status_code == 400:
        try:
            b = r.json()
        except Exception:
            b = {"_raw": r.text}
        msg = (b.get("message") or "").lower()
        if "refusing to delete active" in msg:
            print("[OK] /api/delete_model correctly refused deleting active weights")
        else:
            raise RuntimeError(f"DELETE /api/delete_model unexpected 400: {b}")
    else:
        raise RuntimeError(f"DELETE /api/delete_model failed: {r.status_code} {r.text}")


def test_dojo_capture_and_training() -> None:
    print("\n== Dojo Capture & Training ==")

    # start capture
    r = _post_json("/api/start_capture", {"session_name": "smoke_session", "source": "screen"})
    body = _assert_status(r, 200, "POST /api/start_capture")
    session_id = body.get("session_id")
    if not session_id:
        raise RuntimeError(f"start_capture missing session_id: {body}")
    print("[OK] start_capture ->", session_id)

    # ingest a couple frames + inputs
    for i in range(2):
        ts = int(time.time() * 1000)
        frame = {"session_id": session_id, "image": _tiny_png_data_url(64, 32, (i * 50, 10, 10)), "timestamp": ts}
        r = _post_json("/api/ingest_frame", frame, timeout=12)
        _assert_status(r, 200, "POST /api/ingest_frame")
        print(f"[OK] ingest_frame {i+1}")

        keys = ["w"] if i == 0 else ["space"]
        inp = {"session_id": session_id, "keys": keys, "timestamp": ts}
        r = _post_json("/api/ingest_input", inp)
        _assert_status(r, 200, "POST /api/ingest_input")
        print(f"[OK] ingest_input {i+1}")

    # stop capture (zip)
    r = _post_json("/api/stop_capture", {"session_id": session_id}, timeout=20)
    body = _assert_status(r, 200, "POST /api/stop_capture")
    if not body.get("success"):
        raise RuntimeError(f"stop_capture did not succeed: {body}")
    print("[OK] stop_capture ->", {"dataset_path": body.get("dataset_path"), "zip": body.get("zip")})

    # train offline (keep epochs low to avoid slow runs)
    r = _post_json("/api/train_offline", {"dataset": session_id, "model_type": "nn", "epochs": 1}, timeout=10)
    body = _assert_status(r, 200, "POST /api/train_offline")
    job_id = body.get("job_id")
    if not job_id:
        raise RuntimeError(f"train_offline missing job_id: {body}")
    print("[OK] train_offline started ->", job_id)

    # poll job status briefly (don’t require completion; just verify endpoint works)
    deadline = time.time() + 10.0
    last = None
    while time.time() < deadline:
        r = _get(f"/api/train_status/{job_id}", timeout=6)
        body = _assert_status(r, 200, "GET /api/train_status/<job_id>")
        last = body
        if body.get("status") in ("completed", "failed"):
            break
        time.sleep(0.8)

    print("[OK] train_status ->", {"status": (last or {}).get("status"), "error": (last or {}).get("error")})


def test_metrics_and_logs() -> None:
    print("\n== Metrics & Logs ==")

    r = _get("/api/metrics")
    body = _assert_status(r, 200, "GET /api/metrics")
    for k in ("cpu", "ram", "rps"):
        if k not in body:
            raise RuntimeError(f"/api/metrics missing {k}: {body}")
    print("[OK] /api/metrics ->", body)

    # logs (best effort; may be empty but endpoint must work)
    r = _get("/api/service_log/nn")
    body = _assert_status(r, 200, "GET /api/service_log/nn")
    print("[OK] /api/service_log/nn (tail chars):", len((body.get("log") or "")))

    r = _get("/api/service_log/transformer")
    body = _assert_status(r, 200, "GET /api/service_log/transformer")
    print("[OK] /api/service_log/transformer (tail chars):", len((body.get("log") or "")))


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    print("== Frontend API Smoke: train + start services + exercise all UI endpoints ==")

    nn_proc = tr_proc = cb_proc = None

    try:
        # 1) Train models
        quick_train_nn(2.0)
        quick_train_transformer(2.0)

        # 2) Spawn services
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT)

        nn_proc = _spawn(PROJECT / "deployment/deploy_nn.py", env)
        tr_proc = _spawn(PROJECT / "deployment/deploy_transformer.py", env)
        cb_proc = _spawn(PROJECT / "deployment/control_backend.py", env)

        # 3) Health checks
        if not _wait_health(NN_URL):
            raise RuntimeError("NN service did not become healthy")
        if not _wait_health(TR_URL):
            raise RuntimeError("Transformer service did not become healthy")
        if not _wait_health(CB_URL):
            raise RuntimeError("Control backend did not become healthy")

        # 4) Run full suite
        test_system_and_orchestration()
        test_inference_lab()
        test_model_repository()
        test_dojo_capture_and_training()
        test_metrics_and_logs()

        print("\n[PASS] Frontend API smoke test succeeded.")
        print(f"UI available at: {CB_URL}/")
        return 0

    except Exception as e:
        print(f"\n[FAIL] Frontend API smoke test failed: {e}")
        return 1

    finally:
        _terminate(cb_proc, "control_backend")
        _terminate(tr_proc, "transformer_service")
        _terminate(nn_proc, "nn_service")


if __name__ == "__main__":
    raise SystemExit(main())
