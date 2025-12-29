"""
deployment/control_backend.py

Control Backend API (Production-ready)
- Serves the frontend UI at "/"
- Manages NN and Transformer services via subprocess
- Provides real API endpoints consumed by the frontend
- Adds robust port/process handling + better error messages
- Avoids per-worker state issues by recommending WORKERS=1 for orchestration

NEW (Model management):
- GET  /api/models
- POST /api/upload_model   (multipart/form-data)
- POST /api/load_model     (JSON)
"""

from __future__ import annotations

import atexit
import os
import sys
import subprocess
import signal
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Paths / Project root
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Model directories
MODELS_DIR = ROOT_DIR / "models"
NN_MODELS_DIR = MODELS_DIR / "neural_network"
TR_MODELS_DIR = MODELS_DIR / "transformer"
NN_UPLOADS_DIR = NN_MODELS_DIR / "uploads"
TR_UPLOADS_DIR = TR_MODELS_DIR / "uploads"
NN_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TR_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Default "active" weights (services load these by default)
NN_ACTIVE_WEIGHTS = NN_MODELS_DIR / "nn_model_finetuned.pth"
TR_ACTIVE_WEIGHTS = TR_MODELS_DIR / "transformer_model_finetuned.pth"

ALLOWED_MODEL_EXTENSIONS = {".pth"}  # keep strict for safety

# Service script absolute paths
NN_PORT = int(os.environ.get("NN_PORT", "5000"))
TRANSFORMER_PORT = int(os.environ.get("TRANSFORMER_PORT", "5001"))
NN_SCRIPT = ROOT_DIR / "deployment" / "deploy_nn.py"
TRANSFORMER_SCRIPT = ROOT_DIR / "deployment" / "deploy_transformer.py"

# Backend host/port (useful for non-default)
BACKEND_HOST = os.environ.get("HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("PORT", "8000"))

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("control_backend")

# -----------------------------------------------------------------------------
# Flask app (serve frontend statics)
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# -----------------------------------------------------------------------------
# Global state (IMPORTANT):
# If you run gunicorn with multiple workers, each worker has its own state.
# For orchestrating subprocess services reliably, run gunicorn with WORKERS=1.
# -----------------------------------------------------------------------------
service_processes: Dict[str, Optional[subprocess.Popen]] = {"nn": None, "transformer": None}
active_model = "nn"


def _service_log_path(service_name: str) -> Path:
    return LOG_DIR / f"{service_name}.log"


def _is_windows() -> bool:
    return os.name == "nt"


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def is_service_running(port: int) -> bool:
    """Check if a service responds on /health."""
    try:
        resp = requests.get(_health_url(port), timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


def _port_in_use(port: int) -> bool:
    """Best-effort check if TCP port is bound (even if /health isn't responding)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_service(service_name: str, script_path: Path, port: int) -> bool:
    """
    Start a service via subprocess. Uses project root as cwd and sets PYTHONPATH
    so local imports resolve. Writes logs to logs/<service>.log.
    """
    global service_processes

    proc = service_processes.get(service_name)
    if proc is not None and proc.poll() is None:
        logger.info("%s service already running (pid=%s)", service_name, proc.pid)
        return True

    if is_service_running(port):
        logger.info("%s service already healthy on port %s", service_name, port)
        return True

    if _port_in_use(port):
        logger.error(
            "%s port %s is already in use but service is not healthy. "
            "Stop the process using that port (or fix its /health).",
            service_name,
            port,
        )
        return False

    if not script_path.exists():
        logger.error("Service script missing: %s", script_path)
        return False

    logger.info("Starting %s service on port %s using %s", service_name, port, script_path)

    log_path = _service_log_path(service_name)
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"

    try:
        preexec = os.setsid if (not _is_windows()) else None

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=log_f,
            stderr=log_f,
            preexec_fn=preexec,
        )
        service_processes[service_name] = proc

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.error("%s exited early. Check %s", service_name, log_path)
                return False
            if is_service_running(port):
                logger.info("%s started OK (pid=%s). Logs: %s", service_name, proc.pid, log_path)
                return True
            time.sleep(0.4)

        logger.error("%s did not become healthy in time. Check %s", service_name, log_path)
        return False

    except Exception as e:
        logger.exception("Error starting %s: %s", service_name, e)
        return False


def stop_service(service_name: str) -> bool:
    """Stop a tracked subprocess (best effort)."""
    global service_processes

    proc = service_processes.get(service_name)
    if proc is None or proc.poll() is not None:
        service_processes[service_name] = None
        logger.info("%s is not running", service_name)
        return True

    logger.info("Stopping %s (pid=%s)...", service_name, proc.pid)

    try:
        if not _is_windows():
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()

        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            if not _is_windows():
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()

        service_processes[service_name] = None
        logger.info("%s stopped", service_name)
        return True

    except Exception as e:
        logger.exception("Error stopping %s: %s", service_name, e)
        return False


# -----------------------------------------------------------------------------
# Model management helpers
# -----------------------------------------------------------------------------
def _model_dir_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_MODELS_DIR
    if model_type == "transformer":
        return TR_MODELS_DIR
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _uploads_dir_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_UPLOADS_DIR
    if model_type == "transformer":
        return TR_UPLOADS_DIR
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _active_weights_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_ACTIVE_WEIGHTS
    if model_type == "transformer":
        return TR_ACTIVE_WEIGHTS
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _safe_resolve_within_root(rel_path: str) -> Path:
    """
    Resolve a user-supplied relative path safely within ROOT_DIR.
    Reject path traversal.
    """
    candidate = (ROOT_DIR / rel_path).resolve()
    root = ROOT_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid path (outside project root).")
    return candidate


def _file_meta(p: Path, model_type: str) -> Dict[str, Any]:
    st = p.stat()
    return {
        "id": _rel(p),  # stable identifier the frontend can send back
        "name": p.name,
        "model_type": model_type,
        "bytes": int(st.st_size),
        "modified": int(st.st_mtime),
        "is_active": p.resolve() == _active_weights_for(model_type).resolve(),
        "location": "uploads" if "uploads" in p.parts else "models",
    }


def _list_models(model_type: str) -> List[Dict[str, Any]]:
    base = _model_dir_for(model_type)
    if not base.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(base.rglob("*"), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
            continue
        out.append(_file_meta(p, model_type))
    return out


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _restart_service_for(model_type: str) -> bool:
    if model_type == "nn":
        stop_service("nn")
        return start_service("nn", NN_SCRIPT, NN_PORT)
    stop_service("transformer")
    return start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)


# -----------------------------------------------------------------------------
# Frontend routes (serve UI)
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:asset_path>", methods=["GET"])
def frontend_assets(asset_path: str):
    target = FRONTEND_DIR / asset_path
    if target.exists() and target.is_file():
        return send_from_directory(str(FRONTEND_DIR), asset_path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# -----------------------------------------------------------------------------
# API Endpoints (existing)
# -----------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify(
        {
            "nn_running": is_service_running(NN_PORT),
            "transformer_running": is_service_running(TRANSFORMER_PORT),
            "active_model": active_model,
            "timestamp": time.time(),
        }
    )


@app.route("/api/start_nn", methods=["POST"])
def start_nn():
    ok = start_service("nn", NN_SCRIPT, NN_PORT)
    return jsonify({"success": ok, "message": "NN started" if ok else "NN failed to start"}), (200 if ok else 500)


@app.route("/api/stop_nn", methods=["POST"])
def stop_nn():
    ok = stop_service("nn")
    return jsonify({"success": ok, "message": "NN stopped" if ok else "NN failed to stop"}), (200 if ok else 500)


@app.route("/api/start_transformer", methods=["POST"])
def start_transformer():
    ok = start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
    return jsonify({"success": ok, "message": "Transformer started" if ok else "Transformer failed to start"}), (
        200 if ok else 500
    )


@app.route("/api/stop_transformer", methods=["POST"])
def stop_transformer():
    ok = stop_service("transformer")
    return jsonify({"success": ok, "message": "Transformer stopped" if ok else "Transformer failed to stop"}), (
        200 if ok else 500
    )


@app.route("/api/set_active_model", methods=["POST"])
def set_active_model():
    global active_model
    data = request.get_json(silent=True) or {}
    model = data.get("model", "nn")

    if model not in ("nn", "transformer"):
        return jsonify({"success": False, "message": 'Invalid model. Must be "nn" or "transformer"'}), 400

    active_model = model
    logger.info("Active model set to: %s", active_model)
    return jsonify({"success": True, "active_model": active_model, "message": f"Active model set to {active_model}"})


@app.route("/api/test_predict", methods=["POST"])
def test_predict():
    data = request.get_json(silent=True) or {}
    model = data.get("model", "nn")

    test_state = np.random.rand(128).tolist()
    port = NN_PORT if model == "nn" else TRANSFORMER_PORT

    try:
        resp = requests.post(f"http://127.0.0.1:{port}/predict", json={"state": test_state}, timeout=6)
        if resp.status_code == 200:
            payload = resp.json()
            return jsonify({"success": True, "action": payload.get("action", "unknown"), "model": model})
        return jsonify({"success": False, "message": f"Predict service error: {resp.status_code}"}), 500
    except requests.RequestException:
        return jsonify({"success": False, "message": f"Failed to connect to {model}. Start it first."}), 500


@app.route("/api/service_log/<service_name>", methods=["GET"])
def service_log(service_name: str):
    if service_name not in ("nn", "transformer"):
        return jsonify({"error": "invalid service"}), 400
    p = _service_log_path(service_name)
    if not p.exists():
        return jsonify({"log": "", "note": "no log yet"}), 200
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    return jsonify({"log": "\n".join(lines)}), 200


# -----------------------------------------------------------------------------
# NEW: Model management API
# -----------------------------------------------------------------------------
@app.route("/api/models", methods=["GET"])
def api_models():
    """
    Returns available model files and which ones are "active".
    The frontend can use `id` in /api/load_model.
    """
    return jsonify(
        {
            "nn": _list_models("nn"),
            "transformer": _list_models("transformer"),
            "active": {
                "nn": _rel(NN_ACTIVE_WEIGHTS) if NN_ACTIVE_WEIGHTS.exists() else None,
                "transformer": _rel(TR_ACTIVE_WEIGHTS) if TR_ACTIVE_WEIGHTS.exists() else None,
            },
            "allowed_extensions": sorted(ALLOWED_MODEL_EXTENSIONS),
            "timestamp": time.time(),
        }
    )


@app.route("/api/upload_model", methods=["POST"])
def api_upload_model():
    """
    Upload a model weights file (multipart/form-data).
    Fields:
      - file: required (the .pth file)
      - model_type: optional ("nn" or "transformer") — if omitted we'll try to infer.
    Stores into:
      models/<type>/uploads/<timestamp>__<filename>.pth
    """
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Missing file field (multipart name must be 'file')."}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"success": False, "message": "Empty upload."}), 400

    model_type = (request.form.get("model_type") or "").strip().lower()
    filename = secure_filename(f.filename)

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify(
            {
                "success": False,
                "message": f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_MODEL_EXTENSIONS)}",
            }
        ), 400

    if model_type not in ("nn", "transformer"):
        # Infer best-effort by filename
        low = filename.lower()
        model_type = "transformer" if ("transformer" in low or "tr_" in low) else "nn"

    dest_dir = _uploads_dir_for(model_type)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"{stamp}__{filename}"

    try:
        f.save(dest)
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        return jsonify({"success": False, "message": "Failed to save uploaded file."}), 500

    meta = _file_meta(dest, model_type)
    logger.info("Uploaded model: %s", meta["id"])
    return jsonify({"success": True, "model": meta}), 200


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """
    Select a model file and make it active.
    JSON body:
      {
        "model_type": "nn" | "transformer",
        "model_id": "models/neural_network/uploads/....pth",   # from GET /api/models
        "restart_service": true  # optional, default true
      }

    Behavior:
      - Copies the selected file into the default active weights path:
          nn          -> models/neural_network/nn_model_finetuned.pth
          transformer -> models/transformer/transformer_model_finetuned.pth
      - Optionally restarts the corresponding service so the new weights load.
    """
    data = request.get_json(silent=True) or {}
    model_type = (data.get("model_type") or "").strip().lower()
    model_id = (data.get("model_id") or "").strip()
    restart = bool(data.get("restart_service", True))

    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'."}), 400
    if not model_id:
        return jsonify({"success": False, "message": "model_id is required (use GET /api/models)."}), 400

    try:
        src = _safe_resolve_within_root(model_id)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    if not src.exists() or not src.is_file():
        return jsonify({"success": False, "message": "model_id not found on disk."}), 404
    if src.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify({"success": False, "message": "Invalid model file type."}), 400

    # Enforce that the file actually belongs to the right model directory (nn vs transformer)
    expected_root = _model_dir_for(model_type).resolve()
    if expected_root not in src.resolve().parents and src.resolve() != expected_root:
        return jsonify(
            {
                "success": False,
                "message": f"model_id must be inside {_rel(expected_root)} for model_type='{model_type}'.",
            }
        ), 400

    dst = _active_weights_for(model_type)

    try:
        _atomic_copy(src, dst)
        logger.info("Loaded model for %s: %s -> %s", model_type, _rel(src), _rel(dst))
    except Exception as e:
        logger.exception("Failed to activate model: %s", e)
        return jsonify({"success": False, "message": "Failed to activate model (copy failed)."}), 500

    restarted = None
    if restart:
        restarted = _restart_service_for(model_type)

    return jsonify(
        {
            "success": True,
            "message": f"Active {model_type} weights updated.",
            "active_weights": _rel(dst),
            "source": _rel(src),
            "service_restarted": restarted,
        }
    ), 200


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200


def cleanup():
    logger.info("Shutting down control backend...")
    stop_service("nn")
    stop_service("transformer")


atexit.register(cleanup)

if __name__ == "__main__":
    logger.info("Starting Control Backend API on %s:%s ...", BACKEND_HOST, BACKEND_PORT)
    logger.info("UI: http://localhost:%s/", BACKEND_PORT)
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)
