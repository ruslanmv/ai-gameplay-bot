"""
deployment/control_backend.py

Control Backend API (Production-ready)
- Serves the frontend UI at "/"
- Manages NN and Transformer services via subprocess
- Provides real API endpoints consumed by the frontend
- Adds robust port/process handling + better error messages
- Avoids per-worker state issues by recommending WORKERS=1 for orchestration
"""

from __future__ import annotations

import atexit
import os
import sys
import subprocess
import signal
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Paths / Project root
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

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
    # Use 127.0.0.1 to avoid IPv6/host issues
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


def _try_kill_by_pid(pid: int) -> None:
    """Kill a PID (best effort)."""
    try:
        if _is_windows():
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def start_service(service_name: str, script_path: Path, port: int) -> bool:
    """
    Start a service via subprocess. Uses project root as cwd and sets PYTHONPATH
    so local imports resolve. Writes logs to logs/<service>.log.
    """
    global service_processes

    # If already tracked and running
    proc = service_processes.get(service_name)
    if proc is not None and proc.poll() is None:
        logger.info("%s service already running (pid=%s)", service_name, proc.pid)
        return True

    # If already responding, treat as running (even if not spawned by us)
    if is_service_running(port):
        logger.info("%s service already healthy on port %s", service_name, port)
        return True

    # If port is bound but not healthy, don't start another instance (would hit "connection in use")
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

        # Wait briefly and verify health
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
# Frontend routes (serve UI)
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # Serve the real UI at /
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:asset_path>", methods=["GET"])
def frontend_assets(asset_path: str):
    # Serve static assets from frontend folder
    target = FRONTEND_DIR / asset_path
    if target.exists() and target.is_file():
        return send_from_directory(str(FRONTEND_DIR), asset_path)
    # For SPA routes, serve index.html
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# -----------------------------------------------------------------------------
# API Endpoints
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
    """Return last ~200 lines of a service log to help debug start failures from the UI."""
    if service_name not in ("nn", "transformer"):
        return jsonify({"error": "invalid service"}), 400
    p = _service_log_path(service_name)
    if not p.exists():
        return jsonify({"log": "", "note": "no log yet"}), 200
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    return jsonify({"log": "\n".join(lines)}), 200


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
