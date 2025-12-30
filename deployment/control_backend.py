"""
deployment/control_backend.py

Enterprise Gameplay Bot - Control Backend (Flask)

Production-ready fixes included:
- API contracts aligned with the provided frontend index.html:
  - GET /api/models -> {models:[{filename,type,size_bytes,mtime_epoch,meta?,path?},...]}
  - POST /api/upload_model accepts:
      - file (.pth)            [REQUIRED]   (frontend sends "file")
      - model_type / type      [OPTIONAL]   (frontend sends "model_type")
      - meta file              [OPTIONAL]   (accepts "meta" or "meta_file")
      - meta json string       [OPTIONAL]   ("meta_json" or "meta")
  - POST /api/load_model expects {model_type, filename}

- Dojo endpoints tolerant to UI calls missing session_id:
  - /api/ingest_frame, /api/ingest_input, /api/stop_capture fallback to last session

- Robust service orchestration:
  - detects port collisions
  - attempts to identify/kill stale processes started from this repo (deploy_nn.py / deploy_transformer.py)
  - starts services with proper PYTHONPATH
  - keeps log file handles open
  - better error messages for frontend

NOTE:
- For subprocess orchestration reliability, run a single worker (WORKERS=1) if using gunicorn.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename


# ---------------------------------------------------------------------
# Paths / Project root
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

FRONTEND_DIR = ROOT_DIR / "frontend"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = ROOT_DIR / "models"
NN_MODELS_DIR = MODELS_DIR / "neural_network"
TR_MODELS_DIR = MODELS_DIR / "transformer"
NN_UPLOADS_DIR = NN_MODELS_DIR / "uploads"
TR_UPLOADS_DIR = TR_MODELS_DIR / "uploads"
NN_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TR_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

NN_ACTIVE_WEIGHTS = NN_MODELS_DIR / "nn_model_finetuned.pth"
TR_ACTIVE_WEIGHTS = TR_MODELS_DIR / "transformer_model_finetuned.pth"

ALLOWED_MODEL_EXTENSIONS = {".pth"}

NN_PORT = int(os.environ.get("NN_PORT", "5000"))
TRANSFORMER_PORT = int(os.environ.get("TRANSFORMER_PORT", "5001"))

NN_SCRIPT = ROOT_DIR / "deployment" / "deploy_nn.py"
TRANSFORMER_SCRIPT = ROOT_DIR / "deployment" / "deploy_transformer.py"

BACKEND_HOST = os.environ.get("HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("PORT", "8000"))

START_TIME = time.time()


# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ---------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------
service_processes: Dict[str, Optional[subprocess.Popen]] = {"nn": None, "transformer": None}
service_logs: Dict[str, Optional[io.TextIOWrapper]] = {"nn": None, "transformer": None}

active_model = "nn"

_last_session_id: str | None = None

_req_times = deque(maxlen=5000)
_req_lock = Lock()

_sessions_lock = Lock()
_sessions: Dict[str, Dict[str, Any]] = {}

_jobs_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _is_windows() -> bool:
    return os.name == "nt"


def _service_log_path(service_name: str) -> Path:
    return LOG_DIR / f"{service_name}.log"


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _predict_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/predict"


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex(("127.0.0.1", port)) == 0


def is_service_running(port: int) -> bool:
    try:
        r = requests.get(_health_url(port), timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _open_service_log(service_name: str) -> io.TextIOWrapper:
    p = _service_log_path(service_name)
    return open(p, "a", buffering=1, encoding="utf-8")  # line-buffered


def _record_request_for_rps() -> None:
    now = time.time()
    with _req_lock:
        _req_times.append(now)


def _compute_rps(window_sec: int = 5) -> float:
    now = time.time()
    cutoff = now - float(window_sec)
    with _req_lock:
        n = 0
        for t in reversed(_req_times):
            if t < cutoff:
                break
            n += 1
    return n / float(window_sec)


def _read_json_sidecar(pth_path: Path) -> Optional[dict]:
    sidecar = pth_path.with_suffix(pth_path.suffix + ".json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _file_meta_frontend(p: Path, model_type: str) -> Dict[str, Any]:
    st = p.stat()
    return {
        "filename": p.name,
        "type": "nn" if model_type == "nn" else "transformer",
        "size_bytes": int(st.st_size),
        "mtime_epoch": int(st.st_mtime),
        "meta": _read_json_sidecar(p),
        "path": str(p.resolve().relative_to(ROOT_DIR.resolve()).as_posix()),
    }


def _model_dir_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_MODELS_DIR
    if model_type in ("transformer", "tf"):
        return TR_MODELS_DIR
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _uploads_dir_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_UPLOADS_DIR
    if model_type in ("transformer", "tf"):
        return TR_UPLOADS_DIR
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _active_weights_for(model_type: str) -> Path:
    if model_type == "nn":
        return NN_ACTIVE_WEIGHTS
    if model_type in ("transformer", "tf"):
        return TR_ACTIVE_WEIGHTS
    raise ValueError("model_type must be 'nn' or 'transformer'")


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _list_models_flat() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in ("nn", "transformer"):
        base = _model_dir_for(t)
        if not base.exists():
            continue
        for p in base.rglob("*.pth"):
            if p.is_file():
                out.append(_file_meta_frontend(p, t))
    out.sort(key=lambda d: (d.get("type", ""), d.get("mtime_epoch", 0)), reverse=True)
    return out


def _resolve_model_file(model_type: str, filename: str) -> Optional[Path]:
    name = Path(filename).name  # basename only (no traversal)
    if not name.lower().endswith(".pth"):
        return None
    base = _model_dir_for(model_type)
    candidates = [p for p in base.rglob(name) if p.is_file() and p.suffix.lower() in ALLOWED_MODEL_EXTENSIONS]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _try_kill_process_holding_port(port: int) -> Tuple[bool, str]:
    """
    If port is in use, attempt to find and kill the process ONLY if it looks like
    one of our python services (deploy_nn.py / deploy_transformer.py).
    """
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                conns = proc.net_connections(kind="inet")
                for c in conns:
                    if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                        cmd = " ".join(proc.info.get("cmdline") or [])
                        if "deploy_nn.py" in cmd or "deploy_transformer.py" in cmd:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except Exception:
                                proc.kill()
                            return True, f"killed stale process pid={proc.pid} on port {port}"
                        return False, f"port {port} held by non-target process pid={proc.pid}"
            except Exception:
                continue
    except Exception as e:
        return False, f"psutil scan failed: {e}"
    return False, f"port {port} in use but process not found"


def start_service(service_name: str, script_path: Path, port: int) -> Tuple[bool, Optional[int], str]:
    """
    Returns (ok, pid, message)
    """
    if service_name not in ("nn", "transformer"):
        return False, None, "invalid service_name"

    proc = service_processes.get(service_name)
    if proc is not None and proc.poll() is None:
        return True, proc.pid, "already running (tracked)"

    if is_service_running(port):
        return True, None, "already running (external)"

    if not script_path.exists():
        return False, None, f"script not found: {script_path}"

    if _port_in_use(port):
        killed, msg = _try_kill_process_holding_port(port)
        if not killed:
            return False, None, f"port {port} is in use ({msg})"
        time.sleep(0.3)

    try:
        lf = service_logs.get(service_name)
        if lf is None or lf.closed:
            service_logs[service_name] = _open_service_log(service_name)
        lf = service_logs[service_name]
    except Exception as e:
        return False, None, f"failed to open log file: {e}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else str(ROOT_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        preexec = os.setsid if (not _is_windows()) else None
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=lf,
            stderr=lf,
            preexec_fn=preexec,
        )
        service_processes[service_name] = proc

        deadline = time.time() + 12.0
        while time.time() < deadline:
            if proc.poll() is not None:
                return False, None, f"{service_name} exited early (rc={proc.returncode})"
            if is_service_running(port):
                return True, proc.pid, "started"
            time.sleep(0.35)

        return False, None, f"{service_name} did not become healthy in time"
    except Exception as e:
        return False, None, f"failed to spawn: {e}"


def stop_service(service_name: str) -> Tuple[bool, str]:
    proc = service_processes.get(service_name)
    if proc is None or proc.poll() is not None:
        service_processes[service_name] = None
        return True, "already stopped"

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
        return True, "stopped"
    except Exception as e:
        return False, f"stop failed: {e}"


def _png_dataurl_dummy() -> str:
    img = Image.new("RGB", (16, 8), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _decode_dataurl_to_jpg_bytes(data: str) -> bytes:
    """
    Accepts data URL or base64. Re-encodes to JPEG bytes for consistent storage.
    """
    if not isinstance(data, str) or not data.strip():
        raise ValueError("image must be a non-empty string")

    s = data.strip()
    if s.startswith("data:"):
        parts = s.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URL format")
        s = parts[1]

    raw = base64.b64decode(s, validate=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _keys_to_action(keys: List[str]) -> str:
    s = set((k or "").lower() for k in keys if isinstance(k, str))
    if "w" in s:
        return "move_forward"
    if "s" in s:
        return "move_backward"
    if "a" in s:
        return "turn_left"
    if "d" in s:
        return "turn_right"
    if "space" in s:
        return "jump"
    if "e" in s:
        return "interact"
    if "f" in s:
        return "use_item"
    if "tab" in s:
        return "open_inventory"
    if "q" in s:
        return "cast_spell"
    return "move_forward"


def _build_dataset_csv_from_session(session_id: str) -> Path:
    """
    Builds a CSV dataset with 128 features + action_index.
    Uses deployment.feature_extractor.image_to_features_128.
    """
    from deployment.feature_extractor import image_to_features_128

    sess_dir = RAW_DIR / session_id
    frames_dir = sess_dir / "frames"
    inputs_path = sess_dir / "inputs.jsonl"
    out_csv = PROCESSED_DIR / f"{session_id}__dataset.csv"

    if not frames_dir.exists():
        raise RuntimeError("frames directory missing")
    if not inputs_path.exists():
        raise RuntimeError("inputs.jsonl missing")

    inputs: Dict[int, List[str]] = {}
    for line in inputs_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
            ts = int(obj.get("timestamp", 0))
            keys = obj.get("keys", [])
            if ts:
                inputs[ts] = keys if isinstance(keys, list) else []
        except Exception:
            continue

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    rows: List[Dict[str, Any]] = []
    input_ts_sorted = sorted(inputs.keys())

    def nearest_ts(target: int) -> Optional[int]:
        if not input_ts_sorted:
            return None
        import bisect

        i = bisect.bisect_left(input_ts_sorted, target)
        candidates = []
        if 0 <= i < len(input_ts_sorted):
            candidates.append(input_ts_sorted[i])
        if 0 <= i - 1 < len(input_ts_sorted):
            candidates.append(input_ts_sorted[i - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda t: abs(t - target))
        return best if abs(best - target) <= 250 else None

    action_map = {
        "move_forward": 0,
        "move_backward": 1,
        "turn_left": 2,
        "turn_right": 3,
        "attack": 4,
        "jump": 5,
        "interact": 6,
        "use_item": 7,
        "open_inventory": 8,
        "cast_spell": 9,
    }

    for fp in frame_files:
        try:
            ts = int(fp.stem.split("_", 1)[1])
        except Exception:
            continue

        ts2 = ts if ts in inputs else nearest_ts(ts)
        keys = inputs.get(ts2, []) if ts2 is not None else []
        action_str = _keys_to_action(keys)
        action_idx = action_map.get(action_str, 0)

        b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
        feats = image_to_features_128(b64)

        row: Dict[str, Any] = {f"f{i}": feats[i] for i in range(128)}
        row["action"] = action_idx
        row["timestamp"] = ts
        rows.append(row)

    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [f"f{i}" for i in range(128)] + ["action", "timestamp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_csv


def _run_training_job(job_id: str, dataset_session: str, model_type: str, epochs: int) -> None:
    def set_job(**kw):
        with _jobs_lock:
            _jobs[job_id].update(**kw)

    try:
        set_job(status="building_dataset")
        csv_path = _build_dataset_csv_from_session(dataset_session)
        set_job(status="training", dataset_csv=str(csv_path))

        if model_type == "nn":
            expected_csv = PROCESSED_DIR / "nn_dataset.csv"
            shutil.copy2(csv_path, expected_csv)
            cmd = [sys.executable, str(ROOT_DIR / "models" / "neural_network" / "nn_training.py")]
            produced = NN_ACTIVE_WEIGHTS
            updir = NN_UPLOADS_DIR
        else:
            expected_csv = PROCESSED_DIR / "transformer_dataset.csv"
            shutil.copy2(csv_path, expected_csv)
            cmd = [sys.executable, str(ROOT_DIR / "models" / "transformer" / "transformer_training.py")]
            produced = TR_ACTIVE_WEIGHTS
            updir = TR_UPLOADS_DIR

        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log_lines: List[str] = []
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            if line:
                log_lines.append(line.rstrip("\n"))
                if len(log_lines) > 300:
                    log_lines = log_lines[-300:]
                set_job(log="\n".join(log_lines))

        rc = proc.wait()
        if rc != 0:
            set_job(status="failed", error=f"training script exited with code {rc}")
            return

        set_job(status="saving_model")

        if not produced.exists():
            set_job(status="failed", error=f"Expected weights not found at {produced}")
            return

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"{stamp}__{model_type}__trained_from_{dataset_session}.pth"
        out_path = updir / out_name
        shutil.copy2(produced, out_path)

        meta = {
            "trained_from_session": dataset_session,
            "epochs": epochs,
            "created": int(time.time()),
            "source_csv": str(csv_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()),
        }
        out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        set_job(status="completed", output_model=str(out_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()))
    except Exception as e:
        set_job(status="failed", error=str(e))


# ---------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:asset_path>", methods=["GET"])
def frontend_assets(asset_path: str):
    target = FRONTEND_DIR / asset_path
    if target.exists() and target.is_file():
        return send_from_directory(str(FRONTEND_DIR), asset_path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ---------------------------------------------------------------------
# System & Orchestration APIs
# ---------------------------------------------------------------------
@app.before_request
def _track_rps_before():
    if request.path.startswith("/api/"):
        _record_request_for_rps()


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "healthy", "uptime": int(time.time() - START_TIME)}), 200


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(
        {
            "nn_running": is_service_running(NN_PORT),
            "transformer_running": is_service_running(TRANSFORMER_PORT),
            "active_model": active_model,
            "timestamp": time.time(),
        }
    ), 200


@app.route("/api/start_nn", methods=["POST"])
def api_start_nn():
    ok, pid, msg = start_service("nn", NN_SCRIPT, NN_PORT)
    return jsonify({"success": ok, "pid": pid, "message": msg}), (200 if ok else 500)


@app.route("/api/stop_nn", methods=["POST"])
def api_stop_nn():
    ok, msg = stop_service("nn")
    return jsonify({"success": ok, "message": msg}), (200 if ok else 500)


@app.route("/api/start_transformer", methods=["POST"])
def api_start_transformer():
    ok, pid, msg = start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
    return jsonify({"success": ok, "pid": pid, "message": msg}), (200 if ok else 500)


@app.route("/api/stop_transformer", methods=["POST"])
def api_stop_transformer():
    ok, msg = stop_service("transformer")
    return jsonify({"success": ok, "message": msg}), (200 if ok else 500)


@app.route("/api/set_active_model", methods=["POST"])
def api_set_active_model():
    global active_model
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "").strip().lower()
    if model not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model must be 'nn' or 'transformer'"}), 400
    active_model = model
    return jsonify({"active_model": active_model}), 200


# ---------------------------------------------------------------------
# Model Repository APIs
# ---------------------------------------------------------------------
@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({"models": _list_models_flat()}), 200


@app.route("/api/upload_model", methods=["POST"])
def api_upload_model():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Missing file field 'file'"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"success": False, "message": "Empty upload"}), 400

    model_type = (request.form.get("model_type") or request.form.get("type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    if model_type not in ("nn", "transformer", ""):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'"}), 400
    if model_type == "":
        low = f.filename.lower()
        model_type = "transformer" if ("transformer" in low or low.startswith("tr_")) else "nn"

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify({"success": False, "message": f"Unsupported extension {ext}"}), 400

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_dir = _uploads_dir_for(model_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stamp}__{filename}"

    try:
        f.save(dest)
    except Exception as e:
        return jsonify({"success": False, "message": f"save failed: {e}"}), 500

    meta_obj: Optional[dict] = None
    meta_json = request.form.get("meta_json")
    if meta_json:
        try:
            meta_obj = json.loads(meta_json)
        except Exception:
            meta_obj = {"raw_meta": meta_json}
    else:
        meta_str = request.form.get("meta")
        if meta_str:
            try:
                meta_obj = json.loads(meta_str)
            except Exception:
                meta_obj = {"raw_meta": meta_str}

    meta_file = request.files.get("meta") or request.files.get("meta_file")
    if meta_file:
        try:
            meta_obj = json.loads(meta_file.read().decode("utf-8", errors="replace"))
        except Exception:
            pass

    if meta_obj is not None:
        try:
            dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
        except Exception:
            pass

    return jsonify({"success": True, "path": str(dest.resolve().relative_to(ROOT_DIR.resolve()).as_posix())}), 200


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.get_json(silent=True) or {}
    model_type = (data.get("model_type") or data.get("type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    filename = (data.get("filename") or "").strip()

    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'"}), 400
    if not filename:
        return jsonify({"success": False, "message": "filename is required"}), 400

    src = _resolve_model_file(model_type, filename)
    if src is None:
        return jsonify({"success": False, "message": "model file not found"}), 404

    dst = _active_weights_for(model_type)
    try:
        _atomic_copy(src, dst)
    except Exception as e:
        return jsonify({"success": False, "message": f"failed to activate model: {e}"}), 500

    if model_type == "nn":
        stop_service("nn")
        start_service("nn", NN_SCRIPT, NN_PORT)
    else:
        stop_service("transformer")
        start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)

    return jsonify({"success": True, "message": "Model reloaded", "active_path": str(dst)}), 200


@app.route("/api/delete_model", methods=["DELETE"])
def api_delete_model():
    data = request.get_json(silent=True) or {}
    model_type = (data.get("type") or data.get("model_type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    filename = (data.get("filename") or "").strip()

    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "type must be 'nn' or 'transformer'"}), 400
    if not filename:
        return jsonify({"success": False, "message": "filename is required"}), 400

    name = Path(filename).name
    base = _model_dir_for(model_type)
    matches = [p for p in base.rglob(name) if p.is_file() and p.suffix.lower() in ALLOWED_MODEL_EXTENSIONS]
    if not matches:
        return jsonify({"success": False, "message": "not found"}), 404

    active_path = _active_weights_for(model_type).resolve()
    target = matches[0].resolve()
    if target == active_path:
        return jsonify({"success": False, "message": "Refusing to delete active weights file"}), 400

    try:
        matches[0].unlink(missing_ok=True)
        sidecar = matches[0].with_suffix(matches[0].suffix + ".json")
        if sidecar.exists():
            sidecar.unlink(missing_ok=True)
    except Exception as e:
        return jsonify({"success": False, "message": f"delete failed: {e}"}), 500

    return jsonify({"success": True}), 200


# ---------------------------------------------------------------------
# Dojo (Training & Capture) APIs
# ---------------------------------------------------------------------
@app.route("/api/start_capture", methods=["POST"])
def api_start_capture():
    data = request.get_json(silent=True) or {}
    session_name = (data.get("session_name") or "session").strip()
    source = (data.get("source") or "screen").strip().lower()
    if source not in ("screen", "remote"):
        return jsonify({"success": False, "message": "source must be 'screen' or 'remote'"}), 400

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    sess_dir = RAW_DIR / session_id
    (sess_dir / "frames").mkdir(parents=True, exist_ok=True)
    (sess_dir / "inputs.jsonl").write_text("", encoding="utf-8")

    info = {
        "session_id": session_id,
        "session_name": session_name,
        "source": source,
        "created": int(time.time()),
        "frames": 0,
        "inputs": 0,
    }
    with _sessions_lock:
        _sessions[session_id] = info

    global _last_session_id
    _last_session_id = session_id

    return jsonify({"session_id": session_id, "status": "recording"}), 200


@app.route("/api/ingest_frame", methods=["POST"])
def api_ingest_frame():
    """
    Accepts:
      { "session_id": "sess_xxx", "image": "data:image/jpeg;base64,...", "timestamp": 12345 }
    session_id is optional; falls back to last session for UI convenience.
    """
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or _last_session_id or "").strip()
    image_str = data.get("image")
    ts = int(data.get("timestamp") or int(time.time() * 1000))

    if not session_id:
        return jsonify({"status": "error", "message": "missing session_id"}), 400

    with _sessions_lock:
        if session_id not in _sessions:
            return jsonify({"status": "error", "message": "invalid session_id"}), 400

    if not image_str:
        return jsonify({"status": "error", "message": "missing image"}), 400

    try:
        jpg = _decode_dataurl_to_jpg_bytes(image_str)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    sess_dir = RAW_DIR / session_id
    frames_dir = sess_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fp = frames_dir / f"frame_{ts}.jpg"
    fp.write_bytes(jpg)

    with _sessions_lock:
        _sessions[session_id]["frames"] = int(_sessions[session_id].get("frames", 0)) + 1

    return jsonify({"status": "queued"}), 200


@app.route("/api/ingest_input", methods=["POST"])
def api_ingest_input():
    """
    Accepts:
      { "session_id": "sess_xxx", "keys": ["w","space"], "timestamp": 12345 }
    session_id is optional; falls back to last session for UI convenience.
    """
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or _last_session_id or "").strip()
    ts = int(data.get("timestamp") or int(time.time() * 1000))
    keys = data.get("keys", [])

    if not session_id:
        return jsonify({"status": "error", "message": "missing session_id"}), 400

    with _sessions_lock:
        if session_id not in _sessions:
            return jsonify({"status": "error", "message": "invalid session_id"}), 400

    if not isinstance(keys, list):
        return jsonify({"status": "error", "message": "keys must be a list"}), 400

    sess_dir = RAW_DIR / session_id
    inputs_path = sess_dir / "inputs.jsonl"
    rec = {"timestamp": ts, "keys": keys}

    with inputs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with _sessions_lock:
        _sessions[session_id]["inputs"] = int(_sessions[session_id].get("inputs", 0)) + 1

    return jsonify({"status": "logged"}), 200


@app.route("/api/stop_capture", methods=["POST"])
def api_stop_capture():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or _last_session_id or "").strip()

    if not session_id:
        return jsonify({"success": False, "message": "missing session_id"}), 400

    with _sessions_lock:
        if session_id not in _sessions:
            return jsonify({"success": False, "message": "invalid session_id"}), 400

    sess_dir = RAW_DIR / session_id
    zip_path = RAW_DIR / f"{session_id}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sess_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(sess_dir)))

    return jsonify({"success": True, "dataset_path": str(sess_dir), "zip": str(zip_path)}), 200


@app.route("/api/train_offline", methods=["POST"])
def api_train_offline():
    """
    Accepts:
      { "dataset": "sess_xxx", "model_type": "nn"|"transformer", "epochs": 10 }
    """
    data = request.get_json(silent=True) or {}
    dataset = (data.get("dataset") or _last_session_id or "").strip()
    model_type = (data.get("model_type") or "nn").strip().lower()
    epochs = int(data.get("epochs") or 10)

    if model_type == "tf":
        model_type = "transformer"
    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'"}), 400

    if not dataset:
        return jsonify({"success": False, "message": "dataset must be provided"}), 400

    # dataset can be a known session OR an existing raw folder
    with _sessions_lock:
        known = dataset in _sessions
    if (not known) and not (RAW_DIR / dataset).exists():
        return jsonify({"success": False, "message": "dataset must be a valid session_id"}), 400

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "created": int(time.time())}

    t = Thread(target=_run_training_job, args=(job_id, dataset, model_type, epochs), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "started"}), 200


@app.route("/api/train_status/<job_id>", methods=["GET"])
def api_train_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "job not found"}), 404
    return jsonify(job), 200


# ---------------------------------------------------------------------
# Inference Lab APIs
# ---------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "payload must be JSON object"}), 400

    if active_model == "nn":
        if not is_service_running(NN_PORT):
            ok, _, msg = start_service("nn", NN_SCRIPT, NN_PORT)
            if not ok:
                return jsonify({"success": False, "error": "nn not available", "details": msg}), 503
        port = NN_PORT
    else:
        if not is_service_running(TRANSFORMER_PORT):
            ok, _, msg = start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
            if not ok:
                return jsonify({"success": False, "error": "transformer not available", "details": msg}), 503
        port = TRANSFORMER_PORT

    t0 = time.time()
    try:
        resp = requests.post(_predict_url(port), json=data, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "model service error",
                        "status_code": resp.status_code,
                        "details": resp.text,
                        "router_latency_ms": latency_ms,
                        "active_model": active_model,
                    }
                ),
                502,
            )
        payload = resp.json()
        payload["active_model"] = active_model
        payload["router_latency_ms"] = latency_ms
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"success": False, "error": "failed to reach model service", "details": str(e)}), 502


@app.route("/api/test_predict", methods=["POST"])
def api_test_predict():
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "nn").strip().lower()
    if model not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model must be 'nn' or 'transformer'"}), 400

    if model == "nn":
        port, script, svc = NN_PORT, NN_SCRIPT, "nn"
    else:
        port, script, svc = TRANSFORMER_PORT, TRANSFORMER_SCRIPT, "transformer"

    if not is_service_running(port):
        ok, _, msg = start_service(svc, script, port)
        if not ok:
            return jsonify({"success": False, "message": "failed to start service", "details": msg}), 503

    dummy = {"image": _png_dataurl_dummy()}

    t0 = time.time()
    try:
        resp = requests.post(_predict_url(port), json=dummy, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return jsonify({"success": False, "message": f"service returned {resp.status_code}", "latency_ms": latency_ms, "details": resp.text}), 502
        out = resp.json()
        return jsonify({"action": out.get("action", "UNKNOWN"), "latency_ms": latency_ms}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 502


# ---------------------------------------------------------------------
# Analytics & Logs APIs
# ---------------------------------------------------------------------
@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent

    gpu = None
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            gpu = float(out.strip().splitlines()[0])
    except Exception:
        gpu = None

    rps = _compute_rps(window_sec=5)
    return jsonify({"cpu": cpu, "ram": ram, "gpu": gpu, "rps": rps}), 200


@app.route("/api/service_log/<service_name>", methods=["GET"])
def api_service_log(service_name: str):
    if service_name not in ("nn", "transformer"):
        return jsonify({"error": "invalid service"}), 400
    p = _service_log_path(service_name)
    if not p.exists():
        return jsonify({"log": "", "note": "no log yet"}), 200
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    return jsonify({"log": "\n".join(lines)}), 200


# ---------------------------------------------------------------------
# Legacy health
# ---------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200


def cleanup():
    stop_service("nn")
    stop_service("transformer")
    for k, f in list(service_logs.items()):
        try:
            if f and not f.closed:
                f.flush()
                f.close()
        except Exception:
            pass
        service_logs[k] = None


atexit.register(cleanup)


if __name__ == "__main__":
    print(f"Starting Control Backend on {BACKEND_HOST}:{BACKEND_PORT}")
    print(f"UI: http://localhost:{BACKEND_PORT}/")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)
