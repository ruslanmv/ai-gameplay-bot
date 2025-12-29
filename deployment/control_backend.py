"""
deployment/control_backend.py

Enterprise Gameplay Bot - Control Backend (Flask)
- Serves frontend UI at "/"
- Orchestrates microservices: NN (:5000) and Transformer (:5001)
- Model repository APIs (.pth + optional .json metadata)
- Dojo APIs: capture frames/inputs, stop sessions, run offline training jobs
- Inference Lab: /api/predict routes to active model
- Analytics: /api/metrics and /api/service_log/<svc>

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
import subprocess
import sys
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

import psutil
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Paths / Project root
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------
service_processes: Dict[str, Optional[subprocess.Popen]] = {"nn": None, "transformer": None}
active_model = "nn"

# RPS tracking (simple rolling window)
_req_times = deque(maxlen=5000)
_req_lock = Lock()

# Dojo session tracking
_sessions_lock = Lock()
_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> info

# Training jobs
_jobs_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> status

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _service_log_path(service_name: str) -> Path:
    return LOG_DIR / f"{service_name}.log"

def _is_windows() -> bool:
    return os.name == "nt"

def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"

def is_service_running(port: int) -> bool:
    try:
        resp = requests.get(_health_url(port), timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False

def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_service(service_name: str, script_path: Path, port: int) -> bool:
    global service_processes

    proc = service_processes.get(service_name)
    if proc is not None and proc.poll() is None:
        return True
    if is_service_running(port):
        return True
    if _port_in_use(port):
        return False
    if not script_path.exists():
        return False

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
                return False
            if is_service_running(port):
                return True
            time.sleep(0.4)
        return False
    except Exception:
        return False

def stop_service(service_name: str) -> bool:
    global service_processes

    proc = service_processes.get(service_name)
    if proc is None or proc.poll() is not None:
        service_processes[service_name] = None
        return True

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
        return True
    except Exception:
        return False

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

def _safe_resolve_within_root(rel_path: str) -> Path:
    candidate = (ROOT_DIR / rel_path).resolve()
    root = ROOT_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid path (outside project root).")
    return candidate

def _file_meta(p: Path, model_type: str) -> Dict[str, Any]:
    st = p.stat()
    meta_path = p.with_suffix(p.suffix + ".json")
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None

    return {
        "filename": p.name,
        "type": "nn" if model_type == "nn" else "transformer",
        "size": st.st_size,
        "date": int(st.st_mtime),
        "path": str(p.resolve().relative_to(ROOT_DIR.resolve()).as_posix()),
        "meta": meta,
    }

def _list_models_flat() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in ("nn", "transformer"):
        base = _model_dir_for(t)
        if not base.exists():
            continue
        for p in base.rglob("*.pth"):
            if p.is_file():
                out.append(_file_meta(p, t))
    out.sort(key=lambda d: (d.get("type",""), d.get("date",0)), reverse=True)
    return out

def _record_request_for_rps() -> None:
    now = time.time()
    with _req_lock:
        _req_times.append(now)

def _compute_rps(window_sec: int = 5) -> float:
    now = time.time()
    cutoff = now - float(window_sec)
    with _req_lock:
        n = 0
        # iterate from right (newest) for speed
        for t in reversed(_req_times):
            if t < cutoff:
                break
            n += 1
    return n / float(window_sec)

def _decode_dataurl_to_jpg_bytes(data: str) -> bytes:
    """
    Accepts data URL or base64. Re-encodes to JPEG bytes for consistent storage.
    """
    if not isinstance(data, str) or not data.strip():
        raise ValueError("image must be a non-empty string")

    s = data.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    raw = base64.b64decode(s, validate=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _keys_to_action(keys: List[str]) -> str:
    """
    Minimal heuristic mapping: choose primary key to action label.
    """
    s = set((k or "").lower() for k in keys if isinstance(k, str))
    if "w" in s: return "move_forward"
    if "s" in s: return "move_backward"
    if "a" in s: return "turn_left"
    if "d" in s: return "turn_right"
    if "space" in s: return "jump"
    if "e" in s: return "interact"
    if "f" in s: return "use_item"
    if "tab" in s: return "open_inventory"
    if "q" in s: return "cast_spell"
    # default fallback
    return "move_forward"

def _build_dataset_csv_from_session(session_id: str) -> Path:
    """
    Builds a CSV dataset with 128 features + action_index.
    Uses deployment.feature_extractor (16x8 grayscale flatten).
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

    # Load inputs keyed by timestamp
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

    # Pair frames with nearest input timestamp (exact match first, else nearest within 250ms)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    rows: List[Dict[str, Any]] = []

    input_ts_sorted = sorted(inputs.keys())

    def nearest_ts(target: int) -> Optional[int]:
        if not input_ts_sorted:
            return None
        # binary search
        import bisect
        i = bisect.bisect_left(input_ts_sorted, target)
        candidates = []
        if 0 <= i < len(input_ts_sorted): candidates.append(input_ts_sorted[i])
        if 0 <= i-1 < len(input_ts_sorted): candidates.append(input_ts_sorted[i-1])
        if not candidates: return None
        best = min(candidates, key=lambda t: abs(t - target))
        if abs(best - target) <= 250:
            return best
        return None

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
        # frame_<timestamp>.jpg
        try:
            ts = int(fp.stem.split("_", 1)[1])
        except Exception:
            continue

        ts2 = ts if ts in inputs else nearest_ts(ts)
        keys = inputs.get(ts2, []) if ts2 is not None else []
        action_str = _keys_to_action(keys)
        action_idx = action_map.get(action_str, 0)

        # Read file and convert to features using the same method as inference
        b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
        feats = image_to_features_128(b64)

        row: Dict[str, Any] = {f"f{i}": feats[i] for i in range(128)}
        row["action"] = action_idx
        row["timestamp"] = ts
        rows.append(row)

    # Write CSV
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
    """
    Background training:
    - build processed CSV from session capture
    - run the repo's training script (best-effort, may take time)
    - copy resulting .pth into uploads with timestamp
    """
    def set_job(**kw):
        with _jobs_lock:
            _jobs[job_id].update(**kw)

    try:
        set_job(status="building_dataset")
        csv_path = _build_dataset_csv_from_session(dataset_session)
        set_job(status="training", dataset_csv=str(csv_path))

        # We trigger existing training scripts, but they expect fixed dataset paths.
        # So we copy into the expected location (nn uses data/processed/nn_dataset.csv).
        if model_type == "nn":
            expected_csv = PROCESSED_DIR / "nn_dataset.csv"
            shutil.copy2(csv_path, expected_csv)

            cmd = [sys.executable, str(ROOT_DIR / "models" / "neural_network" / "nn_training.py")]
            env = os.environ.copy()
        else:
            expected_csv = PROCESSED_DIR / "transformer_dataset.csv"
            shutil.copy2(csv_path, expected_csv)

            cmd = [sys.executable, str(ROOT_DIR / "models" / "transformer" / "transformer_training.py")]
            env = os.environ.copy()

        # Best-effort: if script writes finetuned weights to default locations, we then archive them.
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream logs into job
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

        # Archive produced weights
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if model_type == "nn":
            produced = NN_ACTIVE_WEIGHTS
            updir = NN_UPLOADS_DIR
        else:
            produced = TR_ACTIVE_WEIGHTS
            updir = TR_UPLOADS_DIR

        if not produced.exists():
            set_job(status="failed", error=f"Expected weights not found at {produced}")
            return

        out_name = f"{stamp}__{model_type}__trained_from_{dataset_session}.pth"
        out_path = updir / out_name
        shutil.copy2(produced, out_path)

        # Save metadata
        meta = {
            "trained_from_session": dataset_session,
            "epochs": epochs,
            "created": int(time.time()),
            "source_csv": str(csv_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()),
        }
        (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        set_job(status="completed", output_model=str(out_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()))

    except Exception as e:
        set_job(status="failed", error=str(e))

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
# System & Orchestration APIs
# -----------------------------------------------------------------------------
@app.before_request
def _track_rps():
    # track API request rate (lightweight)
    if request.path.startswith("/api/"):
        _record_request_for_rps()

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "healthy",
        "uptime": int(time.time() - START_TIME),
        "timestamp": time.time(),
    }), 200

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "nn_running": is_service_running(NN_PORT),
        "transformer_running": is_service_running(TRANSFORMER_PORT),
        "active_model": active_model,
        "timestamp": time.time(),
    })

@app.route("/api/start_nn", methods=["POST"])
def start_nn():
    ok = start_service("nn", NN_SCRIPT, NN_PORT)
    return jsonify({"success": ok, "pid": service_processes["nn"].pid if (ok and service_processes["nn"]) else None}), (200 if ok else 500)

@app.route("/api/stop_nn", methods=["POST"])
def stop_nn():
    ok = stop_service("nn")
    return jsonify({"success": ok}), (200 if ok else 500)

@app.route("/api/start_transformer", methods=["POST"])
def start_transformer():
    ok = start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
    return jsonify({"success": ok, "pid": service_processes["transformer"].pid if (ok and service_processes["transformer"]) else None}), (200 if ok else 500)

@app.route("/api/stop_transformer", methods=["POST"])
def stop_transformer():
    ok = stop_service("transformer")
    return jsonify({"success": ok}), (200 if ok else 500)

@app.route("/api/set_active_model", methods=["POST"])
def set_active_model():
    global active_model
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "").strip().lower()
    if model not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model must be 'nn' or 'transformer'"}), 400
    active_model = model
    return jsonify({"active_model": active_model}), 200

# -----------------------------------------------------------------------------
# Model Repository APIs
# -----------------------------------------------------------------------------
@app.route("/api/models", methods=["GET"])
def api_models():
    """
    Spec-friendly response:
      { "models": [ {filename,type,size,date,...}, ... ] }
    """
    models = _list_models_flat()
    return jsonify({"models": models}), 200

@app.route("/api/upload_model", methods=["POST"])
def api_upload_model():
    """
    Upload .pth plus optional meta json.
    FormData:
      - file (.pth) REQUIRED
      - type: nn|tf|transformer (optional)
      - meta: json string (optional) OR meta_file (optional file)
    """
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Missing file field 'file'"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"success": False, "message": "Empty upload"}), 400

    model_type = (request.form.get("type") or request.form.get("model_type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    if model_type not in ("nn", "transformer", ""):
        return jsonify({"success": False, "message": "type must be 'nn' or 'transformer'"}), 400
    if model_type == "":
        # infer
        low = f.filename.lower()
        model_type = "transformer" if "transformer" in low or low.startswith("tr_") else "nn"

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify({"success": False, "message": f"Unsupported extension {ext}"}), 400

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_dir = _uploads_dir_for(model_type)
    dest = dest_dir / f"{stamp}__{filename}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    f.save(dest)

    # meta handling
    meta_obj = None
    meta_str = request.form.get("meta")
    if meta_str:
        try:
            meta_obj = json.loads(meta_str)
        except Exception:
            meta_obj = {"raw_meta": meta_str}

    if "meta_file" in request.files and request.files["meta_file"]:
        try:
            meta_file = request.files["meta_file"]
            meta_obj = json.loads(meta_file.read().decode("utf-8", errors="replace"))
        except Exception:
            pass

    if meta_obj is not None:
        dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

    return jsonify({"success": True, "path": str(dest.resolve().relative_to(ROOT_DIR.resolve()).as_posix())}), 200

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """
    Spec:
      { "model_type": "nn"|"transformer", "filename": "v1.pth" }
    Behavior:
      - searches in model dir + uploads
      - copies to active weights location
      - restarts corresponding service
    """
    data = request.get_json(silent=True) or {}
    model_type = (data.get("model_type") or data.get("type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    filename = (data.get("filename") or "").strip()

    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'"}), 400
    if not filename:
        return jsonify({"success": False, "message": "filename is required"}), 400

    base = _model_dir_for(model_type)
    candidates = list(base.rglob(filename))
    candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED_MODEL_EXTENSIONS]
    if not candidates:
        return jsonify({"success": False, "message": "model file not found"}), 404

    src = candidates[0]
    dst = _active_weights_for(model_type)
    _atomic_copy(src, dst)

    # restart service so it reloads weights
    if model_type == "nn":
        stop_service("nn")
        start_service("nn", NN_SCRIPT, NN_PORT)
    else:
        stop_service("transformer")
        start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)

    return jsonify({"success": True, "message": "Model reloaded", "active_path": str(dst.resolve().relative_to(ROOT_DIR.resolve()).as_posix())}), 200

@app.route("/api/delete_model", methods=["DELETE"])
def api_delete_model():
    """
    Spec:
      { "filename": "v1.pth", "type": "nn"|"transformer" }
    """
    data = request.get_json(silent=True) or {}
    model_type = (data.get("type") or data.get("model_type") or "").strip().lower()
    if model_type == "tf":
        model_type = "transformer"
    filename = (data.get("filename") or "").strip()

    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "type must be 'nn' or 'transformer'"}), 400
    if not filename:
        return jsonify({"success": False, "message": "filename is required"}), 400

    base = _model_dir_for(model_type)
    matches = [p for p in base.rglob(filename) if p.is_file()]
    if not matches:
        return jsonify({"success": False, "message": "not found"}), 404

    # refuse deleting active weights
    active_path = _active_weights_for(model_type).resolve()
    target = matches[0].resolve()
    if target == active_path:
        return jsonify({"success": False, "message": "Refusing to delete active weights file"}), 400

    # delete .pth and possible .json sidecar
    matches[0].unlink(missing_ok=True)
    sidecar = matches[0].with_suffix(matches[0].suffix + ".json")
    if sidecar.exists():
        sidecar.unlink(missing_ok=True)

    return jsonify({"success": True}), 200

# -----------------------------------------------------------------------------
# Dojo (Training & Capture) APIs
# -----------------------------------------------------------------------------
@app.route("/api/start_capture", methods=["POST"])
def start_capture_api():
    data = request.get_json(silent=True) or {}
    session_name = (data.get("session_name") or "session").strip()
    source = (data.get("source") or "screen").strip().lower()
    if source not in ("screen", "remote"):
        return jsonify({"success": False, "message": "source must be 'screen' or 'remote'"}), 400

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    sess_dir = RAW_DIR / session_id
    frames_dir = sess_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
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

    return jsonify({"session_id": session_id, "status": "recording"}), 200

@app.route("/api/ingest_frame", methods=["POST"])
def ingest_frame_api():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    image_str = data.get("image")
    ts = int(data.get("timestamp") or int(time.time() * 1000))

    if not session_id or session_id not in _sessions:
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
        _sessions[session_id]["frames"] += 1

    return jsonify({"status": "queued"}), 200

@app.route("/api/ingest_input", methods=["POST"])
def ingest_input_api():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    ts = int(data.get("timestamp") or int(time.time() * 1000))
    keys = data.get("keys", [])

    if not session_id or session_id not in _sessions:
        return jsonify({"status": "error", "message": "invalid session_id"}), 400
    if not isinstance(keys, list):
        return jsonify({"status": "error", "message": "keys must be a list"}), 400

    sess_dir = RAW_DIR / session_id
    inputs_path = sess_dir / "inputs.jsonl"
    rec = {"timestamp": ts, "keys": keys}

    with inputs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with _sessions_lock:
        _sessions[session_id]["inputs"] += 1

    return jsonify({"status": "logged"}), 200

@app.route("/api/stop_capture", methods=["POST"])
def stop_capture_api():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id or session_id not in _sessions:
        return jsonify({"success": False, "message": "invalid session_id"}), 400

    sess_dir = RAW_DIR / session_id
    zip_path = RAW_DIR / f"{session_id}.zip"

    # zip session directory
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sess_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(sess_dir)))

    return jsonify({"success": True, "dataset_path": str(sess_dir.resolve().relative_to(ROOT_DIR.resolve()).as_posix()), "zip": str(zip_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix())}), 200

@app.route("/api/train_offline", methods=["POST"])
def train_offline_api():
    data = request.get_json(silent=True) or {}
    dataset = (data.get("dataset") or "").strip()       # session_id
    model_type = (data.get("model_type") or "nn").strip().lower()
    epochs = int(data.get("epochs") or 10)

    if model_type == "tf":
        model_type = "transformer"
    if model_type not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model_type must be 'nn' or 'transformer'"}), 400

    if not dataset or dataset not in _sessions:
        # allow training from existing raw folder too
        if not (RAW_DIR / dataset).exists():
            return jsonify({"success": False, "message": "dataset must be a valid session_id"}), 400

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "created": int(time.time())}

    t = Thread(target=_run_training_job, args=(job_id, dataset, model_type, epochs), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "started"}), 200

@app.route("/api/train_status/<job_id>", methods=["GET"])
def train_status_api(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "job not found"}), 404
    return jsonify(job), 200

# -----------------------------------------------------------------------------
# Inference Lab APIs
# -----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Main inference endpoint.
    Forwards request JSON to the active microservice /predict.
    Accepts:
      { "image": "data:image/jpeg;base64,...", "features": [...] }
    Returns:
      { "action": "...", "confidence": 0.95, "tensor_viz": [...], ... }
    """
    data = request.get_json(silent=True) or {}

    # ensure service is running
    if active_model == "nn":
        if not is_service_running(NN_PORT):
            start_service("nn", NN_SCRIPT, NN_PORT)
        port = NN_PORT
    else:
        if not is_service_running(TRANSFORMER_PORT):
            start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)
        port = TRANSFORMER_PORT

    t0 = time.time()
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/predict", json=data, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return jsonify({"success": False, "error": "model service error", "status_code": resp.status_code, "details": resp.text}), 502

        payload = resp.json()
        payload["active_model"] = active_model
        payload["router_latency_ms"] = latency_ms
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"success": False, "error": "failed to reach model service", "details": str(e)}), 502

@app.route("/api/test_predict", methods=["POST"])
def test_predict():
    """
    Mock prediction for dashboard health check.
    { "model": "nn" } -> calls that service
    """
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "nn").strip().lower()
    if model not in ("nn", "transformer"):
        return jsonify({"success": False, "message": "model must be 'nn' or 'transformer'"}), 400

    port = NN_PORT if model == "nn" else TRANSFORMER_PORT
    if model == "nn" and not is_service_running(NN_PORT):
        start_service("nn", NN_SCRIPT, NN_PORT)
    if model == "transformer" and not is_service_running(TRANSFORMER_PORT):
        start_service("transformer", TRANSFORMER_SCRIPT, TRANSFORMER_PORT)

    # send dummy features
    dummy = {"features": [0.0] * 128}
    t0 = time.time()
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/predict", json=dummy, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return jsonify({"success": False, "message": f"service returned {resp.status_code}", "latency_ms": latency_ms}), 502
        out = resp.json()
        return jsonify({"action": out.get("action", "UNKNOWN"), "latency_ms": latency_ms}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 502

# -----------------------------------------------------------------------------
# Analytics & Logs APIs
# -----------------------------------------------------------------------------
@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent

    # GPU (best-effort): if nvidia-smi exists, parse utilization
    gpu = None
    try:
        import shutil as _sh
        if _sh.which("nvidia-smi"):
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            gpu = float(out.strip().splitlines()[0])
    except Exception:
        gpu = None

    rps = _compute_rps(window_sec=5)
    return jsonify({"cpu": cpu, "ram": ram, "gpu": gpu, "rps": rps}), 200

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
# Health (legacy)
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

def cleanup():
    stop_service("nn")
    stop_service("transformer")

atexit.register(cleanup)

if __name__ == "__main__":
    print(f"Starting Control Backend on {BACKEND_HOST}:{BACKEND_PORT}")
    print(f"UI: http://localhost:{BACKEND_PORT}/")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)
