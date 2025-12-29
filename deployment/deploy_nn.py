from __future__ import annotations

import logging
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nn_service")

# -----------------------------------------------------------------------------
# Fix imports (ensure repo root is importable)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.neural_network.nn_model import GameplayNN
from deployment.feature_extractor import safe_features_from_payload

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

HOST = os.environ.get("NN_HOST", "0.0.0.0")
PORT = int(os.environ.get("NN_PORT", "5000"))

MODEL_PATH = ROOT_DIR / "models" / "neural_network" / "nn_model_finetuned.pth"
INPUT_SIZE = 128
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10

ACTION_MAPPING: Dict[int, str] = {
    0: "MOVE_FORWARD",
    1: "MOVE_BACKWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "ATTACK",
    5: "JUMP",
    6: "INTERACT",
    7: "USE_ITEM",
    8: "OPEN_INVENTORY",
    9: "CAST_SPELL",
}

model = GameplayNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model_loaded = False
model_error: str | None = None

def load_weights() -> None:
    global model_loaded, model_error
    if MODEL_PATH.exists():
        try:
            logger.info("Loading NN weights from %s", MODEL_PATH)
            state = torch.load(str(MODEL_PATH), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            model_loaded = True
            model_error = None
            logger.info("NN weights loaded.")
        except Exception as e:
            model_loaded = False
            model_error = str(e)
            logger.exception("Failed to load NN weights: %s", e)
    else:
        model_loaded = False
        model_error = f"Model weights not found at {MODEL_PATH}"
        logger.warning("%s. Running with RANDOM weights.", model_error)
        model.eval()

load_weights()

def infer(features: List[float]) -> Dict[str, Any]:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = model(x)  # already softmax in this model
        probs_list = probs.squeeze(0).tolist()
        action_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(max(probs_list)) if probs_list else 0.0
        return {
            "action": ACTION_MAPPING.get(action_idx, "UNKNOWN_ACTION"),
            "confidence": conf,
            "tensor_viz": probs_list,
            "action_index": action_idx,
        }

@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    payload = request.get_json(silent=True) or {}

    features, err = safe_features_from_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    try:
        out = infer(features)
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["model_loaded"] = model_loaded
        out["model_path"] = str(MODEL_PATH)
        if not model_loaded:
            out["warning"] = model_error
        return jsonify(out), 200
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "service": "neural_network",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "error": model_error,
    }), (200 if model_loaded else 503)

if __name__ == "__main__":
    logger.info("Starting NN service on %s:%s ...", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False)
