from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nn_service")

# -----------------------------------------------------------------------------
# Fix imports (ensure repo root is importable)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.neural_network.nn_model import GameplayNN  # noqa: E402
from deployment.feature_extractor import safe_features_from_payload  # noqa: E402

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

HOST = os.environ.get("NN_HOST", "0.0.0.0")
PORT = int(os.environ.get("NN_PORT", "5000"))

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "neural_network" / "nn_model_finetuned.pth"
MODEL_PATH = Path(os.environ.get("NN_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()

INPUT_SIZE = int(os.environ.get("NN_INPUT_SIZE", "128"))
HIDDEN_SIZE = int(os.environ.get("NN_HIDDEN_SIZE", "64"))
OUTPUT_SIZE = int(os.environ.get("NN_OUTPUT_SIZE", "10"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

model = GameplayNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
model_loaded: bool = False
model_error: Optional[str] = None


def load_weights() -> None:
    global model_loaded, model_error

    if not MODEL_PATH.exists():
        model_loaded = False
        model_error = f"Model weights not found at {MODEL_PATH}"
        logger.error(model_error)
        return

    try:
        logger.info("Loading NN weights from %s (device=%s)", MODEL_PATH, DEVICE)
        state = torch.load(str(MODEL_PATH), map_location=DEVICE)

        # Support common checkpoint formats
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys (strict=False): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys (strict=False): %s", unexpected)

        model.eval()
        model_loaded = True
        model_error = None
        logger.info("NN weights loaded successfully.")
    except Exception as e:
        model_loaded = False
        model_error = str(e)
        logger.exception("Failed to load NN weights: %s", e)


load_weights()


def infer(features: List[float]) -> Dict[str, Any]:
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, F)
    with torch.no_grad():
        probs = model(x)  # assumes model outputs probabilities
        probs_list = probs.squeeze(0).detach().cpu().tolist()
        action_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(max(probs_list)) if probs_list else 0.0
        return {
            "action": ACTION_MAPPING.get(action_idx, "UNKNOWN_ACTION"),
            "confidence": conf,
            "tensor_viz": probs_list,
            "action_index": action_idx,
            "device": DEVICE,
        }


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "payload must be a JSON object"}), 400

    if not model_loaded:
        return jsonify({"error": "Model not loaded", "details": model_error}), 503

    # IMPORTANT: feature_extractor currently supports ONLY 128-dim features.
    features, err = safe_features_from_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    # Enforce expected input size here (production-safe validation)
    if features is None or len(features) != INPUT_SIZE:
        return jsonify({"error": f"features must be length {INPUT_SIZE}"}), 400

    try:
        out = infer(features)
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["model_loaded"] = model_loaded
        out["model_path"] = str(MODEL_PATH)
        out["input_size"] = INPUT_SIZE
        return jsonify(out), 200
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    load_weights()
    if not model_loaded:
        return jsonify({"success": False, "error": model_error}), 503
    return jsonify({"success": True, "model_path": str(MODEL_PATH)}), 200


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify(
            {
                "status": "healthy" if model_loaded else "degraded",
                "service": "neural_network",
                "model_loaded": model_loaded,
                "model_path": str(MODEL_PATH),
                "device": DEVICE,
                "input_size": INPUT_SIZE,
                "error": model_error,
            }
        ),
        (200 if model_loaded else 503),
    )


if __name__ == "__main__":
    logger.info("Starting NN service on %s:%s (device=%s)", HOST, PORT, DEVICE)
    app.run(host=HOST, port=PORT, debug=False)
