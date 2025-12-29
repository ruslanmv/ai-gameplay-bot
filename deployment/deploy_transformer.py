# deployment/deploy_transformer.py
"""
Transformer Inference Service (Production-ready)
- Robust model loading (works from any CWD)
- Clear errors if weights are missing/incompatible
- /predict and /health endpoints for Control Backend
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# Local import (assumes transformer_model.py is importable from project root)
from models.transformer.transformer_model import GameplayTransformer


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("transformer_service")

# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # project root
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "transformer" / "transformer_model_finetuned.pth"

MODEL_PATH = Path(os.environ.get("TRANSFORMER_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()

INPUT_SIZE = int(os.environ.get("TRANSFORMER_INPUT_SIZE", "128"))
NUM_HEADS = int(os.environ.get("TRANSFORMER_NUM_HEADS", "4"))
HIDDEN_SIZE = int(os.environ.get("TRANSFORMER_HIDDEN_SIZE", "64"))
NUM_LAYERS = int(os.environ.get("TRANSFORMER_NUM_LAYERS", "2"))
OUTPUT_SIZE = int(os.environ.get("TRANSFORMER_OUTPUT_SIZE", "10"))

HOST = os.environ.get("TRANSFORMER_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRANSFORMER_PORT", "5001"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_MAPPING: Dict[int, str] = {
    0: "move_forward",
    1: "move_backward",
    2: "turn_left",
    3: "turn_right",
    4: "attack",
    5: "jump",
    6: "interact",
    7: "use_item",
    8: "open_inventory",
    9: "cast_spell",
}

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Model init + load (robust)
# -----------------------------------------------------------------------------
model = GameplayTransformer(INPUT_SIZE, NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
model_loaded = False
model_load_error: str | None = None


def _load_weights() -> None:
    """Load model weights safely and record status for /health."""
    global model_loaded, model_load_error

    if not MODEL_PATH.exists():
        model_loaded = False
        model_load_error = (
            f"Model weights not found at: {MODEL_PATH}. "
            f"Train the transformer or set TRANSFORMER_MODEL_PATH."
        )
        logger.error(model_load_error)
        return

    try:
        state = torch.load(str(MODEL_PATH), map_location=DEVICE)

        # Common cases:
        # - state is a dict of tensors (state_dict)
        # - state may be a checkpoint dict with a state_dict key
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise ValueError("Loaded object is not a state_dict (dict).")

        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()

        model_loaded = True
        model_load_error = None

        if missing:
            logger.warning("Missing keys when loading weights (strict=False): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading weights (strict=False): %s", unexpected)

        logger.info("Transformer weights loaded: %s (device=%s)", MODEL_PATH, DEVICE)

    except Exception as e:
        model_loaded = False
        model_load_error = f"Failed to load weights from {MODEL_PATH}: {e}"
        logger.exception(model_load_error)


_load_weights()


def _validate_state(state: Any) -> tuple[bool, str]:
    if not isinstance(state, list):
        return False, "'state' must be a list"
    if len(state) != INPUT_SIZE:
        return False, f"'state' must be a list of length {INPUT_SIZE}"
    # Ensure elements are numeric
    for i, v in enumerate(state):
        if not isinstance(v, (int, float)):
            return False, f"'state[{i}]' must be a number"
    return True, ""


def predict_action(state: List[float]) -> str:
    """Predict an action label from a 128-dim state."""
    # Shape: (1, INPUT_SIZE)
    x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

        # Support either logits or probabilities
        if out.dim() == 2 and out.size(0) == 1:
            action_index = int(torch.argmax(out, dim=1).item())
        else:
            action_index = int(torch.argmax(out).item())

    return ACTION_MAPPING.get(action_index, "unknown_action")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if "state" not in data:
        return jsonify({"error": "Missing 'state' in request"}), 400

    ok, msg = _validate_state(data["state"])
    if not ok:
        return jsonify({"error": msg}), 400

    if not model_loaded:
        return jsonify({"error": "Model not loaded", "details": model_load_error}), 503

    try:
        action = predict_action(data["state"])
        return jsonify({"action": action}), 200
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify(
            {
                "status": "healthy" if model_loaded else "degraded",
                "service": "transformer",
                "model_loaded": model_loaded,
                "model_path": str(MODEL_PATH),
                "device": DEVICE,
                "error": model_load_error,
            }
        ),
        200 if model_loaded else 503,
    )


if __name__ == "__main__":
    logger.info("Starting Transformer service on %s:%s (device=%s)", HOST, PORT, DEVICE)
    app.run(host=HOST, port=PORT, debug=False)
