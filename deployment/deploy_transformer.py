from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Imports / paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.transformer.transformer_model import GameplayTransformer
from deployment.feature_extractor import safe_features_from_payload

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("transformer_service")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = GameplayTransformer(INPUT_SIZE, NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
model_loaded: bool = False
model_error: Optional[str] = None


def load_weights() -> None:
    global model_loaded, model_error
    if not MODEL_PATH.exists():
        model_loaded = False
        model_error = f"Model weights not found at: {MODEL_PATH}"
        logger.error(model_error)
        return

    try:
        state = torch.load(str(MODEL_PATH), map_location=DEVICE)
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
        logger.info("Transformer weights loaded: %s (device=%s)", MODEL_PATH, DEVICE)
    except Exception as e:
        model_loaded = False
        model_error = str(e)
        logger.exception("Failed to load Transformer weights: %s", e)


load_weights()

# -----------------------------------------------------------------------------
# Payload -> features (production-safe)
# -----------------------------------------------------------------------------
def _extract_and_validate_features(payload: dict) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Uses shared safe_features_from_payload(payload) (currently hardcoded to 128 for image-derived features).
    Production-safe behavior:
      - Always validate output length against INPUT_SIZE.
      - If INPUT_SIZE != 128 and payload uses "image", fail with a clear message.
    """
    if not isinstance(payload, dict):
        return None, "payload must be a JSON object"

    # If user uses image but expects non-128 input, fail early with a clear explanation.
    if payload.get("image") is not None and INPUT_SIZE != 128:
        return None, (
            f"image-based feature extraction outputs 128 features, but TRANSFORMER_INPUT_SIZE={INPUT_SIZE}. "
            f"Send 'features' with length {INPUT_SIZE}, or upgrade feature_extractor to support {INPUT_SIZE}."
        )

    feats, err = safe_features_from_payload(payload)  # <-- NO expected_dim kwarg (fixes your crash)
    if err:
        return None, err
    if feats is None:
        return None, "failed to extract features"

    if len(feats) != INPUT_SIZE:
        return None, f"features length mismatch: got {len(feats)}, expected {INPUT_SIZE}"

    return feats, None


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def _to_model_input(features: List[float]) -> torch.Tensor:
    """
    Safer default for transformer-like models: (B, S, F) with S=1.
    If your GameplayTransformer expects (B, F), it can still handle reshape internally,
    but (1,1,F) avoids the common rank mismatch.
    """
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    return x.view(1, 1, -1)  # (F,) -> (1, 1, F)


def infer(features: List[float]) -> Dict[str, Any]:
    x = _to_model_input(features)
    with torch.no_grad():
        out = model(x)

        # Normalize output into a 1D vector of class scores
        if isinstance(out, (list, tuple)):
            out = out[0]

        if out.dim() == 3:
            # (B, S, C) -> last token
            vec = out[0, -1, :]
        elif out.dim() == 2:
            # (B, C)
            vec = out[0, :]
        else:
            vec = out.reshape(-1)

        probs = torch.softmax(vec, dim=0)
        probs_list = probs.detach().cpu().tolist()
        action_idx = int(torch.argmax(probs).item())
        conf = float(max(probs_list)) if probs_list else 0.0

        return {
            "action": ACTION_MAPPING.get(action_idx, "UNKNOWN_ACTION"),
            "confidence": conf,
            "tensor_viz": probs_list,
            "action_index": action_idx,
            "device": DEVICE,
        }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    payload = request.get_json(silent=True) or {}

    if not model_loaded:
        return jsonify({"error": "Model not loaded", "details": model_error}), 503

    features, err = _extract_and_validate_features(payload)
    if err:
        return jsonify({"error": err}), 400

    try:
        out = infer(features)
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["model_path"] = str(MODEL_PATH)
        out["model_loaded"] = model_loaded
        out["input_size"] = INPUT_SIZE
        return jsonify(out), 200
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
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
                "service": "transformer",
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
    logger.info("Starting Transformer service on %s:%s (device=%s)", HOST, PORT, DEVICE)
    app.run(host=HOST, port=PORT, debug=False)
