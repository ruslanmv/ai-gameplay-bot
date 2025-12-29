import logging
from pathlib import Path
import sys

import torch
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deploy-nn")

# -----------------------------------------------------------------------------
# Fix imports (ensure repo root and models dir are importable)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
NN_DIR = ROOT_DIR / "models" / "neural_network"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(NN_DIR))

from models.neural_network.nn_model import GameplayNN

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = ROOT_DIR / "models" / "neural_network" / "nn_model_finetuned.pth"
INPUT_SIZE = 128
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10

model = GameplayNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

if MODEL_PATH.exists():
    logger.info("Loading NN weights from %s", MODEL_PATH)
    state = torch.load(str(MODEL_PATH), map_location="cpu")
    model.load_state_dict(state)
else:
    logger.warning("Model weights not found at %s. Running with RANDOM weights.", MODEL_PATH)

model.eval()

ACTION_MAPPING = {
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


def predict_action(state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(state_tensor)
        action_index = torch.argmax(probabilities).item()
    return ACTION_MAPPING.get(action_index, "unknown_action")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if "state" not in data:
        return jsonify({"error": "Missing 'state' in request"}), 400

    state = data["state"]
    if not isinstance(state, list) or len(state) != INPUT_SIZE:
        return jsonify({"error": f"'state' must be a list of length {INPUT_SIZE}"}), 400

    try:
        action = predict_action(state)
        return jsonify({"action": action}), 200
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "neural_network"}), 200


if __name__ == "__main__":
    logger.info("Starting NN service on port 5000...")
    app.run(host="0.0.0.0", port=5000)
