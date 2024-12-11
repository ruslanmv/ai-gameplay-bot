import torch
import numpy as np
from flask import Flask, request, jsonify
from nn_model import GameplayNN

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Neural Network model
MODEL_PATH = "models/neural_network/nn_model_finetuned.pth"
INPUT_SIZE = 128
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10

model = GameplayNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Action mapping (example, update as needed)
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
    9: "cast_spell"
}

def predict_action(state):
    """
    Predict the action based on the input state using the pre-trained model.
    Args:
        state (list): Input state features.
    Returns:
        str: Predicted action.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(state_tensor)
        action_index = torch.argmax(probabilities).item()
    return ACTION_MAPPING.get(action_index, "unknown_action")

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict the action from a given state.
    Input JSON format:
        {
            "state": [feature1, feature2, ..., feature128]
        }
    Returns:
        JSON response with the predicted action.
    """
    data = request.get_json()
    if "state" not in data:
        return jsonify({"error": "Missing 'state' in request"}), 400

    state = data["state"]
    if not isinstance(state, list) or len(state) != INPUT_SIZE:
        return jsonify({"error": f"'state' must be a list of length {INPUT_SIZE}"}), 400

    try:
        action = predict_action(state)
        return jsonify({"action": action}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server
    app.run(host="0.0.0.0", port=5000)
