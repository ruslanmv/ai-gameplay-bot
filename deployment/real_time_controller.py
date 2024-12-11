import requests
import numpy as np

# Configuration
NN_API_URL = "http://localhost:5000/predict"
TRANSFORMER_API_URL = "http://localhost:5001/predict"

def get_action_from_nn(state):
    """
    Get the predicted action from the Neural Network model.
    Args:
        state (list): Input state features.
    Returns:
        str: Predicted action.
    """
    response = requests.post(NN_API_URL, json={"state": state})
    if response.status_code == 200:
        return response.json().get("action", "unknown_action")
    else:
        print(f"Error from NN API: {response.json()}")
        return "error"

def get_action_from_transformer(state):
    """
    Get the predicted action from the Transformer model.
    Args:
        state (list): Input state features.
    Returns:
        str: Predicted action.
    """
    response = requests.post(TRANSFORMER_API_URL, json={"state": state})
    if response.status_code == 200:
        return response.json().get("action", "unknown_action")
    else:
        print(f"Error from Transformer API: {response.json()}")
        return "error"

def unified_predictor(state, use_transformer=True):
    """
    Unified predictor that uses either the Neural Network or Transformer model.
    Args:
        state (list): Input state features.
        use_transformer (bool): Whether to use the Transformer model.
    Returns:
        str: Predicted action.
    """
    if use_transformer:
        return get_action_from_transformer(state)
    else:
        return get_action_from_nn(state)

if __name__ == "__main__":
    # Example usage
    example_state = list(np.random.rand(128))  # Randomly generated state

    print("Using Neural Network:")
    action_nn = unified_predictor(example_state, use_transformer=False)
    print(f"Predicted Action (NN): {action_nn}")

    print("\nUsing Transformer:")
    action_transformer = unified_predictor(example_state, use_transformer=True)
    print(f"Predicted Action (Transformer): {action_transformer}")
