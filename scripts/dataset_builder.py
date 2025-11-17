import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_image_features(image_path, target_size=(224, 224)):
    """
    Extract features from an image.
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    Returns:
        numpy.ndarray: Flattened image features
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return np.zeros(target_size[0] * target_size[1] // 64)  # Return dummy features

        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Downsample for feature vector
        img = cv2.resize(img, (target_size[0] // 8, target_size[1] // 8))
        features = img.flatten() / 255.0

        return features
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return np.zeros(target_size[0] * target_size[1] // 64)


def map_action_to_index(action_str):
    """
    Map action string to index.
    Args:
        action_str (str): Action string
    Returns:
        int: Action index
    """
    action_mapping = {
        "move_forward": 0,
        "move_backward": 1,
        "turn_left": 2,
        "turn_right": 3,
        "attack": 4,
        "jump": 5,
        "interact": 6,
        "use_item": 7,
        "open_inventory": 8,
        "cast_spell": 9
    }
    return action_mapping.get(action_str.lower().strip(), 0)


def build_dataset(frames_dir, actions_file, output_file, extract_features=True):
    """
    Build a dataset from extracted frames and mapped actions.
    Args:
        frames_dir (str): Directory containing extracted frames.
        actions_file (str): Path to the file containing actions.
        output_file (str): Path to save the resulting dataset.
        extract_features (bool): Whether to extract image features or just save paths
    """
    logger.info(f"Building dataset from {frames_dir}")

    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    actions = []

    with open(actions_file, "r") as f:
        actions = [line.strip() for line in f.readlines()]

    if len(frames) != len(actions):
        logger.warning(f"Frame count ({len(frames)}) != action count ({len(actions)})")
        # Truncate to minimum length
        min_len = min(len(frames), len(actions))
        frames = frames[:min_len]
        actions = actions[:min_len]

    if extract_features:
        logger.info("Extracting features from frames...")
        features_list = []
        action_indices = []

        for frame_path, action in tqdm(zip(frames, actions), total=len(frames)):
            features = extract_image_features(frame_path)
            action_idx = map_action_to_index(action)

            features_list.append(features)
            action_indices.append(action_idx)

        # Create DataFrame with features
        features_array = np.array(features_list)
        data = {f"feature_{i}": features_array[:, i] for i in range(features_array.shape[1])}
        data["action"] = action_indices

        df = pd.DataFrame(data)
    else:
        # Simple dataset with paths
        action_indices = [map_action_to_index(a) for a in actions]
        data = {
            "frame": frames,
            "action": action_indices
        }
        df = pd.DataFrame(data)

    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Dataset saved to {output_file} ({len(df)} samples)")

if __name__ == "__main__":
    frames_dir = "data/processed/frames"
    actions_file = "data/raw/annotations/actions.txt"
    output_file = "data/processed/nn_dataset.csv"
    build_dataset(frames_dir, actions_file, output_file)
