"""
Generate Sample Data for AI Gameplay Bot
Creates synthetic frames, annotations, and datasets for testing
"""

import os
import numpy as np
import pandas as pd
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_frames(output_dir="data/processed/frames", num_frames=30):
    """
    Generate synthetic game frames for testing.
    Args:
        output_dir (str): Directory to save frames
        num_frames (int): Number of frames to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating {num_frames} sample frames...")

    for i in range(num_frames):
        # Create a synthetic image (random colored noise)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        # Add some structure (rectangles, circles) to make it more interesting
        color1 = tuple(np.random.randint(0, 256, 3).tolist())
        color2 = tuple(np.random.randint(0, 256, 3).tolist())

        cv2.rectangle(img, (50, 50), (150, 150), color1, -1)
        cv2.circle(img, (112, 112), 30, color2, -1)

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, img)

    logger.info(f"Generated {num_frames} frames in {output_dir}")


def generate_sample_dataset(output_file="data/processed/nn_dataset.csv", num_samples=30):
    """
    Generate a sample dataset with features and actions.
    Args:
        output_file (str): Path to save the dataset
        num_samples (int): Number of samples to generate
    """
    logger.info(f"Generating sample dataset with {num_samples} samples...")

    # Generate random features (128 dimensions)
    features = np.random.rand(num_samples, 128)

    # Generate random actions (0-9)
    actions = np.random.randint(0, 10, num_samples)

    # Create DataFrame
    data = {f"feature_{i}": features[:, i] for i in range(features.shape[1])}
    data["action"] = actions

    df = pd.DataFrame(data)

    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info(f"Sample dataset saved to {output_file}")


def generate_transformer_dataset(output_file="data/processed/transformer_dataset.csv", num_samples=100):
    """
    Generate a sample dataset for transformer training (larger).
    Args:
        output_file (str): Path to save the dataset
        num_samples (int): Number of samples to generate
    """
    logger.info(f"Generating transformer dataset with {num_samples} samples...")

    # Generate random features (128 dimensions)
    features = np.random.rand(num_samples, 128)

    # Generate random actions (0-9) with some temporal correlation
    actions = []
    current_action = np.random.randint(0, 10)
    for i in range(num_samples):
        # 70% chance to keep same action, 30% chance to switch
        if np.random.rand() > 0.7:
            current_action = np.random.randint(0, 10)
        actions.append(current_action)

    # Create DataFrame
    data = {f"feature_{i}": features[:, i] for i in range(features.shape[1])}
    data["action"] = actions

    df = pd.DataFrame(data)

    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info(f"Transformer dataset saved to {output_file}")


def generate_feedback_data(output_file="feedback/user_feedback.csv", num_samples=20):
    """
    Generate sample user feedback data.
    Args:
        output_file (str): Path to save feedback data
        num_samples (int): Number of feedback samples
    """
    logger.info(f"Generating sample feedback data...")

    actions = ["move_forward", "attack", "jump", "turn_left", "turn_right",
               "interact", "use_item", "cast_spell", "move_backward", "open_inventory"]

    feedback_data = {
        "predicted_action": np.random.choice(actions, num_samples),
        "correct_action": np.random.choice(actions, num_samples),
        "confidence": np.random.rand(num_samples),
        "timestamp": pd.date_range(start='2024-01-01', periods=num_samples, freq='H')
    }

    df = pd.DataFrame(feedback_data)

    # Calculate if prediction was correct
    df['is_correct'] = (df['predicted_action'] == df['correct_action']).astype(int)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info(f"Feedback data saved to {output_file}")


def generate_all_sample_data():
    """
    Generate all sample data needed for the project.
    """
    logger.info("=== Generating All Sample Data ===")

    # Generate frames
    generate_sample_frames(num_frames=30)

    # Generate datasets
    generate_sample_dataset(num_samples=30)
    generate_transformer_dataset(num_samples=100)

    # Generate feedback data
    generate_feedback_data(num_samples=20)

    logger.info("\n=== Sample Data Generation Complete ===")
    logger.info("You can now run the training and deployment scripts!")


if __name__ == "__main__":
    generate_all_sample_data()
