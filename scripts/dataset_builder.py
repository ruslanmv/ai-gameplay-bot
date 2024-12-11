import pandas as pd
import os

def build_dataset(frames_dir, actions_file, output_file):
    """
    Build a dataset from extracted frames and mapped actions.
    Args:
        frames_dir (str): Directory containing extracted frames.
        actions_file (str): Path to the file containing actions.
        output_file (str): Path to save the resulting dataset.
    """
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    actions = []

    with open(actions_file, "r") as f:
        actions = [line.strip() for line in f.readlines()]

    if len(frames) != len(actions):
        raise ValueError("Number of frames and actions must match.")

    data = {
        "frame": frames,
        "action": actions
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    frames_dir = "data/processed/frames"
    actions_file = "data/raw/annotations/actions.txt"
    output_file = "data/processed/nn_dataset.csv"
    build_dataset(frames_dir, actions_file, output_file)
