import pandas as pd
from sklearn.utils import resample

# Paths to dataset and feedback
DATASET_PATH = "data/processed/nn_dataset.csv"
FEEDBACK_PATH = "feedback/user_feedback.csv"
OUTPUT_PATH = "data/processed/updated_nn_dataset.csv"

# Load dataset and feedback
dataset = pd.read_csv(DATASET_PATH)
feedback = pd.read_csv(FEEDBACK_PATH)

# Apply feedback corrections
print("Applying feedback corrections...")
for _, row in feedback.iterrows():
    frame_id = row["frame_id"]
    corrected_action = row["corrected_action"]
    dataset.loc[dataset["frame"] == frame_id, "action"] = corrected_action

# Balance dataset based on feedback
print("Balancing dataset...")
majority_class = dataset["action"].value_counts().idxmax()
minority_classes = dataset["action"].value_counts().index.drop(majority_class)

balanced_dataset = dataset[dataset["action"] == majority_class]
for cls in minority_classes:
    resampled = resample(
        dataset[dataset["action"] == cls],
        replace=True,
        n_samples=len(balanced_dataset),
        random_state=42
    )
    balanced_dataset = pd.concat([balanced_dataset, resampled])

# Save updated dataset
balanced_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"Updated dataset saved to {OUTPUT_PATH}")
