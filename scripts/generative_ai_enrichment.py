import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_noise_to_features(features, noise_level=0.05):
    """
    Add Gaussian noise to feature vector.
    Args:
        features (numpy.ndarray): Original feature vector
        noise_level (float): Standard deviation of noise
    Returns:
        numpy.ndarray: Noisy features
    """
    noise = np.random.normal(0, noise_level, features.shape)
    noisy_features = features + noise
    # Clip to valid range
    noisy_features = np.clip(noisy_features, 0, 1)
    return noisy_features


def generate_augmented_sample(row, feature_columns, noise_level=0.05):
    """
    Generate an augmented sample by adding noise to features.
    Args:
        row (pd.Series): Original data row
        feature_columns (list): List of feature column names
        noise_level (float): Noise level for augmentation
    Returns:
        dict: Augmented sample
    """
    augmented = {}

    # Add noise to features
    for col in feature_columns:
        if col in row:
            original_value = row[col]
            augmented[col] = original_value + np.random.normal(0, noise_level)
        else:
            augmented[col] = row[col]

    # Keep action label the same
    augmented['action'] = row['action']

    return augmented


def enrich_dataset_with_generative_ai(input_file, output_file, enrichment_factor=2, noise_level=0.05):
    """
    Enrich a dataset by generating synthetic entries using data augmentation.
    Args:
        input_file (str): Path to the original dataset.
        output_file (str): Path to save the enriched dataset.
        enrichment_factor (int): How many synthetic entries to generate per original entry.
        noise_level (float): Amount of noise to add for augmentation
    """
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)

    # Identify feature columns (all except 'action' and 'frame')
    feature_columns = [col for col in df.columns if col not in ['action', 'frame']]

    logger.info(f"Original dataset size: {len(df)} samples")
    logger.info(f"Feature columns: {len(feature_columns)}")

    enriched_data = []

    # Add original samples
    for _, row in df.iterrows():
        enriched_data.append(row.to_dict())

    # Generate synthetic samples
    logger.info(f"Generating {enrichment_factor}x augmented samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        for _ in range(enrichment_factor):
            synthetic_row = generate_augmented_sample(row, feature_columns, noise_level)
            enriched_data.append(synthetic_row)

    enriched_df = pd.DataFrame(enriched_data)

    # Shuffle the dataset
    enriched_df = enriched_df.sample(frac=1, random_state=42).reset_index(drop=True)

    enriched_df.to_csv(output_file, index=False)
    logger.info(f"Enriched dataset saved to {output_file}")
    logger.info(f"Final dataset size: {len(enriched_df)} samples (enrichment: {len(enriched_df)/len(df):.2f}x)")

if __name__ == "__main__":
    input_file = "data/processed/nn_dataset.csv"
    output_file = "data/processed/enriched_dataset.csv"
    enrich_dataset_with_generative_ai(input_file, output_file)
