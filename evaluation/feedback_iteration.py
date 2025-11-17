"""
Feedback Iteration Module
Applies user feedback to improve the dataset and model performance
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Process user feedback to improve datasets."""

    def __init__(self, dataset_path, feedback_path, output_path):
        """
        Initialize feedback processor.

        Args:
            dataset_path (str): Path to original dataset
            feedback_path (str): Path to user feedback file
            output_path (str): Path to save updated dataset
        """
        self.dataset_path = dataset_path
        self.feedback_path = feedback_path
        self.output_path = output_path
        self.dataset = None
        self.feedback = None

    def load_data(self):
        """Load dataset and feedback data."""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.dataset = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded: {len(self.dataset)} samples")

            if os.path.exists(self.feedback_path):
                logger.info(f"Loading feedback from {self.feedback_path}")
                self.feedback = pd.read_csv(self.feedback_path)
                logger.info(f"Feedback loaded: {len(self.feedback)} entries")
            else:
                logger.warning(f"Feedback file not found: {self.feedback_path}")
                self.feedback = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def apply_corrections(self):
        """Apply feedback corrections to dataset."""
        if self.feedback.empty:
            logger.info("No feedback to apply")
            return 0

        corrections_applied = 0
        logger.info("Applying feedback corrections...")

        try:
            for idx, row in self.feedback.iterrows():
                # Handle different feedback formats
                if 'predicted_action' in row and 'correct_action' in row:
                    # Find samples with matching predicted action
                    mask = self.dataset['action'] == row['predicted_action']
                    if mask.any():
                        # Update with correct action
                        self.dataset.loc[mask, 'action'] = row['correct_action']
                        corrections_applied += mask.sum()

            logger.info(f"Applied {corrections_applied} corrections")
            return corrections_applied

        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            raise

    def balance_dataset(self, strategy='oversample'):
        """
        Balance dataset to handle class imbalance.

        Args:
            strategy (str): 'oversample' or 'undersample'
        """
        logger.info(f"Balancing dataset using {strategy} strategy...")

        try:
            # Get class distribution
            class_counts = self.dataset['action'].value_counts()
            logger.info(f"Class distribution before balancing:\n{class_counts}")

            if strategy == 'oversample':
                # Oversample minority classes
                target_count = class_counts.max()
                balanced_parts = []

                for action_class in class_counts.index:
                    class_data = self.dataset[self.dataset['action'] == action_class]

                    if len(class_data) < target_count:
                        # Oversample
                        resampled = resample(
                            class_data,
                            replace=True,
                            n_samples=target_count,
                            random_state=42
                        )
                        balanced_parts.append(resampled)
                    else:
                        balanced_parts.append(class_data)

                self.dataset = pd.concat(balanced_parts, ignore_index=True)

            elif strategy == 'undersample':
                # Undersample majority classes
                target_count = class_counts.min()
                balanced_parts = []

                for action_class in class_counts.index:
                    class_data = self.dataset[self.dataset['action'] == action_class]
                    resampled = resample(
                        class_data,
                        replace=False,
                        n_samples=min(len(class_data), target_count),
                        random_state=42
                    )
                    balanced_parts.append(resampled)

                self.dataset = pd.concat(balanced_parts, ignore_index=True)

            # Shuffle dataset
            self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)

            # Log new distribution
            new_counts = self.dataset['action'].value_counts()
            logger.info(f"Class distribution after balancing:\n{new_counts}")

        except Exception as e:
            logger.error(f"Error balancing dataset: {e}")
            raise

    def save_dataset(self):
        """Save updated dataset."""
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.dataset.to_csv(self.output_path, index=False)
            logger.info(f"Updated dataset saved to {self.output_path}")
            logger.info(f"Total samples: {len(self.dataset)}")

            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.dataset),
                'num_classes': self.dataset['action'].nunique(),
                'class_distribution': self.dataset['action'].value_counts().to_dict()
            }

            metadata_path = self.output_path.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def process(self, balance_strategy='oversample'):
        """
        Run complete feedback processing pipeline.

        Args:
            balance_strategy (str): Balancing strategy to use
        """
        logger.info("Starting feedback processing pipeline...")

        self.load_data()
        corrections = self.apply_corrections()
        self.balance_dataset(strategy=balance_strategy)
        self.save_dataset()

        logger.info("Feedback processing complete!")
        return {
            'corrections_applied': corrections,
            'final_size': len(self.dataset),
            'output_path': self.output_path
        }


def main():
    """Main execution function."""
    # Paths
    DATASET_PATH = "data/processed/nn_dataset.csv"
    FEEDBACK_PATH = "feedback/user_feedback.csv"
    OUTPUT_PATH = "data/processed/updated_nn_dataset.csv"

    # Create processor and run
    processor = FeedbackProcessor(DATASET_PATH, FEEDBACK_PATH, OUTPUT_PATH)
    results = processor.process(balance_strategy='oversample')

    logger.info(f"Processing results: {results}")


if __name__ == "__main__":
    main()
