"""
Transformer Fine-Tuning Module
Fine-tune pre-trained Transformer models on specific datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from transformer_model import GameplayTransformer
from transformer_training import SequenceGameplayDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def finetune_transformer_model(model_path, dataset_path, output_model_path, epochs=5, batch_size=16, lr=0.0001):
    """
    Fine-tune the Transformer model for real-time performance.

    Args:
        model_path (str): Path to the pre-trained model
        dataset_path (str): Path to the fine-tuning dataset
        output_model_path (str): Path to save the fine-tuned model
        epochs (int): Number of fine-tuning epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate

    Returns:
        dict: Fine-tuning results
    """
    logger.info(f"Starting Transformer fine-tuning from {model_path}")
    logger.info(f"Dataset: {dataset_path}")

    try:
        # Load dataset
        dataset = SequenceGameplayDataset(dataset_path, sequence_length=10)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Loaded {len(dataset)} sequences for fine-tuning")

        # Load pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GameplayTransformer(
            input_dim=128,
            num_classes=10,
            num_heads=4,
            num_layers=3,
            hidden_dim=256,
            sequence_length=10
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}. Using random initialization.")

        model = model.to(device)
        logger.info(f"Model loaded on {device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Fine-tuning loop
        model.train()
        best_accuracy = 0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for sequences, actions in dataloader:
                sequences, actions = sequences.to(device), actions.to(device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, actions)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()

            # Step scheduler
            scheduler.step()

            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                       f"Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}")

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), output_model_path)
                logger.info(f"New best model saved with accuracy: {accuracy:.2f}%")

        logger.info(f"Fine-tuning complete! Best accuracy: {best_accuracy:.2f}%")

        return {
            'final_loss': avg_loss,
            'final_accuracy': accuracy,
            'best_accuracy': best_accuracy,
            'output_path': output_model_path
        }

    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


if __name__ == "__main__":
    results = finetune_transformer_model(
        model_path="models/transformer/transformer_model.pth",
        dataset_path="data/processed/transformer_dataset.csv",
        output_model_path="models/transformer/transformer_model_finetuned.pth",
        epochs=10,
        batch_size=16,
        lr=0.0001
    )

    logger.info(f"Fine-tuning results: {results}")
