"""
Neural Network Training Module
Handles training, validation, and model saving for GameplayNN
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from nn_model import GameplayNN


class GameplayDataset(Dataset):
    """
    Custom Dataset for loading gameplay data.

    Args:
        csv_path (str): Path to the dataset CSV file
        transform (callable, optional): Optional transform to apply to samples
    """

    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Separate features and labels
        self.features = self.data.drop('action', axis=1).values.astype(np.float32)
        self.labels = self.data['action'].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class NNTrainer:
    """
    Trainer class for GameplayNN model.

    Args:
        model (GameplayNN): The neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cuda' or 'cpu')
    """

    def __init__(self, model, train_loader, val_loader=None, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(self.train_loader, desc="Training"):
            features, labels = features.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model on validation set.

        Returns:
            tuple: (average_loss, accuracy)
        """
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation"):
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs, save_dir="models/neural_network", model_name="nn_model.pth"):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs (int): Number of training epochs
            save_dir (str): Directory to save the best model
            model_name (str): Name of the saved model file

        Returns:
            dict: Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            # Validate
            if self.val_loader:
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(save_dir, model_name)
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Model saved to {model_path}")

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat()
        }

        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining completed! History saved to {history_path}")

        return history


def main():
    """
    Main training script.
    """
    # Configuration
    DATASET_PATH = "data/processed/nn_dataset.csv"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    INPUT_SIZE = 128
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 10
    VAL_SPLIT = 0.2

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    full_dataset = GameplayDataset(DATASET_PATH)

    # Split into train and validation
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Initialize model
    model = GameplayNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE
    )

    # Initialize trainer
    trainer = NNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # Train model
    history = trainer.train(num_epochs=NUM_EPOCHS)

    print("\nTraining Summary:")
    print(f"Best validation loss: {min(history['val_losses']):.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracies']):.2f}%")


if __name__ == "__main__":
    main()
