"""
Neural Network Model for AI Gameplay Bot
Implements a feedforward neural network for action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GameplayNN(nn.Module):
    """
    Feedforward Neural Network for gameplay action prediction.

    Args:
        input_size (int): Size of the input feature vector
        hidden_size (int): Size of hidden layers
        output_size (int): Number of possible actions
        dropout_rate (float): Dropout probability for regularization
    """

    def __init__(self, input_size=128, hidden_size=64, output_size=10, dropout_rate=0.3):
        super(GameplayNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden layers
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, output_size)
        """
        # First hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Third hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        return x

    def predict(self, state):
        """
        Predict action from a single state.

        Args:
            state (list or numpy.ndarray): Input state features

        Returns:
            int: Predicted action index
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            probabilities = self.forward(state)
            action_index = torch.argmax(probabilities, dim=1).item()
        return action_index

    def get_model_info(self):
        """
        Get model architecture information.

        Returns:
            dict: Model configuration and parameter count
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ConvGameplayNN(nn.Module):
    """
    Convolutional Neural Network for gameplay prediction from image frames.

    Args:
        num_classes (int): Number of possible actions
        input_channels (int): Number of input channels (3 for RGB)
    """

    def __init__(self, num_classes=10, input_channels=3):
        super(ConvGameplayNN, self).__init__()

        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Assuming input image size is 224x224, after 3 pooling layers: 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the convolutional network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == "__main__":
    # Test the models
    print("Testing GameplayNN...")
    model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
    print(model.get_model_info())

    # Test forward pass
    test_input = torch.randn(1, 128)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities: {output}")

    # Test prediction
    prediction = model.predict(test_input)
    print(f"Predicted action: {prediction}")

    print("\nTesting ConvGameplayNN...")
    conv_model = ConvGameplayNN(num_classes=10)
    test_image = torch.randn(1, 3, 224, 224)
    conv_output = conv_model(test_image)
    print(f"Conv output shape: {conv_output.shape}")
