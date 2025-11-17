"""
Unit tests for Neural Network Model
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'neural_network'))

from nn_model import GameplayNN, ConvGameplayNN


class TestGameplayNN:
    """Test cases for GameplayNN model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        assert model.input_size == 128
        assert model.hidden_size == 64
        assert model.output_size == 10

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        test_input = torch.randn(1, 128)
        output = model(test_input)

        assert output.shape == (1, 10)
        assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-5)  # Check softmax

    def test_batch_processing(self):
        """Test model can process batches."""
        model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        batch_size = 32
        test_input = torch.randn(batch_size, 128)
        output = model(test_input)

        assert output.shape == (batch_size, 10)

    def test_predict_method(self):
        """Test the predict method."""
        model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        test_state = np.random.rand(128)
        action = model.predict(test_state)

        assert isinstance(action, int)
        assert 0 <= action < 10

    def test_get_model_info(self):
        """Test model info retrieval."""
        model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        info = model.get_model_info()

        assert 'input_size' in info
        assert 'hidden_size' in info
        assert 'output_size' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0


class TestConvGameplayNN:
    """Test cases for ConvGameplayNN model."""

    def test_conv_model_initialization(self):
        """Test convolutional model initialization."""
        model = ConvGameplayNN(num_classes=10, input_channels=3)
        assert model.num_classes == 10

    def test_conv_forward_pass(self):
        """Test forward pass with image input."""
        model = ConvGameplayNN(num_classes=10, input_channels=3)
        test_image = torch.randn(1, 3, 224, 224)
        output = model(test_image)

        assert output.shape == (1, 10)
        assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_conv_batch_processing(self):
        """Test batch processing for convolutional model."""
        model = ConvGameplayNN(num_classes=10, input_channels=3)
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 224, 224)
        output = model(test_images)

        assert output.shape == (batch_size, 10)


if __name__ == '__main__':
    pytest.main([__file__])
