import torch
import torch.nn as nn

class GameplayTransformer(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, output_size):
        """
        Transformer-based model for gameplay action prediction.
        Args:
            input_size (int): Size of input features.
            num_heads (int): Number of attention heads.
            hidden_size (int): Dimension of the hidden layer.
            num_layers (int): Number of transformer layers.
            output_size (int): Number of possible actions (classes).
        """
        super(GameplayTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)  # Project input to hidden dimension
        x = self.transformer_encoder(x)  # Pass through transformer layers
        x = self.fc(x.mean(dim=0))  # Pooling and output
        x = self.softmax(x)  # Convert to probabilities
        return x
