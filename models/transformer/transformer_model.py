import torch
import torch.nn as nn


class GameplayTransformer(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, output_size):
        """
        Transformer-based model for gameplay action prediction.

        Expected input:
          - (batch, input_size)            -> treated as single-timestep sequence
          - (batch, seq_len, input_size)   -> sequence input

        Output:
          - (batch, output_size) probabilities
        """
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,  # IMPORTANT: makes input (batch, seq, hidden)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - (B, F) or (B, T, F)
        returns:
          - (B, C)
        """
        if x.dim() == 2:
            # (B, F) -> (B, 1, F)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected input of shape (B,F) or (B,T,F), got {tuple(x.shape)}")

        x = self.embedding(x)                  # (B, T, H)
        x = self.transformer_encoder(x)        # (B, T, H)
        x = x.mean(dim=1)                      # pool over T -> (B, H)
        logits = self.fc(x)                    # (B, C)
        return logits
