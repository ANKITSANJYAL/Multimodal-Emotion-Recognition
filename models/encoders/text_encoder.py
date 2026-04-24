"""Legacy text encoder: Transformer with sinusoidal positional encoding.

Used when encoder_type='legacy' is selected in config. Processes
pre-extracted word embeddings (e.g., GloVe 300-d) through a Transformer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs.

    Args:
        d_model: Model dimension.
        dropout: Dropout rate.
        max_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout.

        Args:
            x: Input tensor (L, B, d_model) in time-first format.

        Returns:
            Positionally-encoded tensor (L, B, d_model).
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """Transformer encoder for discrete text token embeddings.

    Args:
        input_dim: Input embedding dimension (e.g., 300 for GloVe).
        hidden_dim: Transformer model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode text features through Transformer.

        Args:
            x: Text embeddings (B, L, input_dim).

        Returns:
            Encoded features (B, L, hidden_dim).
        """
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        return self.layer_norm(self.transformer(x))
