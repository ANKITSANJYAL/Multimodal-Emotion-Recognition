"""Legacy video encoder: Conv1d + Transformer for facial action unit sequences.

Used when encoder_type='legacy' is selected in config. Processes
pre-extracted features (e.g., VisualFacet42 FAUs 35-d) through local
convolution followed by a Transformer.
"""

import torch
import torch.nn as nn

from .text_encoder import PositionalEncoding


class VideoEncoder(nn.Module):
    """Conformer-lite for facial action unit sequences (VisualFacet42).

    Args:
        input_dim: Input feature dimension (e.g., 35 for FAU).
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 35,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
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
        """Encode video features through Conv1d + Transformer.

        Args:
            x: Video features (B, L, input_dim).

        Returns:
            Encoded features (B, L, hidden_dim).
        """
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.local_conv(x)
        x = x.permute(0, 2, 1)  # (B, L, hidden_dim)

        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        return self.layer_norm(self.transformer(x))
