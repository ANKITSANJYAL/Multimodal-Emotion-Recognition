import torch
import torch.nn as nn
from .text_encoder import PositionalEncoding

class AudioEncoder(nn.Module):
    """Conformer-lite architecture for continuous acoustic features (COVAREP)."""
    def __init__(self, input_dim=74, hidden_dim=512, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()

        # Audio is continuous; Conv1d smoothing is mathematically superior here
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # Conv1d expects (Batch, Channels, Time)
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)
        x = x.permute(0, 2, 1) # Back to (Batch, Seq_Len, Hidden_Dim)

        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        return self.layer_norm(self.transformer(x))
