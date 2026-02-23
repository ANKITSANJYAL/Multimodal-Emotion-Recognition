import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TextEncoder(nn.Module):
    """SOTA Transformer specifically for discrete Text token embeddings."""
    def __init__(self, input_dim=300, hidden_dim=512, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        # Text is discrete; Linear projection is optimal here
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        x = self.input_proj(x)
        x = x.transpose(0, 1) # Prep for PE
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # Back to batch_first

        return self.layer_norm(self.transformer(x))
