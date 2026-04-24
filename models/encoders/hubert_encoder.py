"""HuBERT-based audio encoder for foundation-model-grade acoustic features.

Wraps HuggingFace HuBERT-base with a frozen backbone and learnable temporal
convolution + projection head. Supports both raw waveform input and
pre-extracted feature input (e.g., COVAREP 74-d) for backward compatibility.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HuBERTEncoder(nn.Module):
    """Pre-trained HuBERT encoder with frozen backbone and learnable projection.

    Args:
        hidden_dim: Output hidden dimension for downstream fusion.
        model_name: HuggingFace model identifier.
        freeze_backbone: Whether to freeze the HuBERT parameters.
        dropout: Dropout rate in the projection head.
        fallback_input_dim: If provided, adds a Conv1d + Transformer path
            for pre-extracted feature inputs (e.g., COVAREP 74-d).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        model_name: str = "facebook/hubert-base-ls960",
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        fallback_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_backbone = False

        # Attempt to load the pre-trained backbone
        try:
            from transformers import HubertModel

            self.backbone = HubertModel.from_pretrained(model_name)
            self.backbone_dim = self.backbone.config.hidden_size  # 768
            self.has_backbone = True

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                logger.info(
                    "HuBERT backbone frozen (%s, dim=%d)",
                    model_name,
                    self.backbone_dim,
                )
        except (ImportError, ValueError, OSError) as e:
            logger.warning(
                "HuBERT backbone unavailable (%s). "
                "Falling back to Conv1d+Transformer mode.", e
            )

        # Fallback path for pre-extracted features (COVAREP, etc.)
        self.fallback_input_dim = fallback_input_dim
        if fallback_input_dim is not None:
            self.fallback_conv = nn.Sequential(
                nn.Conv1d(fallback_input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fallback_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=3
            )
            self.fallback_norm = nn.LayerNorm(hidden_dim)

        # Projection from backbone dim to hidden_dim
        if self.has_backbone:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio input to hidden representations.

        Args:
            x: Either raw waveforms (B, T_samples) for HuBERT backbone,
               or pre-extracted features (B, L, input_dim) for fallback.
            attention_mask: Optional mask for waveform mode.

        Returns:
            Hidden representations of shape (B, L, hidden_dim).
        """
        if self.has_backbone and x.dim() == 2:
            # Raw waveform mode
            outputs = self.backbone(x, attention_mask=attention_mask)
            features = outputs.last_hidden_state  # (B, L', 768)
            return self.projection(features)  # (B, L', hidden_dim)

        if self.fallback_input_dim is not None and x.dim() == 3:
            # Pre-extracted feature fallback (e.g., COVAREP)
            h = x.permute(0, 2, 1)  # (B, D, L)
            h = self.fallback_conv(h)  # (B, hidden_dim, L)
            h = h.permute(0, 2, 1)  # (B, L, hidden_dim)
            h = self.fallback_transformer(h)
            return self.fallback_norm(h)  # (B, L, hidden_dim)

        raise ValueError(
            f"HuBERTEncoder: unexpected input shape {x.shape}. "
            f"Expected (B, T_samples) for waveform or (B, L, D) for features."
        )
