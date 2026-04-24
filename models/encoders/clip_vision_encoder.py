"""CLIP ViT-based video encoder for foundation-model-grade visual features.

Wraps HuggingFace CLIP ViT-B/16 vision encoder. Processes per-frame
embeddings through the frozen CLIP backbone, then applies a learnable
temporal Transformer to capture dynamics across frames.

Falls back to a Conv1d + Transformer path for pre-extracted features
(e.g., VisualFacet42 FAUs) when CLIP is unavailable.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CLIPVisionEncoder(nn.Module):
    """Pre-trained CLIP ViT encoder with temporal Transformer for video.

    Args:
        hidden_dim: Output hidden dimension for downstream fusion.
        model_name: HuggingFace CLIP model identifier.
        freeze_backbone: Whether to freeze the CLIP ViT parameters.
        num_temporal_layers: Number of temporal Transformer layers.
        dropout: Dropout rate.
        fallback_input_dim: If provided, adds a Conv1d + Transformer path
            for pre-extracted feature inputs (e.g., FAU 35-d).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze_backbone: bool = True,
        num_temporal_layers: int = 2,
        dropout: float = 0.1,
        fallback_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_backbone = False

        # Attempt to load the pre-trained CLIP vision backbone
        try:
            from transformers import CLIPVisionModel

            self.backbone = CLIPVisionModel.from_pretrained(model_name)
            self.backbone_dim = self.backbone.config.hidden_size  # 768
            self.has_backbone = True

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                logger.info(
                    "CLIP ViT backbone frozen (%s, dim=%d)",
                    model_name,
                    self.backbone_dim,
                )

            # Frame-level projection
            self.frame_proj = nn.Sequential(
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        except (ImportError, ValueError, OSError) as e:
            logger.warning(
                "CLIP ViT backbone unavailable (%s). "
                "Falling back to Conv1d+Transformer mode.", e
            )

        # Fallback path for pre-extracted features (FAU, etc.)
        self.fallback_input_dim = fallback_input_dim
        if fallback_input_dim is not None:
            self.fallback_conv = nn.Sequential(
                nn.Conv1d(fallback_input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Temporal Transformer: captures frame-to-frame dynamics
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=num_temporal_layers
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode video input to hidden representations.

        Args:
            x: Pre-extracted features (B, L, input_dim) for fallback mode.
            pixel_values: Raw pixel tensors (B, L, C, H, W) for CLIP mode.
                Each frame is processed independently through CLIP ViT.

        Returns:
            Hidden representations of shape (B, L, hidden_dim).
        """
        if self.has_backbone and pixel_values is not None:
            # Process each frame through CLIP ViT
            B, L, C, H, W = pixel_values.shape
            frames_flat = pixel_values.reshape(B * L, C, H, W)  # (B*L, C, H, W)
            outputs = self.backbone(pixel_values=frames_flat)
            # Use the pooled [CLS] output per frame
            frame_features = outputs.pooler_output  # (B*L, 768)
            frame_features = frame_features.reshape(B, L, -1)  # (B, L, 768)
            h = self.frame_proj(frame_features)  # (B, L, hidden_dim)
        elif self.fallback_input_dim is not None and x.dim() == 3:
            # Pre-extracted feature fallback (e.g., FAU)
            h = x.permute(0, 2, 1)  # (B, D, L)
            h = self.fallback_conv(h)  # (B, hidden_dim, L)
            h = h.permute(0, 2, 1)  # (B, L, hidden_dim)
        else:
            raise ValueError(
                f"CLIPVisionEncoder: unexpected input. Got x.shape={x.shape}, "
                f"pixel_values={'None' if pixel_values is None else pixel_values.shape}"
            )

        # Temporal modeling across frames
        h = self.temporal_transformer(h)
        return self.output_norm(h)  # (B, L, hidden_dim)
