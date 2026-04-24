"""RoBERTa-based text encoder for foundation-model-grade feature extraction.

Wraps HuggingFace RoBERTa-base with a frozen backbone and a learnable
projection head to produce hidden representations compatible with the
LatentBottleneck fusion pipeline.

Supports two modes:
  1. Token-ID mode: receives raw token IDs (B, L) and runs the full RoBERTa
  2. Embedding mode: receives pre-extracted embeddings (B, L, D) and projects
     them through the projection head only (backward-compatible with GloVe)
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RoBERTaEncoder(nn.Module):
    """Pre-trained RoBERTa encoder with frozen backbone and learnable projection.

    Args:
        hidden_dim: Output hidden dimension for downstream fusion.
        model_name: HuggingFace model identifier.
        freeze_backbone: Whether to freeze the RoBERTa parameters.
        num_projection_layers: Depth of the projection MLP.
        dropout: Dropout rate in the projection head.
        fallback_input_dim: If provided, adds a fallback linear projection
            for pre-extracted embedding inputs (e.g., GloVe 300-d).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        model_name: str = "roberta-base",
        freeze_backbone: bool = True,
        num_projection_layers: int = 2,
        dropout: float = 0.1,
        fallback_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.has_backbone = False

        # Attempt to load the pre-trained backbone
        try:
            from transformers import RobertaModel

            self.backbone = RobertaModel.from_pretrained(model_name)
            self.backbone_dim = self.backbone.config.hidden_size  # 768 for base
            self.has_backbone = True

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                logger.info(
                    "RoBERTa backbone frozen (%s, dim=%d)",
                    model_name,
                    self.backbone_dim,
                )
            else:
                logger.info(
                    "RoBERTa backbone trainable (%s, dim=%d)",
                    model_name,
                    self.backbone_dim,
                )
        except (ImportError, ValueError, OSError) as e:
            logger.warning(
                "RoBERTa backbone unavailable (%s). "
                "Falling back to projection-only mode.", e
            )
            self.backbone_dim = fallback_input_dim or 300

        # Fallback projection for pre-extracted embeddings (e.g., GloVe)
        if fallback_input_dim is not None:
            self.fallback_proj = nn.Linear(fallback_input_dim, self.backbone_dim)
        else:
            self.fallback_proj = None

        # Learnable projection head: backbone_dim → hidden_dim
        layers = []
        in_dim = self.backbone_dim
        for i in range(num_projection_layers):
            out_dim = hidden_dim if i == num_projection_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.projection = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text input to hidden representations.

        Args:
            x: Either token IDs (B, L) of dtype long, or pre-extracted
               embeddings (B, L, D) of dtype float.
            attention_mask: Optional mask for token-ID mode (B, L).

        Returns:
            Hidden representations of shape (B, L, hidden_dim).
        """
        if self.has_backbone and x.dtype == torch.long:
            # Token-ID mode: run full RoBERTa
            outputs = self.backbone(
                input_ids=x,
                attention_mask=attention_mask,
            )
            features = outputs.last_hidden_state  # (B, L, 768)
        elif self.fallback_proj is not None:
            # Pre-extracted embedding mode (e.g., GloVe)
            features = self.fallback_proj(x)  # (B, L, backbone_dim)
        else:
            features = x  # Assume x is already at backbone_dim

        return self.projection(features)  # (B, L, hidden_dim)
