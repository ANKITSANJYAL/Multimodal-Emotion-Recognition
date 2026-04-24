"""Perceiver-style cross-modal attention fusion module.

Replaces naive concat+MLP with a learnable bottleneck that performs
cross-attention from bottleneck tokens to each modality, followed by
self-attention. This allows fine-grained cross-modal interaction while
keeping computation manageable via the bottleneck.

Reference: Jaegle et al. (2021) "Perceiver: General Perception with
Iterative Attention" (ICML).
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossModalAttentionFusion(nn.Module):
    """Perceiver-style cross-modal fusion with learned bottleneck tokens.

    Architecture:
        1. Project each modality to a common dim
        2. Concatenate all modality tokens into a single KV sequence
        3. Cross-attention: bottleneck queries attend to all modality tokens
        4. Self-attention: bottleneck tokens interact among themselves
        5. Output: fused representation at bottleneck token positions

    Args:
        hidden_dim: Common hidden dimension for all modalities.
        num_bottleneck_tokens: Number of learnable bottleneck tokens.
            If None, uses the input sequence length.
        num_heads: Number of attention heads.
        num_cross_attn_layers: Number of cross-attention layers.
        num_self_attn_layers: Number of self-attention layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_bottleneck_tokens: Optional[int] = None,
        num_heads: int = 8,
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens

        # Learnable bottleneck tokens (initialized if num_bottleneck_tokens is set)
        if num_bottleneck_tokens is not None:
            self.bottleneck_tokens = nn.Parameter(
                torch.randn(1, num_bottleneck_tokens, hidden_dim) * 0.02
            )
        else:
            self.bottleneck_tokens = None

        # Modality type embeddings to distinguish T/A/V tokens in the KV sequence
        self.modality_embeddings = nn.Embedding(3, hidden_dim)

        # Cross-attention layers: bottleneck queries → modality KV
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms_q = nn.ModuleList()
        self.cross_attn_norms_kv = nn.ModuleList()
        self.cross_ffn_layers = nn.ModuleList()
        self.cross_ffn_norms = nn.ModuleList()

        for _ in range(num_cross_attn_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.cross_attn_norms_q.append(nn.LayerNorm(hidden_dim))
            self.cross_attn_norms_kv.append(nn.LayerNorm(hidden_dim))
            self.cross_ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.cross_ffn_norms.append(nn.LayerNorm(hidden_dim))

        # Self-attention layers among bottleneck tokens
        self.self_attn_layers = nn.ModuleList()
        self.self_attn_norms = nn.ModuleList()
        self.self_ffn_layers = nn.ModuleList()
        self.self_ffn_norms = nn.ModuleList()

        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.self_attn_norms.append(nn.LayerNorm(hidden_dim))
            self.self_ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.self_ffn_norms.append(nn.LayerNorm(hidden_dim))

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal features via cross-modal attention.

        Args:
            text_feat: Text features (B, L, hidden_dim).
            audio_feat: Audio features (B, L, hidden_dim).
            video_feat: Video features (B, L, hidden_dim).

        Returns:
            Fused representation (B, L_out, hidden_dim) where L_out is
            num_bottleneck_tokens if set, else L (input seq length).
        """
        B, L, D = text_feat.shape
        device = text_feat.device

        # Add modality type embeddings
        text_typed = text_feat + self.modality_embeddings(
            torch.zeros(B, L, dtype=torch.long, device=device)
        )
        audio_typed = audio_feat + self.modality_embeddings(
            torch.ones(B, L, dtype=torch.long, device=device)
        )
        video_typed = video_feat + self.modality_embeddings(
            torch.full((B, L), 2, dtype=torch.long, device=device)
        )

        # Concatenate all modality tokens into KV sequence
        kv = torch.cat([text_typed, audio_typed, video_typed], dim=1)  # (B, 3L, D)

        # Initialize query tokens
        if self.bottleneck_tokens is not None:
            queries = self.bottleneck_tokens.expand(B, -1, -1)  # (B, N, D)
        else:
            # Use mean of modality features as initial queries
            queries = (text_feat + audio_feat + video_feat) / 3.0  # (B, L, D)

        # Cross-attention: queries attend to all modality tokens (pre-norm)
        # Attention is forced to fp32 regardless of AMP context: fp16 softmax
        # can underflow to all-zeros on randomly-initialized weights, causing
        # the downstream LayerNorm (var=0) to emit NaN on the very first step.
        for cross_attn, norm_q, norm_kv, ffn, ffn_norm in zip(
            self.cross_attn_layers,
            self.cross_attn_norms_q,
            self.cross_attn_norms_kv,
            self.cross_ffn_layers,
            self.cross_ffn_norms,
        ):
            orig_dtype = queries.dtype
            q_normed = norm_q(queries).float()
            kv_normed = norm_kv(kv).float()
            attn_out, _ = cross_attn(q_normed, kv_normed, kv_normed)
            attn_out = attn_out.to(orig_dtype)
            queries = queries + attn_out
            queries = queries + ffn(ffn_norm(queries))

        # Self-attention among bottleneck tokens (pre-norm)
        for self_attn, norm, ffn, ffn_norm in zip(
            self.self_attn_layers,
            self.self_attn_norms,
            self.self_ffn_layers,
            self.self_ffn_norms,
        ):
            orig_dtype = queries.dtype
            q_normed = norm(queries).float()
            attn_out, _ = self_attn(q_normed, q_normed, q_normed)
            attn_out = attn_out.to(orig_dtype)
            queries = queries + attn_out
            queries = queries + ffn(ffn_norm(queries))

        return self.output_proj(queries)  # (B, L_out, hidden_dim)
