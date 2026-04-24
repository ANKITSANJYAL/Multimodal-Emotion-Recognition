"""Multimodal VAE bottleneck with configurable encoders and fusion.

Supports legacy (GloVe/COVAREP/FAU) and foundation (RoBERTa/HuBERT/CLIP)
encoder backbones, as well as concat+MLP and Perceiver-style cross-modal
attention fusion methods — all selectable via config.
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.text_encoder import TextEncoder
from ..encoders.audio_encoder import AudioEncoder
from ..encoders.video_encoder import VideoEncoder
from ..encoders.roberta_encoder import RoBERTaEncoder
from ..encoders.hubert_encoder import HuBERTEncoder
from ..encoders.clip_vision_encoder import CLIPVisionEncoder
from ..causal_graph import CausalAttentionGraph
from .crossmodal_transformer import CrossModalAttentionFusion

logger = logging.getLogger(__name__)


class LatentBottleneck(nn.Module):
    """
    Pipeline role: ENCODER + FUSION + VAE BOTTLENECK + CAUSAL GRAPH

    Takes raw multimodal sequences and produces:
      - z_permuted : (B, d_z, L)  — the reparameterized latent, channel-first for the 1D UNet
      - mu         : (B, L, d_z)  — posterior mean,  needed for KL loss
      - logvar     : (B, L, d_z)  — posterior log-variance, needed for KL loss
      - adj_matrix : (B, 3, 3)    — causal adjacency matrix over {Text, Audio, Video}

    Math: implements q_xi(z | x^T, x^A, x^V) = N(z; mu_xi(x), diag(sigma^2_xi(x)))
    and the reparameterization trick z = mu + eps * sigma, eps ~ N(0, I).

    The CausalAttentionGraph runs in parallel on the *pre-fusion* unimodal latents
    so it sees each modality's independent representation before they are collapsed.
    """
    def __init__(
        self,
        text_dim: int = 300,
        audio_dim: int = 74,
        video_dim: int = 35,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        encoder_type: Literal["legacy", "foundation"] = "legacy",
        text_backbone: str = "roberta-base",
        audio_backbone: str = "facebook/hubert-base-ls960",
        video_backbone: str = "openai/clip-vit-base-patch16",
        freeze_backbones: bool = True,
        fusion_type: Literal["concat", "crossmodal"] = "concat",
        num_bottleneck_tokens: int = 50,
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        dag_method: Literal["gumbel", "notears"] = "notears",
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.fusion_type = fusion_type
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # ── 1. Unimodal Encoders  →  each outputs (B, L, hidden_dim) ─────
        if encoder_type == "foundation":
            logger.info("Using foundation model backbones (RoBERTa/HuBERT/CLIP)")
            self.text_enc = RoBERTaEncoder(
                hidden_dim=hidden_dim,
                model_name=text_backbone,
                freeze_backbone=freeze_backbones,
                fallback_input_dim=text_dim,
            )
            self.audio_enc = HuBERTEncoder(
                hidden_dim=hidden_dim,
                model_name=audio_backbone,
                freeze_backbone=freeze_backbones,
                fallback_input_dim=audio_dim,
            )
            self.video_enc = CLIPVisionEncoder(
                hidden_dim=hidden_dim,
                model_name=video_backbone,
                freeze_backbone=freeze_backbones,
                fallback_input_dim=video_dim,
            )
        else:
            logger.info("Using legacy encoders (Transformer/Conv1d)")
            self.text_enc = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim)
            self.audio_enc = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim)
            self.video_enc = VideoEncoder(input_dim=video_dim, hidden_dim=hidden_dim)

        # ── 2. Causal Graph — on unimodal reps BEFORE fusion ─────────────
        self.causal_graph = CausalAttentionGraph(
            latent_dim=hidden_dim, num_nodes=3, dag_method=dag_method,
        )

        # ── 3. Multimodal Fusion ─────────────────────────────────────────
        if fusion_type == "crossmodal":
            logger.info("Using Perceiver-style cross-modal attention fusion")
            self.fusion_net = CrossModalAttentionFusion(
                hidden_dim=hidden_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
                num_cross_attn_layers=num_cross_attn_layers,
                num_self_attn_layers=num_self_attn_layers,
            )
        else:
            logger.info("Using concat + MLP fusion")
            fused_dim = hidden_dim * 3
            self.fusion_net = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # ── 4. VAE projection heads ──────────────────────────────────────
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:  text  (B, L, text_dim)
                audio (B, L, audio_dim)
                video (B, L, video_dim)
        Output: mu     (B, L, latent_dim)
                logvar (B, L, latent_dim)
                t_feat (B, L, hidden_dim)  — unimodal hidden reps for causal graph
                a_feat (B, L, hidden_dim)
                v_feat (B, L, hidden_dim)
        """
        t_feat = self.text_enc(text)    # (B, L, hidden_dim)
        a_feat = self.audio_enc(audio)  # (B, L, hidden_dim)
        v_feat = self.video_enc(video)  # (B, L, hidden_dim)

        if self.fusion_type == "crossmodal":
            hidden = self.fusion_net(t_feat, a_feat, v_feat)  # (B, L_out, hidden_dim)
        else:
            fused = torch.cat([t_feat, a_feat, v_feat], dim=-1)  # (B, L, 3*hidden_dim)
            hidden = self.fusion_net(fused)                        # (B, L, hidden_dim)

        mu = self.fc_mu(hidden)                                # (B, L, latent_dim)
        logvar = self.fc_logvar(hidden)                        # (B, L, latent_dim)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        return mu, logvar, t_feat, a_feat, v_feat

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma, eps ~ N(0, I).
        Differentiable w.r.t. mu and logvar; randomness lives only in eps.
        At inference: deterministic (returns mu).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:  text  (B, L, text_dim)
                audio (B, L, audio_dim)
                video (B, L, video_dim)

        Output: z_permuted  (B, d_z, L)   ← channel-first for UNet1D
                mu          (B, L, d_z)   ← for KL loss
                logvar      (B, L, d_z)   ← for KL loss
                adj_matrix  (B, 3, 3)     ← causal graph over {T, A, V}
        """
        mu, logvar, t_feat, a_feat, v_feat = self.encode(text, audio, video)

        # Causal graph on unimodal hidden reps (B, L, H) — before collapse to latent
        adj_matrix = self.causal_graph(t_feat, a_feat, v_feat)  # (B, 3, 3)

        z = self.reparameterize(mu, logvar)   # (B, L, latent_dim)
        z_permuted = z.permute(0, 2, 1)       # (B, latent_dim, L) for UNet1D

        return z_permuted, mu, logvar, adj_matrix

    @staticmethod
    def compute_kl_loss(
        mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.5, free_bits: float = 0.25
    ) -> torch.Tensor:
        """
        Beta-VAE KL with correct Free Bits (Kingma et al. 2016).

        Free bits (λ): each latent dimension may encode up to λ nats of
        information without incurring any KL penalty. This prevents posterior
        collapse while ensuring that the KL loss is zero when the posterior
        matches the prior (no phantom floor).

        Implementation: KL = β × mean_batch( Σ_d max(0, kl_d - λ) )

        When kl_d < λ: the dimension is "free" → zero loss, zero gradient.
        When kl_d ≥ λ: the dimension pays (kl_d - λ) → normal gradient.

        Cast to float32 for numerical stability under AMP mixed precision.
        """
        mu_f32 = mu.float()
        logvar_f32 = logvar.float()
        # Per-dimension KL: (B, L, D)
        kl_per_dim = -0.5 * (1 + logvar_f32 - mu_f32.pow(2) - logvar_f32.exp())

        if free_bits > 0.0:
            # Threshold free-bits: only penalize KL above the free threshold
            # This ensures collapsed posterior → zero KL loss (not a floor)
            kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)

        # Sum over latent dim, mean over (B, L)
        kl_div = kl_per_dim.sum(dim=-1).mean()
        return beta * kl_div
