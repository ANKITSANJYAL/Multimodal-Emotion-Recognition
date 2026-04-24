"""Lightweight per-modality reconstruction decoder for VAE completeness.

Maps latent representations back to the original feature dimensions for
each modality, enabling a proper reconstruction loss (MSE) that ensures
the VAE latent space captures meaningful information about the input.

Without reconstruction, the KL term alone may drive posterior collapse.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultimodalDecoder(nn.Module):
    """Per-modality MLP decoder: z → (x_hat_text, x_hat_audio, x_hat_video).

    Each modality gets its own 2-layer MLP to reconstruct the original
    feature dimensions from the shared latent space.

    Args:
        latent_dim: Dimension of the latent space.
        text_dim: Original text feature dimension (e.g., 300 for GloVe).
        audio_dim: Original audio feature dimension (e.g., 74 for COVAREP).
        video_dim: Original video feature dimension (e.g., 35 for FAU).
        hidden_dim: Intermediate decoder hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 300,
        audio_dim: int = 74,
        video_dim: int = 35,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.text_decoder = self._build_decoder(latent_dim, hidden_dim, text_dim, dropout)
        self.audio_decoder = self._build_decoder(latent_dim, hidden_dim, audio_dim, dropout)
        self.video_decoder = self._build_decoder(latent_dim, hidden_dim, video_dim, dropout)

        logger.info(
            "MultimodalDecoder initialized: latent=%d → text=%d, audio=%d, video=%d",
            latent_dim, text_dim, audio_dim, video_dim,
        )

    @staticmethod
    def _build_decoder(
        in_dim: int, hidden_dim: int, out_dim: int, dropout: float
    ) -> nn.Sequential:
        """Build a 2-layer MLP decoder for a single modality."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent representations to per-modality reconstructions.

        Args:
            z: Latent representation (B, L, latent_dim).

        Returns:
            Tuple of (x_hat_text, x_hat_audio, x_hat_video), each (B, L, modality_dim).
        """
        return (
            self.text_decoder(z),
            self.audio_decoder(z),
            self.video_decoder(z),
        )

    @staticmethod
    def compute_reconstruction_loss(
        x_hat_text: torch.Tensor,
        x_hat_audio: torch.Tensor,
        x_hat_video: torch.Tensor,
        x_text: torch.Tensor,
        x_audio: torch.Tensor,
        x_video: torch.Tensor,
    ) -> torch.Tensor:
        """Compute aggregate MSE reconstruction loss across all modalities.

        Args:
            x_hat_text: Reconstructed text features (B, L, text_dim).
            x_hat_audio: Reconstructed audio features (B, L, audio_dim).
            x_hat_video: Reconstructed video features (B, L, video_dim).
            x_text: Original text features (B, L, text_dim).
            x_audio: Original audio features (B, L, audio_dim).
            x_video: Original video features (B, L, video_dim).

        Returns:
            Scalar reconstruction loss (mean MSE across all modalities).
        """
        # Cast to float32 for numerical stability under AMP
        loss_text = F.mse_loss(x_hat_text.float(), x_text.float())
        loss_audio = F.mse_loss(x_hat_audio.float(), x_audio.float())
        loss_video = F.mse_loss(x_hat_video.float(), x_video.float())
        return (loss_text + loss_audio + loss_video) / 3.0
