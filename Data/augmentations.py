"""On-the-fly multimodal data augmentation for GPU-side processing.

Applies temporal masking and Gaussian noise injection to combat
overfitting on small datasets like CMU-MOSEI.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MultimodalAugmentation(nn.Module):
    """GPU-side multimodal data augmentation.

    Applies temporal masking (random frame dropout) and Gaussian noise
    injection to audio/video features. Text only receives masking
    (noise on discrete embeddings is not meaningful).

    Args:
        mask_prob: Probability of masking each temporal frame.
        noise_std: Standard deviation of Gaussian noise for audio/video.
    """

    def __init__(self, mask_prob: float = 0.1, noise_std: float = 0.01) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.noise_std = noise_std

    def _temporal_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask temporal frames (set to zero).

        Args:
            x: Input tensor (B, L, D).

        Returns:
            Masked tensor (B, L, D).
        """
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.mask_prob
        return x * mask.float()

    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian jitter to continuous features.

        Args:
            x: Input tensor (B, L, D).

        Returns:
            Noised tensor (B, L, D).
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentation to multimodal inputs (train mode only).

        Args:
            text: Text features (B, L, text_dim).
            audio: Audio features (B, L, audio_dim).
            video: Video features (B, L, video_dim).

        Returns:
            Tuple of augmented (text, audio, video).
        """
        if not self.training:
            return text, audio, video

        text_aug = self._temporal_masking(text)
        audio_aug = self._add_gaussian_noise(self._temporal_masking(audio))
        video_aug = self._add_gaussian_noise(self._temporal_masking(video))

        return text_aug, audio_aug, video_aug
