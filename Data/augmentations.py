import torch
import torch.nn as nn

class MultimodalAugmentation(nn.Module):
    """
    Applies on-the-fly data augmentation to cached multimodal tensors on the GPU.
    Crucial for preventing overfitting on small datasets like CMU-MOSEI.
    """
    def __init__(self, mask_prob=0.1, noise_std=0.01):
        super().__init__()
        self.mask_prob = mask_prob
        self.noise_std = noise_std

    def _temporal_masking(self, x):
        """Randomly masks out temporal frames (sets them to 0)."""
        # x shape: (Batch, Seq_Len, Dim)
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.mask_prob
        return x * mask.float()

    def _add_gaussian_noise(self, x):
        """Adds subtle jitter to continuous visual/acoustic features."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def forward(self, text, audio, video):
        if not self.training:
            return text, audio, video

        # Text: Only mask (don't add continuous noise to word embeddings)
        text_aug = self._temporal_masking(text)

        # Audio/Video: Mask and add noise
        audio_aug = self._add_gaussian_noise(self._temporal_masking(audio))
        video_aug = self._add_gaussian_noise(self._temporal_masking(video))

        return text_aug, audio_aug, video_aug
