import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoders.text_encoder import TextEncoder
from ..encoders.audio_encoder import AudioEncoder
from ..encoders.video_encoder import VideoEncoder

class ModalityEncoder(nn.Module):
    """
    A simple projection network to map raw modality features
    (e.g., from wav2vec or RoBERTa) into a uniform hidden dimension.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class LatentBottleneck(nn.Module):
    """
    Fuses Text, Video, and Audio into a disentangled continuous latent space
    using the Reparameterization Trick.
    """
    def __init__(
        self,
        text_dim=768,       # e.g., RoBERTa base
        audio_dim=768,      # e.g., wav2vec 2.0
        video_dim=512,      # e.g., VisualFacet / ResNet
        hidden_dim=512,
        latent_dim=256
    ):
        super().__init__()

        # 1. Unimodal Encoders
        self.text_enc = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim)
        self.audio_enc = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim)
        self.video_enc = VideoEncoder(input_dim=video_dim, hidden_dim=hidden_dim)

        # 2. Multimodal Fusion (Concatenation -> MLP)
        fused_dim = hidden_dim * 3
        self.fusion_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. VAE Projections (Mean and Log-Variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_dim = latent_dim

    def encode(self, text, audio, video):
        """Passes modalities through encoders and predicts distribution parameters."""
        # Extract unimodal features
        t_feat = self.text_enc(text)
        a_feat = self.audio_enc(audio)
        v_feat = self.video_enc(video)

        # Fuse along the feature dimension
        # Expects inputs of shape: (Batch, Seq_Len, Dim)
        fused = torch.cat([t_feat, a_feat, v_feat], dim=-1)
        hidden = self.fusion_net(fused)

        # Predict parameters of the Gaussian distribution
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + epsilon * sigma
        Allows gradients to flow through the stochastic sampling node.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we just use the mean for deterministic behavior
            return mu

    def forward(self, text, audio, video):
        """
        Returns the sampled latent sequence `z` and the distribution parameters
        needed to compute the KL-Divergence / Total Correlation loss.
        """
        mu, logvar = self.encode(text, audio, video)
        z = self.reparameterize(mu, logvar)

        # z shape: (Batch, Seq_Len, Latent_Dim)
        # We permute it to (Batch, Latent_Dim, Seq_Len) so it fits into our 1D UNet
        z_permuted = z.permute(0, 2, 1)

        return z_permuted, mu, logvar

    @staticmethod
    def compute_kl_loss(mu, logvar, beta=5.0):
        """
        Computes the Kullback-Leibler Divergence against a standard normal prior N(0, I).
        By setting beta > 1 (Beta-VAE approach), we put heavier pressure on the
        bottleneck, which mathematically forces Total Correlation (disentanglement)
        between the latent variables.

        Math: $$D_{KL}(q(z|x) || p(z)) = -0.5 \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)$$
        """
        # Loss shape: (Batch, Seq_Len) -> reduced to scalar
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return beta * kl_div.mean()
