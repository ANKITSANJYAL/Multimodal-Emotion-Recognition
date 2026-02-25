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


from ..causal_graph import CausalAttentionGraph

class LatentBottleneck(nn.Module):
    """
    Fuses Text, Video, and Audio into a disentangled continuous latent space
    using the Reparameterization Trick, guided by a Causal Attention Graph.
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

        # Ensure deterministic initialization
        torch.manual_seed(42)

        # 1. Unimodal Encoders
        self.text_enc = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim)
        self.audio_enc = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim)
        self.video_enc = VideoEncoder(input_dim=video_dim, hidden_dim=hidden_dim)

        # 2. Causal Architecture (DeepMind Level Interpretability)
        self.causal_graph = CausalAttentionGraph(latent_dim=hidden_dim, num_nodes=3)
        
        # 3. Multimodal Fusion (Causal-weighted)
        # We learn how to integrate modalities based on their causal influence
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

        # 4. VAE Projections Headers (Normalized for stability)
        self.header_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        """Research-grade parameter initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, text, audio, video):
        """Passes modalities through encoders and predicts distribution parameters."""
        # Extract unimodal features: (Batch, Seq_Len, hidden_dim)
        t_feat = self.text_enc(text)
        a_feat = self.audio_enc(audio)
        v_feat = self.video_enc(video)

        # Step A: Compute Causal Adjacency Matrix
        # This tells us which modalities are most influential in the current batch
        adj_matrix = self.causal_graph(t_feat, a_feat, v_feat)
        
        # Step B: Causal-Weighted Fusion
        # Calculate importance of each modality node (sum of incoming causal edges)
        # Shape: (Batch, 3)
        causal_influence = adj_matrix.sum(dim=1) 
        causal_weights = F.softmax(causal_influence, dim=-1) # (Batch, 3)

        # Weighted combination of temporal sequences
        # (Batch, 3, 1, 1) * (Batch, 3, Seq_Len, Hidden_Dim)
        stacked_feats = torch.stack([t_feat, a_feat, v_feat], dim=1)
        weighted_fused = (stacked_feats * causal_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        # Final projection before VAE bottleneck
        fused = self.fusion_proj(weighted_fused)
        fused = self.header_norm(fused)

        # Predict parameters of the Gaussian distribution
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        
        # Stability fix: Clamp logvar to prevent NaNs during exp() or KL calculation
        logvar = torch.clamp(logvar, min=-10.0, max=2.0) 
        
        return mu, logvar, adj_matrix

    def reparameterize(self, mu, logvar):
        """Standard Reparameterization Trick with stability check."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return torch.clamp(z, min=-10.0, max=10.0)

    def forward(self, text, audio, video):
        """
        Returns the sampled latent sequence `z` and the causal graph metadata.
        """
        mu, logvar, adj_matrix = self.encode(text, audio, video)
        
        # Sample from the posterior
        z = self.reparameterize(mu, logvar)
        
        # z shape: (Batch, Seq_Len, latent_dim)
        z_permuted = z.permute(0, 2, 1)
        return z_permuted, mu, logvar, adj_matrix

    @staticmethod
    def compute_kl_loss(mu, logvar, beta=5.0):
        """
        Math: $$D_{KL}(q(z|x) || p(z)) = -0.5 \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)$$
        """
        # Loss shape: (Batch, Seq_Len) -> reduced to scalar
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=100.0), dim=-1)
        return beta * kl_div.mean()
