import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Injects the diffusion timestep 't' into the network as a continuous embedding.
    Maps scalar t → R^dim via sine/cosine at log-spaced frequencies.
    This is identical to the transformer positional encoding repurposed for time.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # (B, dim)


class Block1D(nn.Module):
    """
    Base 1D Conv block: Conv1d → GroupNorm → SiLU.
    GroupNorm over channels is stable regardless of batch size (critical for DDP).
    groups=8 divides evenly into all channel sizes used in dim_mults=(1,2,4).
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ResnetBlock1D(nn.Module):
    """
    Residual block conditioned on the combined context embedding
    (time + label + causal weights → fused via time_mlp).

    Input:  x        (B, C_in, L)
            cond_emb (B, time_emb_dim)   ← fused context from UNet1D
    Output: (B, C_out, L)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = Block1D(in_channels, out_channels)
        self.block2 = Block1D(out_channels, out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond_emb):
        h = self.block1(x)
        # Broadcast conditioning: (B, C_out) → (B, C_out, 1) → adds to every time-step
        h = h + self.time_mlp(cond_emb).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    """
    Pipeline role: GENERATIVE PRIOR (Noise Predictor)

    Predicts the noise epsilon_theta(z_t, t, y, w) added to the latent sequence z_t.
    Conditioned on:
      - t  : diffusion timestep   via SinusoidalPositionEmbeddings
      - y  : emotion class label  via nn.Embedding (num_classes + 1 for CFG null token)
      - w  : causal influence vec (B, 3) via a small MLP

    All three conditioning signals are summed into a single `cond_emb` vector
    and injected into every ResnetBlock1D via the time_mlp pathway.

    Input:  x             (B, latent_dim, L)  — noisy latent sequence z_t
            time          (B,)                — diffusion timestep integer
            label         (B,)  long          — emotion label (0..num_classes-1) or num_classes for CFG null
            causal_weights (B, 3) float       — per-modality causal influence from CausalAttentionGraph

    Output: (B, latent_dim, L)  — predicted noise epsilon_hat

    Architecture:
        init_conv → [Down: ResBlock, ResBlock, Downsample] × 2
                  → [Mid: ResBlock, ResBlock]
                  → [Up: ResBlock, ResBlock, Upsample] × 2
                  → final_res_block → final_conv
    """
    def __init__(
        self,
        latent_dim=256,
        num_classes=6,          # MOSEI 6-way; +1 null token for CFG
        dim_mults=(1, 2, 4),
        resnet_block_groups=8
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Input projection ──────────────────────────────────────────────
        init_dim = latent_dim // 2      # 128
        self.init_conv = nn.Conv1d(latent_dim, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *[init_dim * m for m in dim_mults]]   # [128, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ── Conditioning: time + label + causal ───────────────────────────
        time_dim = init_dim * 4         # 512

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),   # (B,) → (B, 128)
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Label embedding: num_classes+1 to reserve index num_classes as the CFG null token
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

        # Causal weights MLP: (B, 3) → (B, time_dim)
        self.causal_mlp = nn.Sequential(
            nn.Linear(3, time_dim // 2),
            nn.SiLU(),
            nn.Linear(time_dim // 2, time_dim),
        )

        # ── U-Net Downsample ──────────────────────────────────────────────
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock1D(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock1D(dim_in, dim_in, time_emb_dim=time_dim),
                # Stride-2 conv to halve sequence length; last stage keeps length unchanged
                nn.Conv1d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
                if not is_last else
                nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1)
            ]))

        # ── U-Net Bottleneck ──────────────────────────────────────────────
        mid_dim = dims[-1]  # 512
        self.mid_block1 = ResnetBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim)

        # ── U-Net Upsample ────────────────────────────────────────────────
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                # Skip connection doubles the channel count on input
                ResnetBlock1D(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock1D(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                nn.ConvTranspose1d(dim_out, dim_in, kernel_size=4, stride=2, padding=1)
                if not is_last else
                nn.Conv1d(dim_out, dim_in, kernel_size=3, padding=1)
            ]))

        # ── Output ────────────────────────────────────────────────────────
        self.final_res_block = ResnetBlock1D(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(init_dim, latent_dim, kernel_size=1)

    def forward(self, x, time, label=None, causal_weights=None):
        """
        x              : (B, latent_dim, L)  — noisy latent z_t
        time           : (B,)  long          — timestep index
        label          : (B,)  long          — class label (or null token for CFG uncond pass)
        causal_weights : (B, 3) float        — modality influence weights from CausalAttentionGraph

        Returns        : (B, latent_dim, L)  — predicted noise epsilon_hat
        """
        # ── Build fused conditioning vector ───────────────────────────────
        cond = self.time_mlp(time)   # (B, time_dim)

        if label is not None:
            cond = cond + self.label_emb(label)   # (B, time_dim)

        if causal_weights is not None:
            cond = cond + self.causal_mlp(causal_weights.float())  # (B, time_dim)

        # ── Forward pass ──────────────────────────────────────────────────
        x = self.init_conv(x)   # (B, init_dim, L)
        residual_init = x       # saved for final skip connection

        h = []  # skip-connection stack

        for block1, block2, downsample in self.downs:
            x = block1(x, cond)
            h.append(x)
            x = block2(x, cond)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for block1, block2, upsample in self.ups:
            skip = h.pop()
            x = self._match_and_cat(x, skip)
            x = block1(x, cond)
            skip = h.pop()
            x = self._match_and_cat(x, skip)
            x = block2(x, cond)
            x = upsample(x)

        # Final skip from init_conv output — ensures full-resolution gradient path
        x = self._match_and_cat(x, residual_init)
        x = self.final_res_block(x, cond)
        return self.final_conv(x)   # (B, latent_dim, L)

    @staticmethod
    def _match_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate x and skip along channel dim, handling length mismatches.

        If x and skip have different sequence lengths (due to stride-2
        downsampling rounding), pad the shorter one with zeros.
        """
        diff = skip.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            skip = F.pad(skip, (0, -diff))
        return torch.cat((x, skip), dim=1)
