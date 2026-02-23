import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Injects the diffusion timestep 't' into the network so it knows
    how much noise is currently in the latent space.
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
        return embeddings


class Block1D(nn.Module):
    """Base 1D Convolutional block with GroupNorm and SiLU."""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ResnetBlock1D(nn.Module):
    """
    Residual block that conditions the 1D latent sequence on the
    current diffusion timestep embedding.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = Block1D(in_channels, out_channels)
        self.block2 = Block1D(out_channels, out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        # Add time embedding (reshaped for 1D broadcasting: B, C, 1)
        time_emb = self.time_mlp(time_emb).unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    """
    Core Generative Network for Affect-Diff.
    Predicts the noise added to the multimodal latent space.
    """
    def __init__(
        self,
        latent_dim=256,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8
    ):
        super().__init__()

        # Dimensions
        init_dim = latent_dim // 2
        self.init_conv = nn.Conv1d(latent_dim, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # U-Net Downsample
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock1D(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock1D(dim_in, dim_in, time_emb_dim=time_dim),
                nn.Conv1d(dim_in, dim_out, kernel_size=4, stride=2, padding=1) if not is_last else nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1)
            ]))

        # U-Net Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim)

        # U-Net Upsample
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock1D(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock1D(dim_out, dim_out, time_emb_dim=time_dim),
                nn.ConvTranspose1d(dim_out, dim_in, kernel_size=4, stride=2, padding=1) if not is_last else nn.Conv1d(dim_out, dim_in, kernel_size=3, padding=1)
            ]))

        self.final_res_block = ResnetBlock1D(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(init_dim, latent_dim, kernel_size=1)

    def forward(self, x, time):
        """
        x: (B, latent_dim, T) - The noisy multimodal latent sequence
        time: (B,) - The diffusion timestep
        Returns: (B, latent_dim, T) - The predicted noise $\epsilon$
        """
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []

        # Downsample
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # Skip connection
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1) # Skip connection
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, h.pop()), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)
