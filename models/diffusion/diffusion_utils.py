import torch
import torch.nn as nn
import math
from tqdm import tqdm

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
    This prevents the latent sequence from being destroyed too quickly compared to a linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """
    Extracts the appropriate scaling factor for the current timestep t
    and reshapes it to broadcast across the batch and sequence dimensions.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class AffectiveDiffusion(nn.Module):
    """
    The Core DDPM logic wrapper for our Affect-Diff 1D U-Net.
    Manages both q(x_t | x_{t-1}) [Forward] and p(x_{t-1} | x_t) [Reverse].
    """
    def __init__(self, unet_model, timesteps=1000, objective='pred_noise'):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps
        self.objective = objective  # We train to predict the noise $\epsilon$

        # Pre-compute the cosine schedule variance parameters $\beta_t$
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.]), alphas_cumprod[:-1]])

        # Register buffers so PyTorch handles device placement (CPU/GPU) automatically
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for Forward Process: q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Calculations for Reverse Process: p_\theta(x_{t-1} | x_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        """
        The Forward Process.
        Equation: $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$
        Directly jumps to timestep 't' without iterating.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """
        Calculates the training loss for a specific timestep.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 1. Create noisy latent $x_t$
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 2. Predict noise using our 1D U-Net
        predicted_noise = self.unet(x_noisy, t)

        # 3. L2 Loss between actual noise and predicted noise
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        A single step of the Reverse Denoising Process.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / (1. - self.betas)), t, x.shape)

        # U-Net predicts the noise
        predicted_noise = self.unet(x, t)

        # Calculate the mean of the posterior
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Add stochasticity based on learned variance
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """
        The full counterfactual hallucination loop.
        Starts with pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$
        and iterates backward to $x_0$.
        """
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)

        # Iterate from T-1 down to 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Denoising Counterfactual', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)

        return img
