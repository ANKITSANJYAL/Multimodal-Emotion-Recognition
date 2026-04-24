import torch
import torch.nn as nn
import math
from tqdm import tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Returns betas β_t ∈ (0,1) for t=1..T.

    alpha_bar_t = cos^2( (t/T + s) / (1+s) * π/2 ) / cos^2( s/(1+s) * π/2 )
    β_t = 1 - alpha_bar_t / alpha_bar_{t-1}

    Clipped to [0.0001, 0.9999] for numerical safety.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    """
    Gathers schedule coefficients at timestep t and reshapes for broadcasting.

    a       : (T,)       — precomputed schedule buffer
    t       : (B,) long  — per-sample timestep index
    x_shape : tuple      — shape of the target tensor (B, C, L)

    Returns : (B, 1, 1)  — broadcast-ready scalar per sample
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class AffectiveDiffusion(nn.Module):
    """
    Pipeline role: DDPM TRAINING LOSS + REVERSE SAMPLING

    Wraps the UNet1D with the full DDPM forward/reverse process.

    Forward process  q(z_t | z_{t-1}): adds Gaussian noise over T steps.
    Reverse process  p_θ(z_{t-1} | z_t, y, w): UNet predicts and removes noise.

    Training:  p_losses()      — returns L_simple = E[||ε - ε_θ(z_t,t,y,w)||²]
    Sampling:  p_sample_loop() — runs full T-step denoising from z_T ~ N(0,I)
                                  with Classifier-Free Guidance (CFG)
    """
    def __init__(self, unet_model, timesteps=1000):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps

        # Pre-compute all schedule constants once; register as buffers so
        # they move to GPU automatically with .to(device)
        betas = cosine_beta_schedule(timesteps)              # (T,)
        alphas = 1.0 - betas                                 # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)        # ᾱ_t  (T,)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]       # ᾱ_{t-1} (T,)
        )

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Forward process constants
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))           # √ᾱ_t
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))    # √(1-ᾱ_t)

        # Reverse process constants
        # sqrt(1/alpha_t)  — used to compute the posterior mean in p_sample
        # NOTE: this is 1/sqrt(alpha_t), NOT 1/sqrt(1-beta_t); those are equal
        #       but the correct derivation is: mu_theta = (1/sqrt(alpha_t)) * (z_t - beta_t/sqrt(1-ᾱ_t) * eps_theta)
        self.register_buffer('sqrt_recip_alphas',
                             torch.sqrt(1.0 / alphas))             # 1/√α_t

        # Posterior variance  q(z_{t-1}|z_t, z_0)
        # β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    # ──────────────────────────────────────────────────────────────────────
    # FORWARD PROCESS
    # ──────────────────────────────────────────────────────────────────────

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: directly sample z_t given z_0 and timestep t.
        q(z_t | z_0) = N(z_t; √ᾱ_t * z_0, (1-ᾱ_t)*I)

        Input:  x_start (B, d_z, L)  — clean latent z_0
                t       (B,)  long   — timestep
        Output: (B, d_z, L)          — noisy latent z_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    # ──────────────────────────────────────────────────────────────────────
    # TRAINING LOSS
    # ──────────────────────────────────────────────────────────────────────

    def p_losses(self, x_start, t, label=None, causal_weights=None, noise=None):
        """
        Computes L_simple = E[||ε - ε_θ(z_t, t, y, w)||²]

        This is the simplified DDPM objective (Ho et al. 2020 Eq. 14), which is
        a weighted ELBO on -log p_θ(z_0).

        Input:  x_start       (B, d_z, L)  — clean latent z_0 from LatentBottleneck
                t             (B,)  long   — random timestep sampled in AffectDiffModule
                label         (B,)  long   — emotion label (with CFG null dropout applied)
                causal_weights (B, 3) float — modality influence from CausalAttentionGraph
        Output: scalar MSE loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # z_t

        # UNet predicts the noise ε that was added
        predicted_noise = self.unet(
            x_noisy, t,
            label=label,
            causal_weights=causal_weights
        )

        return torch.nn.functional.mse_loss(noise.float(), predicted_noise.float())

    # ──────────────────────────────────────────────────────────────────────
    # REVERSE PROCESS (SAMPLING)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, x, t, t_index, label=None, causal_weights=None, cfg_scale=1.0):
        """
        Single reverse step: samples z_{t-1} given z_t.

        DDPM posterior mean:
          μ_θ(z_t, t) = (1/√α_t) * (z_t - β_t/√(1-ᾱ_t) * ε_θ(z_t, t, y, w))

        Classifier-Free Guidance (Ho & Salimans 2022):
          ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
          where s = cfg_scale > 1 sharpens the conditional distribution.

        Input:  x             (B, d_z, L)  — current noisy latent z_t
                t             (B,)  long   — current timestep tensor
                t_index       int          — scalar loop counter (0 = final step)
                label         (B,)  long
                causal_weights (B, 3) float
                cfg_scale     float        — guidance strength (1.0 = no guidance)
        Output: (B, d_z, L)               — denoised z_{t-1}
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Predict noise — optionally with CFG
        if cfg_scale > 1.0 and label is not None:
            # null token index = num_classes (the +1 slot in label_emb)
            null_label = torch.full_like(label, fill_value=self.unet.label_emb.num_embeddings - 1)
            cond_noise   = self.unet(x, t, label=label,      causal_weights=causal_weights)
            uncond_noise = self.unet(x, t, label=null_label,  causal_weights=causal_weights)
            predicted_noise = uncond_noise + cfg_scale * (cond_noise - uncond_noise)
        else:
            predicted_noise = self.unet(x, t, label=label, causal_weights=causal_weights)

        # DDPM posterior mean  (Ho et al. 2020 Algorithm 2)
        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t
        )
        model_mean = torch.clamp(model_mean, min=-12.0, max=12.0)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device, label=None, causal_weights=None, cfg_scale=1.0):
        """
        Full reverse denoising loop: z_T ~ N(0,I) → z_0.
        Used for counterfactual hallucination and verification checks.

        Input:  shape          tuple (B, d_z, L)
                device         torch.device
                label          (B,)  long   — target emotion class
                causal_weights (B, 3) float
                cfg_scale      float
        Output: (B, d_z, L)   — hallucinated clean latent z_0
        """
        b = shape[0]
        z = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Denoising', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t, i,
                              label=label,
                              causal_weights=causal_weights,
                              cfg_scale=cfg_scale)
        return z

    # ──────────────────────────────────────────────────────────────────────
    # DDIM SAMPLING  (Song et al. 2020, "Denoising Diffusion Implicit Models")
    # ──────────────────────────────────────────────────────────────────────

    def _predict_x0_from_noise(self, x_t, t, predicted_noise):
        """Recover x_0 estimate from the noise prediction at timestep t.

        x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        """
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t.clamp(min=1e-8)

    def _get_cfg_noise(self, x, t, label, causal_weights, cfg_scale):
        """Compute noise prediction with optional Classifier-Free Guidance."""
        if cfg_scale > 1.0 and label is not None:
            null_label = torch.full_like(label, fill_value=self.unet.label_emb.num_embeddings - 1)
            cond_noise = self.unet(x, t, label=label, causal_weights=causal_weights)
            uncond_noise = self.unet(x, t, label=null_label, causal_weights=causal_weights)
            return uncond_noise + cfg_scale * (cond_noise - uncond_noise)
        return self.unet(x, t, label=label, causal_weights=causal_weights)

    @torch.no_grad()
    def ddim_sample_step(self, x_t, t_curr, t_prev, label=None,
                         causal_weights=None, cfg_scale=1.0, eta=0.0):
        """Single DDIM reverse step: z_{t_curr} → z_{t_prev}.

        DDIM update rule (Song et al. 2020 Eq. 12):
          x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ²) · ε_θ + σ · ε

        where σ = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})
        and η=0 gives fully deterministic sampling.

        Args:
            x_t: Current noisy latent (B, d_z, L).
            t_curr: Current timestep indices (B,) long.
            t_prev: Previous timestep indices (B,) long.
            label: Class labels (B,) long.
            causal_weights: Modality influence (B, 3).
            cfg_scale: CFG guidance strength.
            eta: Stochasticity parameter (0=deterministic DDIM, 1=DDPM).

        Returns:
            Denoised latent at t_prev (B, d_z, L).
        """
        # Predict noise
        predicted_noise = self._get_cfg_noise(x_t, t_curr, label, causal_weights, cfg_scale)

        # Recover x_0 estimate
        x0_pred = self._predict_x0_from_noise(x_t, t_curr, predicted_noise)
        x0_pred = torch.clamp(x0_pred, -12.0, 12.0)

        # Get schedule values at t_curr and t_prev
        alpha_bar_t = extract(self.alphas_cumprod, t_curr, x_t.shape)
        alpha_bar_t_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)

        # Compute sigma for stochasticity control
        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8)
            * (1.0 - alpha_bar_t / alpha_bar_t_prev.clamp(min=1e-8))
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt((1.0 - alpha_bar_t_prev - sigma ** 2).clamp(min=0.0)) * predicted_noise

        # DDIM update
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, shape, device, label=None, causal_weights=None,
                         cfg_scale=1.0, num_inference_steps=50, eta=0.0):
        """Full DDIM reverse loop with sub-sampled timestep schedule.

        Generates z_0 from z_T ~ N(0,I) in `num_inference_steps` steps
        instead of the full T=1000 DDPM schedule, giving ~20× speedup.

        Args:
            shape: Output shape (B, d_z, L).
            device: Target device.
            label: Class labels (B,) long.
            causal_weights: Modality influence (B, 3).
            cfg_scale: CFG guidance strength.
            num_inference_steps: Number of DDIM steps (default 50).
            eta: Stochasticity (0=deterministic).

        Returns:
            Clean latent z_0 (B, d_z, L).
        """
        b = shape[0]
        z = torch.randn(shape, device=device)

        # Build sub-sampled timestep schedule: evenly spaced from T-1 to 0
        step_size = self.timesteps // num_inference_steps
        timesteps_seq = list(range(0, self.timesteps, step_size))
        timesteps_seq = list(reversed(timesteps_seq))  # [T-step, ..., step, 0]

        for i in tqdm(range(len(timesteps_seq) - 1), desc='DDIM Sampling'):
            t_curr = torch.full((b,), timesteps_seq[i], device=device, dtype=torch.long)
            t_prev = torch.full((b,), timesteps_seq[i + 1], device=device, dtype=torch.long)
            z = self.ddim_sample_step(
                z, t_curr, t_prev,
                label=label,
                causal_weights=causal_weights,
                cfg_scale=cfg_scale,
                eta=eta,
            )

        # Final step to t=0
        t_final = torch.full((b,), timesteps_seq[-1], device=device, dtype=torch.long)
        t_zero = torch.zeros(b, device=device, dtype=torch.long)
        if timesteps_seq[-1] > 0:
            z = self.ddim_sample_step(
                z, t_final, t_zero,
                label=label,
                causal_weights=causal_weights,
                cfg_scale=cfg_scale,
                eta=0.0,  # final step always deterministic
            )

        return z
