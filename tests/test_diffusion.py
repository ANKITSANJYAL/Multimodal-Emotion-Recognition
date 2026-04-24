"""Tests for DDPM/DDIM diffusion forward and reverse processes."""

import pytest
import torch

from models.diffusion.unet_1d import UNet1D
from models.diffusion.diffusion_utils import AffectiveDiffusion


@pytest.fixture
def diffusion_setup():
    """Create a diffusion model with small timestep count for fast testing."""
    latent_dim = 256
    seq_len = 50
    batch_size = 4
    unet = UNet1D(latent_dim=latent_dim)
    diffusion = AffectiveDiffusion(unet, timesteps=100)
    return diffusion, batch_size, latent_dim, seq_len


class TestDiffusion:
    """Tests for the diffusion forward and reverse processes."""

    def test_forward_noise_injection(self, diffusion_setup):
        """q_sample adds noise and preserves shape."""
        diffusion, b, c, l = diffusion_setup
        x_start = torch.randn(b, c, l)
        timestep = torch.tensor([50, 20, 10, 80])

        x_noisy = diffusion.q_sample(x_start, timestep)
        assert x_noisy.shape == (b, c, l)
        assert not torch.allclose(x_start, x_noisy)

    def test_training_loss(self, diffusion_setup):
        """p_losses returns a finite positive scalar."""
        diffusion, b, c, l = diffusion_setup
        x_start = torch.randn(b, c, l)
        t = torch.randint(0, 100, (b,))
        labels = torch.randint(0, 6, (b,))
        cw = torch.ones(b, 3)

        loss = diffusion.p_losses(x_start, t, label=labels, causal_weights=cw)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss > 0

    def test_ddpm_sampling(self, diffusion_setup):
        """DDPM reverse loop produces valid output shape without NaNs."""
        diffusion, b, c, l = diffusion_setup
        device = torch.device("cpu")

        shape = (b, c, l)
        z = diffusion.p_sample_loop(shape, device)
        assert z.shape == (b, c, l)
        assert not torch.isnan(z).any(), "NaNs in DDPM sampling"

    def test_ddim_sampling(self, diffusion_setup):
        """DDIM reverse loop produces valid output shape without NaNs."""
        diffusion, b, c, l = diffusion_setup
        device = torch.device("cpu")

        shape = (b, c, l)
        z = diffusion.ddim_sample_loop(
            shape, device, num_inference_steps=10, eta=0.0
        )
        assert z.shape == (b, c, l)
        assert not torch.isnan(z).any(), "NaNs in DDIM sampling"

    def test_ddim_with_cfg(self, diffusion_setup):
        """DDIM with Classifier-Free Guidance produces valid output."""
        diffusion, b, c, l = diffusion_setup
        device = torch.device("cpu")
        labels = torch.arange(min(b, 6), device=device)
        cw = torch.ones(labels.shape[0], 3, device=device)

        shape = (labels.shape[0], c, l)
        z = diffusion.ddim_sample_loop(
            shape, device, label=labels, causal_weights=cw,
            cfg_scale=3.0, num_inference_steps=10,
        )
        assert z.shape == shape
        assert not torch.isnan(z).any(), "NaNs in DDIM+CFG sampling"

    def test_noise_schedule_properties(self, diffusion_setup):
        """Cosine schedule has monotonically decreasing alphas_cumprod."""
        diffusion, _, _, _ = diffusion_setup
        ac = diffusion.alphas_cumprod
        # alphas_cumprod should be monotonically decreasing
        assert (ac[1:] <= ac[:-1]).all(), "alphas_cumprod not monotonically decreasing"
        # Should start near 1 and end near 0
        assert ac[0] > 0.99, f"First alpha_cumprod too low: {ac[0]}"
        assert ac[-1] < 0.05, f"Last alpha_cumprod too high: {ac[-1]}"
