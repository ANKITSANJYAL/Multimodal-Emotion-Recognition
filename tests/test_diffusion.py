import pytest
import torch
from models.diffusion.unet_1d import UNet1D
from models.diffusion.diffusion_utils import AffectiveDiffusion

@pytest.fixture
def diffusion_setup():
    latent_dim = 256
    seq_len = 50
    batch_size = 4
    unet = UNet1D(latent_dim=latent_dim)
    diffusion = AffectiveDiffusion(unet, timesteps=100)
    return diffusion, batch_size, latent_dim, seq_len

def test_forward_diffusion_noise_injection(diffusion_setup):
    """Tests the q_sample forward process equation."""
    diffusion, b, c, t = diffusion_setup
    x_start = torch.randn(b, c, t) # (Batch, Channels, Time)

    # Pick a random timestep
    timestep = torch.tensor([50, 20, 10, 80])

    x_noisy = diffusion.q_sample(x_start, timestep)

    # The shape must remain strictly identical
    assert x_noisy.shape == (b, c, t)
    # The noisy tensor should mathematically diverge from the start tensor
    assert not torch.allclose(x_start, x_noisy)

def test_reverse_diffusion_loop(diffusion_setup):
    """Tests that the UNet can successfully denoise from pure noise to a valid shape."""
    diffusion, b, c, t = diffusion_setup

    # Use CPU for testing to avoid CUDA initialization overhead in CI/CD
    device = torch.device('cpu')

    shape = (b, c, t)
    # This will run the tqdm loop. For a test, we keep timesteps low (100).
    x_hallucinated = diffusion.p_sample_loop(shape, device)

    assert x_hallucinated.shape == (b, c, t)
    assert not torch.isnan(x_hallucinated).any(), "NaNs detected in reverse diffusion!"
