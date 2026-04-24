"""Shared pytest fixtures for the Affect-Diff test suite.

Provides reusable test fixtures for model instantiation and dummy
data generation. No sys.path hacks — use pytest from project root.
"""

import pytest
import torch
from unittest.mock import MagicMock

from modules.affect_diff_module import AffectDiffModule


@pytest.fixture
def model_config():
    """Default model configuration for testing (matches AffectDiffModule.__init__ signature)."""
    return {
        "text_dim": 300,
        "audio_dim": 74,
        "video_dim": 35,
        "hidden_dim": 512,
        "latent_dim": 256,
        "num_classes": 6,
        "encoder_type": "legacy",
        "text_backbone": "roberta-base",
        "audio_backbone": "facebook/hubert-base-ls960",
        "video_backbone": "openai/clip-vit-base-patch16",
        "freeze_backbones": True,
        "fusion_type": "concat",
        "num_bottleneck_tokens": 50,
        "num_cross_attn_layers": 2,
        "num_self_attn_layers": 2,
        "dag_method": "notears",
        "diffusion_steps": 10,       # Small for fast testing
        "ddim_steps": 5,
        "beta_kl": 5.0,
        "lambda_diff": 1.0,
        "lambda_causal": 0.1,
        "lambda_recon": 0.5,
        "cfg_scale": 3.0,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "ema_decay": 0.999,
        "use_reconstruction": True,
        "use_diffusion": True,
        "use_causal_graph": True,
        "use_augmentation": False,    # Disable for deterministic tests
        "use_beta_tc_vae": False,
        "label_smoothing": 0.1,
        "free_bits": 2.0,
    }


@pytest.fixture
def model(model_config):
    """Instantiated AffectDiffModule for testing."""
    m = AffectDiffModule(**model_config)
    m.eval()
    # Attach a mock Trainer so PL properties (current_epoch, global_rank, etc.) work
    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.logger = None
    trainer.log_dir = "/tmp"
    trainer.loggers = []
    trainer.callback_metrics = {}
    trainer.is_global_zero = True
    trainer.world_size = 1
    trainer.global_rank = 0
    trainer.local_rank = 0
    trainer.node_rank = 0
    m._trainer = trainer
    return m


@pytest.fixture
def dummy_batch():
    """Dummy CMU-MOSEI batch with correct dimensions."""
    B, L = 4, 50
    return {
        "text": torch.randn(B, L, 300),
        "audio": torch.randn(B, L, 74),
        "vision": torch.randn(B, L, 35),
        "labels": torch.randint(0, 6, (B,)),
    }


@pytest.fixture
def small_batch():
    """Small batch for edge case testing."""
    B, L = 2, 50
    return {
        "text": torch.randn(B, L, 300),
        "audio": torch.randn(B, L, 74),
        "vision": torch.randn(B, L, 35),
        "labels": torch.randint(0, 6, (B,)),
    }
