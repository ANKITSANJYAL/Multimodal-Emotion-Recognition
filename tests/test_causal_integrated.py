"""Integration tests for the full Affect-Diff pipeline.

Uses conftest fixtures — no sys.path hacks needed.
Run from project root: python -m pytest tests/ -v
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock


def _attach_mock_trainer(model):
    """Attach a minimal mock Trainer so PL properties (current_epoch, etc.) work."""
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
    model._trainer = trainer
    return model


class TestForwardPass:
    """Forward pass and shape correctness."""

    def test_logit_shape(self, model, dummy_batch):
        """Inference forward returns (B, num_classes) logits."""
        logits = model(dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"])
        assert logits.shape == (4, 6), f"Expected (4,6), got {logits.shape}"

    def test_no_nans_in_logits(self, model, dummy_batch):
        """Logits contain no NaN values."""
        logits = model(dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"])
        assert not torch.isnan(logits).any(), "NaN in logits"


class TestBottleneck:
    """LatentBottleneck output shape and properties."""

    def test_output_shapes(self, model, dummy_batch):
        """Bottleneck returns 4 tensors with correct shapes."""
        z_perm, mu, logvar, adj = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        B, L = 4, 50
        assert z_perm.shape == (B, 256, L), f"z_perm: {z_perm.shape}"
        assert mu.shape == (B, L, 256), f"mu: {mu.shape}"
        assert logvar.shape == (B, L, 256), f"logvar: {logvar.shape}"
        assert adj.shape == (B, 3, 3), f"adj: {adj.shape}"

    def test_logvar_clamped(self, model, dummy_batch):
        """logvar is clamped to [-10, 5] for numerical safety."""
        _, _, logvar, _ = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        assert logvar.min() >= -10.0 - 1e-5, f"logvar min: {logvar.min()}"
        assert logvar.max() <= 5.0 + 1e-5, f"logvar max: {logvar.max()}"


class TestUNetConditioning:
    """UNet1D accepts all conditioning signals."""

    def test_unet_shape(self, model, dummy_batch):
        """UNet accepts label and causal_weights conditioning."""
        z_perm, _, _, _ = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        B = 4
        t = torch.randint(0, 10, (B,))
        labels = dummy_batch["labels"]
        cw = torch.ones(B, 3)
        eps_hat = model.unet(z_perm, t, label=labels, causal_weights=cw)
        assert eps_hat.shape == z_perm.shape, f"UNet output: {eps_hat.shape}"


class TestSharedStep:
    """Training/validation step correctness."""

    def test_training_loss_finite(self, model, dummy_batch):
        """shared_step returns a finite non-NaN loss."""
        model.train()
        _attach_mock_trainer(model)
        loss = model.shared_step(dummy_batch, 0, stage="train")
        assert not torch.isnan(loss), "NaN in training loss"
        assert not torch.isinf(loss), "Inf in training loss"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    def test_validation_loss_finite(self, model, dummy_batch):
        """shared_step in val mode returns a finite loss."""
        model.eval()
        _attach_mock_trainer(model)
        loss = model.shared_step(dummy_batch, 0, stage="val")
        assert not torch.isnan(loss), "NaN in validation loss"
        assert not torch.isinf(loss), "Inf in validation loss"


class TestCausalGraph:
    """Causal graph properties."""

    def test_weights_sum_to_one(self, model, dummy_batch):
        """Causal weights are normalized to sum to 1."""
        _, _, _, adj = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        weights = model.bottleneck.causal_graph.get_causal_weights(adj)
        assert "Text_Influence" in weights
        assert "Audio_Influence" in weights
        assert "Video_Influence" in weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights sum to {total}, not 1.0"

    def test_adj_no_self_loops(self, model, dummy_batch):
        """Adjacency matrix has zero diagonal (no self-loops)."""
        _, _, _, adj = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        diag = torch.diagonal(adj, dim1=-2, dim2=-1)
        assert (diag.abs() < 1e-6).all(), f"Self-loops present: {diag}"

    def test_dag_penalty_finite(self, model, dummy_batch):
        """NOTEARS DAG penalty is finite."""
        _, _, _, adj = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        penalty = model.bottleneck.causal_graph.compute_dag_penalty(adj)
        assert not torch.isnan(penalty), "NaN in DAG penalty"
        assert not torch.isinf(penalty), "Inf in DAG penalty"


class TestDiffusionIntegrated:
    """Diffusion integrated with the full model."""

    def test_q_sample_shape(self, model, dummy_batch):
        """Forward diffusion preserves shape and adds noise."""
        z_perm, _, _, _ = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        B = 4
        t = torch.randint(0, 10, (B,))
        z_noisy = model.diffusion.q_sample(z_perm, t)
        assert z_noisy.shape == z_perm.shape
        assert not torch.allclose(z_perm, z_noisy), "q_sample didn't add noise"

    def test_diffusion_loss_finite(self, model, dummy_batch):
        """p_losses returns a finite positive loss."""
        z_perm, _, _, adj = model.bottleneck(
            dummy_batch["text"], dummy_batch["audio"], dummy_batch["vision"]
        )
        B = 4
        t = torch.randint(0, 10, (B,))
        cw = torch.clamp(adj.sum(dim=1), min=0.0, max=5.0)
        loss = model.diffusion.p_losses(
            z_perm, t, label=dummy_batch["labels"], causal_weights=cw
        )
        assert not torch.isnan(loss), "NaN in diffusion loss"
        assert loss > 0, f"Diffusion loss should be positive: {loss}"


class TestGradientFlow:
    """Gradient flow through the full pipeline."""

    def test_gradients_reach_encoders(self, model_config):
        """Gradients propagate from loss back to encoder parameters."""
        from modules.affect_diff_module import AffectDiffModule

        m = AffectDiffModule(**model_config)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        loss = m.shared_step(batch, 0, stage="train")
        loss.backward()

        # Check encoder has gradients
        encoder_has_grad = False
        for name, p in m.bottleneck.text_enc.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                encoder_has_grad = True
                break
        assert encoder_has_grad, "No gradients reached the text encoder"

    def test_gradients_reach_classifier(self, model_config):
        """Gradients propagate to the classifier head."""
        from modules.affect_diff_module import AffectDiffModule

        m = AffectDiffModule(**model_config)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        loss = m.shared_step(batch, 0, stage="train")
        loss.backward()

        classifier_has_grad = False
        for name, p in m.classifier.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                classifier_has_grad = True
                break
        assert classifier_has_grad, "No gradients reached the classifier"

    def test_gradients_reach_unet(self, model_config):
        """Gradients propagate to the UNet."""
        from modules.affect_diff_module import AffectDiffModule

        m = AffectDiffModule(**model_config)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        loss = m.shared_step(batch, 0, stage="train")
        loss.backward()

        unet_has_grad = False
        for name, p in m.unet.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                unet_has_grad = True
                break
        assert unet_has_grad, "No gradients reached the UNet"


class TestAblationToggles:
    """Ablation toggles correctly disable subsystems."""

    def test_no_diffusion(self, model_config):
        """Model works with diffusion disabled."""
        from modules.affect_diff_module import AffectDiffModule

        cfg = {**model_config, "use_diffusion": False}
        m = AffectDiffModule(**cfg)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        assert m.unet is None
        assert m.diffusion is None
        loss = m.shared_step(batch, 0, stage="train")
        assert not torch.isnan(loss)

    def test_no_reconstruction(self, model_config):
        """Model works with reconstruction disabled."""
        from modules.affect_diff_module import AffectDiffModule

        cfg = {**model_config, "use_reconstruction": False}
        m = AffectDiffModule(**cfg)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        assert m.decoder is None
        loss = m.shared_step(batch, 0, stage="train")
        assert not torch.isnan(loss)

    def test_no_causal_graph(self, model_config):
        """Model works with causal graph penalty disabled."""
        from modules.affect_diff_module import AffectDiffModule

        cfg = {**model_config, "use_causal_graph": False}
        m = AffectDiffModule(**cfg)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        loss = m.shared_step(batch, 0, stage="train")
        assert not torch.isnan(loss)

    def test_classifier_only(self, model_config):
        """Model works with everything disabled — just classification."""
        from modules.affect_diff_module import AffectDiffModule

        cfg = {
            **model_config,
            "use_diffusion": False,
            "use_reconstruction": False,
            "use_causal_graph": False,
            "use_augmentation": False,
            "use_beta_tc_vae": False,
        }
        m = AffectDiffModule(**cfg)
        m.train()
        _attach_mock_trainer(m)

        B, L = 2, 50
        batch = {
            "text": torch.randn(B, L, 300),
            "audio": torch.randn(B, L, 74),
            "vision": torch.randn(B, L, 35),
            "labels": torch.randint(0, 6, (B,)),
        }

        loss = m.shared_step(batch, 0, stage="train")
        assert not torch.isnan(loss)
