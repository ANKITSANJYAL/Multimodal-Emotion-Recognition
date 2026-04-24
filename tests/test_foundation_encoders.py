"""Tests for foundation model encoder wrappers.

Tests the shape correctness and fallback behavior of RoBERTa,
HuBERT, and CLIP ViT encoder wrappers using pre-extracted features
(without requiring the actual pre-trained models).
"""

import pytest
import torch

from models.encoders.roberta_encoder import RoBERTaEncoder
from models.encoders.hubert_encoder import HuBERTEncoder
from models.encoders.clip_vision_encoder import CLIPVisionEncoder


class TestFoundationEncoders:
    """Tests for foundation model encoder fallback modes."""

    def test_roberta_fallback_mode(self):
        """RoBERTa encoder works in fallback mode with GloVe-like embeddings."""
        encoder = RoBERTaEncoder(
            hidden_dim=512,
            fallback_input_dim=300,
        )
        x = torch.randn(4, 50, 300)
        out = encoder(x)
        assert out.shape == (4, 50, 512), f"RoBERTa fallback: {out.shape}"

    def test_hubert_fallback_mode(self):
        """HuBERT encoder works in fallback mode with COVAREP-like features."""
        encoder = HuBERTEncoder(
            hidden_dim=512,
            fallback_input_dim=74,
        )
        x = torch.randn(4, 50, 74)
        out = encoder(x)
        assert out.shape == (4, 50, 512), f"HuBERT fallback: {out.shape}"

    def test_clip_fallback_mode(self):
        """CLIP ViT encoder works in fallback mode with FAU-like features."""
        encoder = CLIPVisionEncoder(
            hidden_dim=512,
            fallback_input_dim=35,
        )
        x = torch.randn(4, 50, 35)
        out = encoder(x)
        assert out.shape == (4, 50, 512), f"CLIP fallback: {out.shape}"

    def test_roberta_gradient_flow(self):
        """Gradients flow through the RoBERTa projection head."""
        encoder = RoBERTaEncoder(hidden_dim=512, fallback_input_dim=300)
        x = torch.randn(2, 10, 300)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check projection layers have gradients
        for name, param in encoder.projection.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_hubert_gradient_flow(self):
        """Gradients flow through the HuBERT fallback path."""
        encoder = HuBERTEncoder(hidden_dim=512, fallback_input_dim=74)
        x = torch.randn(2, 10, 74)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.fallback_conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
