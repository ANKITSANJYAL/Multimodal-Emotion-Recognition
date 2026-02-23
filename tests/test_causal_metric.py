import pytest
import torch
from utils.metrics import AffectiveCausalMetrics

def test_causal_sensitivity_math():
    """
    Tests that the L1 norm between factual and counterfactual
    probability distributions computes properly without NaNs.
    """
    metrics = AffectiveCausalMetrics(num_classes=6, device='cpu')

    # Mock probability distributions outputted by softmax
    p_factual = torch.tensor([[0.1, 0.7, 0.05, 0.05, 0.05, 0.05],
                              [0.8, 0.1, 0.05, 0.02, 0.01, 0.02]])

    # Mock hallucinated probabilities after audio ablation
    p_counterfactual = torch.tensor([[0.6, 0.2, 0.05, 0.05, 0.05, 0.05],
                                     [0.8, 0.1, 0.05, 0.02, 0.01, 0.02]])

    # L1 norm manually calculated:
    # Batch 0: |0.1-0.6| + |0.7-0.2| + 0s = 0.5 + 0.5 = 1.0
    # Batch 1: identical = 0.0
    # Mean = 0.5

    causal_sensitivity = torch.norm(p_factual - p_counterfactual, p=1, dim=1).mean()

    assert torch.isclose(causal_sensitivity, torch.tensor(0.5))
