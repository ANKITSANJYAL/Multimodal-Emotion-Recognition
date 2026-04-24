"""Visualization utilities for affective analysis with W&B logging.

Generates plots for counterfactual intervention analysis and causal
graph influence tracking.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


class AffectiveVisualizer:
    """Visualization tools for multimodal emotion analysis.

    Generates publication-quality plots for W&B logging.

    Args:
        class_names: List of emotion class names.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.class_names = class_names or [
            "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"
        ]

    def plot_counterfactual_shift(
        self,
        p_factual: torch.Tensor,
        p_counterfactual: torch.Tensor,
        modality_ablated: str = "Audio",
    ) -> wandb.Image:
        """Create a bar chart comparing factual vs counterfactual predictions.

        Args:
            p_factual: Factual probability distribution (B, K).
            p_counterfactual: Counterfactual distribution (B, K).
            modality_ablated: Name of the ablated modality.

        Returns:
            W&B Image object for logging.
        """
        p_fac = p_factual.cpu().numpy()[0]
        p_cf = p_counterfactual.cpu().numpy()[0]

        x = np.arange(len(self.class_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, p_fac, width, label="Factual", color="#1f77b4")
        ax.bar(x + width / 2, p_cf, width, label=f"CF (No {modality_ablated})", color="#ff7f0e")

        ax.set_ylabel("Probability")
        ax.set_title(f"Affective Shift: {modality_ablated} Ablation + Generative Healing")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()

        wandb_img = wandb.Image(fig)
        plt.close(fig)
        return wandb_img

    def plot_causal_graph_weights(
        self, weights_dict: Dict[str, float], step: int
    ) -> None:
        """Log causal graph influence weights to W&B.

        Args:
            weights_dict: Dictionary with Text/Audio/Video influence scores.
            step: Global training step.
        """
        wandb.log(
            {
                "Causal_Graph/Text_Influence": weights_dict["Text_Influence"],
                "Causal_Graph/Audio_Influence": weights_dict["Audio_Influence"],
                "Causal_Graph/Video_Influence": weights_dict["Video_Influence"],
                "global_step": step,
            }
        )
