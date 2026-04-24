"""Differentiable causal graph learning over modality nodes.

Supports two DAG learning methods:
  1. Gumbel-Softmax with L1 sparsity (original, simpler)
  2. NOTEARS with exponential trace acyclicity constraint (rigorous)

Reference: Zheng et al. (2018) "DAGs with NO TEARS" (NeurIPS).
"""

import logging
import math
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalAttentionGraph(nn.Module):
    """Learns a differentiable adjacency matrix representing causal influence
    across modalities {Text, Audio, Video}.

    Args:
        latent_dim: Dimension of the input latent representations.
        num_nodes: Number of modality nodes (default 3: T, A, V).
        dag_method: DAG learning method — 'gumbel' or 'notears'.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_nodes: int = 3,
        dag_method: Literal["gumbel", "notears"] = "notears",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.dag_method = dag_method

        # Node embedding projections for attention-based edge scoring
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)

        # Gumbel-Softmax temperature (annealed over training)
        self.register_buffer("temperature", torch.tensor(1.0))

        logger.info(
            "CausalAttentionGraph initialized: method=%s, nodes=%d, dim=%d",
            dag_method, num_nodes, latent_dim,
        )

    def set_temperature(self, temp: float) -> None:
        """Set the Gumbel-Softmax temperature for edge discretization."""
        self.temperature.copy_(torch.tensor(temp))

    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        z_v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the causal adjacency matrix from unimodal latents.

        Args:
            z_t: Text latent (B, L, latent_dim).
            z_a: Audio latent (B, L, latent_dim).
            z_v: Video latent (B, L, latent_dim).

        Returns:
            Adjacency matrix (B, num_nodes, num_nodes) with self-loops masked.
        """
        # Pool temporal dimension: (B, L, D) → (B, D)
        z_t_pooled = z_t.mean(dim=1)
        z_a_pooled = z_a.mean(dim=1)
        z_v_pooled = z_v.mean(dim=1)

        # Stack nodes: (B, N, D)
        nodes = torch.stack([z_t_pooled, z_a_pooled, z_v_pooled], dim=1)

        queries = self.query_proj(nodes)
        keys = self.key_proj(nodes)

        # Scaled dot-product edge scores: (B, N, N)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.latent_dim ** 0.5)
        scores = torch.clamp(scores, min=-20.0, max=20.0)

        if self.dag_method == "gumbel":
            adj_matrix = F.gumbel_softmax(
                scores, tau=self.temperature, hard=False, dim=-1
            )
        else:
            # NOTEARS: use sigmoid to get edge probabilities in [0, 1]
            adj_matrix = torch.sigmoid(scores)

        # Mask out self-loops
        mask = torch.eye(self.num_nodes, device=adj_matrix.device).unsqueeze(0).bool()
        adj_matrix = adj_matrix.masked_fill(mask, 0.0)

        return adj_matrix

    def compute_dag_penalty(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Compute the NOTEARS acyclicity penalty: h(A) = tr(e^{A⊙A}) - d.

        This penalty equals zero iff A is a DAG, and increases with the
        number/strength of cycles in the graph.

        For the Gumbel method, falls back to L1 sparsity penalty.

        Args:
            adj_matrix: Adjacency matrix (B, N, N).

        Returns:
            Scalar penalty (averaged over batch).
        """
        if self.dag_method == "gumbel":
            # L1 sparsity penalty
            return torch.norm(adj_matrix, p=1) / (
                adj_matrix.shape[0] * adj_matrix.shape[1] * adj_matrix.shape[2]
            )

        # NOTEARS: h(A) = tr(e^{A⊙A}) - d
        # Average over batch
        # Cast to float32 for numerical stability (matrix_exp is unstable in fp16)
        adj_f32 = adj_matrix.float()
        A_sq = adj_f32 * adj_f32  # Element-wise square (B, N, N)
        d = self.num_nodes

        # Matrix exponential via eigendecomposition (stable for small matrices)
        # For 3×3, this is efficient
        expm_A = torch.matrix_exp(A_sq)  # (B, N, N)
        trace = torch.diagonal(expm_A, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        h = (trace - d).mean()

        # Combine with L1 sparsity for edge parsimony
        sparsity = adj_f32.abs().mean()

        return h + 0.1 * sparsity

    def get_causal_weights(self, adj_matrix: torch.Tensor) -> Dict[str, float]:
        """Decode the adjacency matrix into normalized modality importance scores.

        Args:
            adj_matrix: Adjacency matrix (B, N, N).

        Returns:
            Dictionary mapping modality names to their normalized importance.
        """
        mean_adj = adj_matrix.detach().mean(dim=0)
        importance = mean_adj.sum(dim=0)
        total = importance.sum() + 1e-8
        normalized = importance / total

        return {
            "Text_Influence": normalized[0].item(),
            "Audio_Influence": normalized[1].item(),
            "Video_Influence": normalized[2].item(),
        }
