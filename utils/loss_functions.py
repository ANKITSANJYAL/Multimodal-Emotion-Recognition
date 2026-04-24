"""Loss functions for latent disentanglement.

Implements the Beta-TC-VAE Total Correlation penalty for rigorous
latent disentanglement using minibatch stratified sampling.

Reference: Chen et al. (2018) "Isolating Sources of Disentanglement
in Variational Autoencoders" (NeurIPS).
"""

import math
from typing import Tuple

import torch


class BetaTCVAELoss:
    """Total Correlation penalty for β-TC-VAE disentanglement.

    Decomposes the KL divergence into mutual information, total correlation,
    and dimension-wise KL, allowing independent weighting of each term.
    """

    @staticmethod
    def compute_tc_penalty(
        z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
        chunk_size: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Total Correlation penalty via minibatch stratified sampling.

        Uses chunked computation to avoid OOM on large batches — the naive
        (B, B, D) pairwise expansion can exhaust GPU memory.

        Args:
            z: Sampled latents (B, D).
            mu: Posterior mean (B, D).
            logvar: Posterior log-variance (B, D).
            chunk_size: Process pairwise distances in chunks of this size.

        Returns:
            Tuple of (tc_loss, kl_loss) scalars.
        """
        # Cast to float32 for numerical stability under fp16-mixed
        z = z.float()
        mu = mu.float()
        logvar = logvar.float()

        batch_size, latent_dim = z.shape

        # log q(z|x): Gaussian log-density under the posterior
        log_q_z_given_x = -0.5 * (
            math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()
        )

        # Chunked pairwise log-density computation to avoid (B, B, D) OOM
        # For each chunk of z rows, compute pairwise log q against all mu/logvar
        # Pre-compute logvar.exp() once to avoid repeated computation
        logvar_exp_inv = (-logvar).exp()  # (B, D) — 1/sigma^2

        pairwise_joint = []  # will hold (chunk,) logsumexp results for joint
        pairwise_marginal = []  # will hold (chunk, D) logsumexp results for marginals

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            z_chunk = z[start:end].unsqueeze(1)  # (C, 1, D)

            # Pairwise log q(z_i | x_j) for this chunk
            diff = z_chunk - mu.unsqueeze(0)  # (C, B, D)
            log_q = -0.5 * (
                math.log(2 * math.pi)
                + logvar.unsqueeze(0)  # (1, B, D)
                + diff.pow(2) * logvar_exp_inv.unsqueeze(0)
            )  # (C, B, D)
            del diff

            # Joint: sum over D first, then logsumexp over B
            pairwise_joint.append(
                torch.logsumexp(log_q.sum(dim=2), dim=1)  # (C,)
            )

            # Marginals: logsumexp over B for each dimension
            pairwise_marginal.append(
                torch.logsumexp(log_q, dim=1)  # (C, D)
            )
            del log_q

        # Concatenate chunks back to (B,) and (B, D)
        log_q_z_joint = torch.cat(pairwise_joint, dim=0)  # (B,)
        log_q_z_marginals = torch.cat(pairwise_marginal, dim=0)  # (B, D)

        # log q(z): joint density estimate
        log_q_z = log_q_z_joint - math.log(batch_size * batch_size)

        # log ∏ q(z_j): sum of marginal log-densities
        log_prod_q_z_j = (log_q_z_marginals - math.log(batch_size)).sum(dim=1)

        # TC = E_q(z|x)[log q(z) - log ∏ q(z_j)]
        tc_loss = (log_q_z - log_prod_q_z_j).mean()

        # Standard KL divergence for the prior term
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        ).mean()

        return tc_loss, kl_loss
