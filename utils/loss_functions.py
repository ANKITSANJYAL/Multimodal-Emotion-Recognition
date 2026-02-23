import torch
import torch.nn.functional as F
import math

class BetaTCVAELoss:
    """
    Computes the Total Correlation penalty for rigorous latent disentanglement.
    Minibatch Stratified Sampling approach.
    """
    @staticmethod
    def compute_tc_penalty(z, mu, logvar):
        """
        z: Sampled latents (Batch, Latent_Dim)
        mu: Mean (Batch, Latent_Dim)
        logvar: Log Variance (Batch, Latent_Dim)
        """
        batch_size, latent_dim = z.shape

        # Calculate log q(z|x)
        # Assuming Gaussian posterior: log N(z; mu, var)
        log_q_z_given_x = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp())

        # Calculate log q(z) and log \prod q(z_j) using the minibatch approximation
        # Reshape to compute pairwise log densities across the batch
        z_expanded = z.unsqueeze(1)      # (Batch, 1, Latent_Dim)
        mu_expanded = mu.unsqueeze(0)    # (1, Batch, Latent_Dim)
        logvar_expanded = logvar.unsqueeze(0) # (1, Batch, Latent_Dim)

        pairwise_log_q_z = -0.5 * (math.log(2 * math.pi) + logvar_expanded + (z_expanded - mu_expanded).pow(2) / logvar_expanded.exp())

        # log q(z) is the logsumexp across the batch dimension
        log_q_z = torch.logsumexp(pairwise_log_q_z.sum(dim=2), dim=1) - math.log(batch_size * batch_size)

        # log \prod q(z_j) is the sum of marginals
        log_prod_q_z_j = (torch.logsumexp(pairwise_log_q_z, dim=1) - math.log(batch_size)).sum(dim=1)

        # Total Correlation = E_q(z|x) [log q(z) - log \prod q(z_j)]
        tc_loss = (log_q_z - log_prod_q_z_j).mean()

        # We also need the standard KL-divergence for the priors
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        return tc_loss, kl_loss
