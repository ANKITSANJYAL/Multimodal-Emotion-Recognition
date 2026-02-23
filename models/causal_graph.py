import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttentionGraph(nn.Module):
    """
    Learns a differentiable adjacency matrix representing the causal influence
    of each modality on the others and on the final affective state.
    Provides the intrinsic explainability required for CVPR.
    """
    def __init__(self, latent_dim=256, num_nodes=3):
        super().__init__()
        self.num_nodes = num_nodes # T, A, V
        self.latent_dim = latent_dim

        # Node embedding projections
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)

        # Gumbel-Softmax temperature for sparse, DAG-like edge generation
        self.register_buffer('temperature', torch.tensor(0.5))

    def forward(self, z_t, z_a, z_v):
        """
        Expects unimodal latents of shape (Batch, Seq_Len, Latent_Dim).
        Returns the Causal Adjacency Matrix.
        """
        # Pool temporal dimension to get node representations: (Batch, Latent_Dim)
        z_t_pooled = z_t.mean(dim=1)
        z_a_pooled = z_a.mean(dim=1)
        z_v_pooled = z_v.mean(dim=1)

        # Stack nodes: (Batch, Num_Nodes, Latent_Dim)
        nodes = torch.stack([z_t_pooled, z_a_pooled, z_v_pooled], dim=1)

        queries = self.query_proj(nodes)
        keys = self.key_proj(nodes)

        # Calculate unnormalized attention scores (Batch, Num_Nodes, Num_Nodes)
        # Scaled dot-product
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.latent_dim ** 0.5)

        # Apply Gumbel-Softmax to force sparse, discrete causal edges
        # instead of a dense, uninterpretable attention matrix
        adj_matrix = F.gumbel_softmax(scores, tau=self.temperature, hard=False, dim=-1)

        # Mask out self-loops (a modality shouldn't causally influence itself in this graph)
        mask = torch.eye(self.num_nodes, device=adj_matrix.device).unsqueeze(0).bool()
        adj_matrix = adj_matrix.masked_fill(mask, 0.0)

        return adj_matrix

    def get_causal_weights(self, adj_matrix):
        """
        Decodes the adjacency matrix into human-readable influence scores.
        Returns a dictionary mapping text, audio, and video to their causal weights.
        """
        # Average across the batch
        mean_adj = adj_matrix.mean(dim=0)

        # Sum of incoming edges for each node determines its graph centrality (importance)
        importance = mean_adj.sum(dim=0)
        total = importance.sum() + 1e-8
        normalized_importance = importance / total

        return {
            'Text_Influence': normalized_importance[0].item(),
            'Audio_Influence': normalized_importance[1].item(),
            'Video_Influence': normalized_importance[2].item()
        }
