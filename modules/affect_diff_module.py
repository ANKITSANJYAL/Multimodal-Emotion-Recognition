import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Assuming these would be imported from our previously written files
from models.fusion.latent_bottleneck import LatentBottleneck
from models.diffusion.unet_1d import UNet1D
from models.diffusion.diffusion_utils import AffectiveDiffusion

class AffectDiffModule(pl.LightningModule):
    """
    The PyTorch Lightning orchestrator for Affect-Diff.
    Handles the joint optimization of the VAE bottleneck, the task classifier,
    and the diffusion generative prior.
    """
    def __init__(
        self,
        text_dim=768,
        audio_dim=768,
        video_dim=512,
        latent_dim=256,
        num_classes=6,      # CMU-MOSEI standard 6 emotions
        diffusion_steps=1000,
        lr=1e-4,
        weight_decay=1e-4,
        beta_kl=5.0,        # Beta-VAE penalty
        lambda_diff=1.0     # Weight for diffusion loss
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1. The Multimodal Bottleneck (from raw features to z)
        self.bottleneck = LatentBottleneck(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            latent_dim=latent_dim
        )

        # 2. The Task Classifier (Transformer-Attention Pooling)
        # Instead of averaging, we use a learnable Query to 'attend' to expressiveness
        self.attention_query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

        # 3. The Generative Diffusion Prior (Learns the manifold of z)
        self.unet = UNet1D(latent_dim=latent_dim, num_classes=num_classes)
        self.diffusion = AffectiveDiffusion(self.unet, timesteps=diffusion_steps)

        # 4. DeepMind Level EMA: Shadow weights for the Generative Prior
        # This makes sampling significantly more robust and stable.
        self.ema_unet = copy.deepcopy(self.unet)
        for param in self.ema_unet.parameters():
            param.requires_grad = False
        self.ema_decay = 0.999

    def _pool_features(self, z):
        """
        Applies learnable Attention Pooling to the latent sequence.
        z: (Batch, Seq_Len, latent_dim)
        Returns: (Batch, latent_dim)
        """
        # Attention Pooling: (B, 1, Dim) @ (B, Dim, Seq) -> (B, 1, Seq)
        z_perm = z.transpose(1, 2)
        attn_weights = torch.matmul(self.attention_query.expand(z.shape[0], -1, -1), z_perm)
        attn_weights = F.softmax(attn_weights / (self.hparams.latent_dim ** 0.5), dim=-1)
        
        # Weighted sum: (B, 1, Seq) @ (B, Seq, Dim) -> (B, Dim)
        z_pooled = torch.matmul(attn_weights, z).squeeze(1)
        return z_pooled

    def forward(self, text, audio, video):
        """Standard forward pass."""
        z, _, _, _ = self.bottleneck(text, audio, video) # (B, Seq, Dim)
        z_pooled = self._pool_features(z)
        logits = self.classifier(z_pooled)
        return logits

    def shared_step(self, batch, batch_idx, stage="train"):
        text, audio, video, labels = batch['text'], batch['audio'], batch['vision'], batch['labels']

        # DeepMind Level Input Sanitization
        text = torch.nan_to_num(text)
        audio = torch.nan_to_num(audio)
        video = torch.nan_to_num(video)

        # --- DeepMind Research Grade: Stochastic Modality Ablation ---
        # Forces model to not rely on just one modality (e.g. Text)
        if stage == "train":
            # 10% chance to drop each modality UNCORRELATEDLY
            # This ensures the model learns to handle missing/noisy sensors
            m_drop = torch.rand(3, device=self.device)
            if m_drop[0] < 0.1: text = torch.zeros_like(text)
            if m_drop[1] < 0.1: audio = torch.zeros_like(audio)
            if m_drop[2] < 0.1: video = torch.zeros_like(video)

        # Step 1: Encode into latent space and extract Causal Graph
        z_permuted, mu, logvar, adj_matrix = self.bottleneck(text, audio, video)

        # Force z into a safe numerical box
        z_permuted = torch.clamp(z_permuted, min=-10.0, max=10.0)

        # Step 2: Task Loss (Emotion Classification with Attention Pooling)
        # bottleneck returns (B, Dim, Seq) for historical reasons, we pool it
        # z here is (B, Seq, Dim)
        z = z_permuted.permute(0, 2, 1)
        z_pooled = self._pool_features(z)

        logits = self.classifier(z_pooled)
        loss_task = F.cross_entropy(logits, labels.long())

        # Step 3: Disentanglement Loss (Beta-VAE)
        loss_kl = self.bottleneck.compute_kl_loss(mu, logvar, beta=self.hparams.beta_kl)

        # Step 4: Causal Regularization (DeepMind Level Sparsity)
        loss_causal_reg = torch.norm(adj_matrix, p=1) / (adj_matrix.shape[0] * adj_matrix.shape[1] * adj_matrix.shape[2])

        # Step 5: Generative Diffusion Loss with Causal Conditioning
        b = text.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (b,), device=self.device).long()
        
        # Classifier-Free Guidance Training: Increase to 20% for stronger conditioning
        drop_mask = (torch.rand(b, device=self.device) < 0.2).long()
        null_label = self.hparams.num_classes
        labels_cond = labels.long() * (1 - drop_mask) + null_label * drop_mask

        # Extract causal influence from the graph to condition the UNet
        causal_influence = adj_matrix.sum(dim=1) # (B, 3)
        # Clamp influence to prevent conditioning explosion
        causal_influence = torch.clamp(causal_influence, min=0.0, max=5.0)

        loss_diff = self.diffusion.p_losses(
            x_start=z_permuted, 
            t=t, 
            label=labels_cond, 
            causal_weights=causal_influence
        )

        # Step 6: Joint Loss with Generative Warmup (Curriculum Learning)
        warmup_factor = min(1.0, (self.current_epoch + 1) / 5.0)
        
        loss_total = (
            loss_task + 
            (warmup_factor * loss_kl) + 
            (warmup_factor * self.hparams.lambda_diff * loss_diff) + 
            (0.1 * loss_causal_reg)
        )
        
        # --- DeepMind Stability: NaN Detection & Recovery ---
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            self.print(f"[Stability Warning] Rank {self.global_rank}: NaN detected in {stage} step. Skipping gradient update.")
            loss_total = loss_task * 0.0 # Return a 0 gradient placeholder
            loss_diff = torch.tensor(0.0, device=self.device)
            loss_causal_reg = torch.tensor(0.0, device=self.device)

        # --- Logging ---
        self.log(f'{stage}_loss', loss_total, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_loss_diff', loss_diff, sync_dist=True)
        self.log(f'{stage}_loss_causal', loss_causal_reg, sync_dist=True)
        
        if stage == "val":
            weights = self.bottleneck.causal_graph.get_causal_weights(adj_matrix)
            for k, v in weights.items():
                self.log(f'causal/{k}', v, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels.long()).float().mean()
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)

        # Step 7: Granular Robustness Checks (Time=0 Reconstruction)
        with torch.no_grad():
            t_zero = torch.zeros((b,), device=self.device).long()
            z_pred_zero = self.unet(z_permuted, t_zero, label=labels, causal_weights=causal_influence)
            # Log the reconstruction fidelity at the clean limit
            latent_drift = F.mse_loss(z_permuted, z_pred_zero)
            self.log(f'{stage}/latent_drift_t0', latent_drift, sync_dist=True)

        return loss_total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA weights after every training step."""
        with torch.no_grad():
            for p, ema_p in zip(self.unet.parameters(), self.ema_unet.parameters()):
                ema_p.data.mul_(self.hparams.get('ema_decay', 0.999)).add_(p.data, alpha=1 - self.hparams.get('ema_decay', 0.999))

    def on_validation_epoch_end(self):
        """
        DeepMind Verification: Counterfactual Hallucination Check.
        We sample latents from the DIFFUSION manifold and check if our CLASSIFIER
        recognizes the intended emotion. This proves the UNet is learning.
        """
        if self.current_epoch % 5 != 0: # Check every 5 epochs to save time
            return
            
        # DDP Stability: Only run the heavy sampling on Rank 0
        if self.global_rank == 0:
            self.print(f"\n[Verification] Epoch {self.current_epoch}: Conducting latent hallucination check...")
            
            # Sample 4 random emotions
            device = self.device
            sample_labels = torch.arange(min(4, self.hparams.num_classes), device=device)
            b = sample_labels.shape[0]
            
            # Use a dummy causal weight (identity-like) for the loop
            causal_influence = torch.ones((b, 3), device=device)
            
            # Temporarily use EMA weights for the diffusion sampler
            original_unet = self.diffusion.unet
            self.diffusion.unet = self.ema_unet
            
            try:
                # Hallucinate latents: Noise -> Denoising -> Latent Emotion
                hallucinated_z = self.diffusion.p_sample_loop(
                    shape=(b, self.hparams.latent_dim, 50), # 50 is the MOSEI seq_len
                    device=device,
                    label=sample_labels,
                    causal_weights=causal_influence,
                    cfg_scale=4.0 # Slightly higher guidance for SOTA push
                )
                
                # Check if our task classifier recognizes these halluncinated states
                with torch.no_grad():
                    # Pool back to (B, Dim) before classification
                    z_hallucinated = hallucinated_z.permute(0, 2, 1)
                    z_pooled = self._pool_features(z_hallucinated)
                    logits = self.classifier(z_pooled)
                    preds = torch.argmax(logits, dim=1)
                    hallucination_acc = (preds == sample_labels).float().mean()
                    
                self.log("val/hallucination_acc", hallucination_acc, rank_zero_only=True)
                self.print(f"[Verification] Hallucination Accuracy: {hallucination_acc:.4f} (Targets: {sample_labels.tolist()} -> Preds: {preds.tolist()})")
                
            except Exception as e:
                self.print(f"[Verification Error] Sampling failed: {e}")
            finally:
                # Restore original UNet weights
                self.diffusion.unet = original_unet
        
        # Ensure all ranks wait for Rank 0 to finish sampling before proceeding
        # This prevents the collective timeout during the transition to the next hook
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_train_epoch_start(self):
        """Anneal causal temperature from 1.0 to 0.5 over 10 epochs."""
        new_temp = max(0.5, 1.0 - (self.current_epoch * 0.05))
        self.bottleneck.causal_graph.set_temperature(new_temp)
        # DeepMind DDP Stability: sync_dist=True is mandatory for epoch-level metrics
        self.log("causal/temperature", new_temp, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        # Explicitly defined to fix Trainer.test MisconfigurationException
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # Cosine Annealing with Warmup can be complex in PL, using standard for now
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
