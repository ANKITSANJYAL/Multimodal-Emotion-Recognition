import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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

        # 2. The Task Classifier (Predicts emotion directly from z)
        # We pool across the time dimension (Sequence Length) and project to classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

        # 3. The Generative Diffusion Prior (Learns the manifold of z)
        self.unet = UNet1D(latent_dim=latent_dim, num_classes=num_classes)
        self.diffusion = AffectiveDiffusion(self.unet, timesteps=diffusion_steps)

    def forward(self, text, audio, video):
        """Standard forward pass."""
        z_permuted, _, _, _ = self.bottleneck(text, audio, video)
        logits = self.classifier(z_permuted)
        return logits

    def shared_step(self, batch, batch_idx, stage="train"):
        text, audio, video, labels = batch['text'], batch['audio'], batch['vision'], batch['labels']

        # DeepMind Level Input Sanitization
        text = torch.nan_to_num(text)
        audio = torch.nan_to_num(audio)
        video = torch.nan_to_num(video)

        # Step 1: Encode into latent space and extract Causal Graph
        z_permuted, mu, logvar, adj_matrix = self.bottleneck(text, audio, video)

        # Force z into a safe numerical box
        z_permuted = torch.clamp(z_permuted, min=-10.0, max=10.0)

        # Step 2: Task Loss (Emotion Classification)
        logits = self.classifier(z_permuted)
        loss_task = F.cross_entropy(logits, labels.long())

        # Step 3: Disentanglement Loss (Beta-VAE)
        loss_kl = self.bottleneck.compute_kl_loss(mu, logvar, beta=self.hparams.beta_kl)

        # Step 4: Causal Regularization (DeepMind Level Sparsity)
        loss_causal_reg = torch.norm(adj_matrix, p=1) / (adj_matrix.shape[0] * adj_matrix.shape[1] * adj_matrix.shape[2])

        # Step 5: Generative Diffusion Loss with Causal Conditioning
        b = text.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (b,), device=self.device).long()
        
        # Classifier-Free Guidance Training: 10% chance to drop label
        drop_mask = (torch.rand(b, device=self.device) < 0.1).long()
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

        return loss_total

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
