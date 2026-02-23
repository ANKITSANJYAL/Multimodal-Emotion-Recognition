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
        self.unet = UNet1D(latent_dim=latent_dim)
        self.diffusion = AffectiveDiffusion(self.unet, timesteps=diffusion_steps)

    def forward(self, text, audio, video):
        """
        Standard forward pass for inference.
        Bypasses diffusion to just predict the emotion label.
        """
        z_permuted, _, _ = self.bottleneck(text, audio, video)
        logits = self.classifier(z_permuted)
        return logits

    def shared_step(self, batch, batch_idx, stage="train"):
        """
        The core mathematical step shared by training and validation.
        Calculates the joint loss function.
        """
        text = batch['text']
        audio = batch['audio']
        video = batch['vision']
        labels = batch['labels'].long() # Shape: (B,)

        # Step 1: Encode into latent space
        z_permuted, mu, logvar = self.bottleneck(text, audio, video)

        # Step 2: Task Loss (Cross Entropy for Emotion Classification)
        logits = self.classifier(z_permuted)
        loss_task = F.cross_entropy(logits, labels)

        # Step 3: Disentanglement Loss (KL-Divergence / Total Correlation)
        loss_kl = self.bottleneck.compute_kl_loss(mu, logvar, beta=self.hparams.beta_kl)

        # Step 4: Generative Diffusion Loss
        # Sample random timesteps for the batch
        b = text.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (b,), device=self.device).long()
        # Train diffusion to predict the noise injected into the latent space
        loss_diff = self.diffusion.p_losses(x_start=z_permuted, t=t)

        # Step 5: Joint Loss Calculation
        loss_total = loss_task + loss_kl + (self.hparams.lambda_diff * loss_diff)

        # --- Logging ---
        # Lightning handles epoch-level averaging automatically
        self.log(f'{stage}_loss', loss_total, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_loss_task', loss_task, sync_dist=True)
        self.log(f'{stage}_loss_kl', loss_kl, sync_dist=True)
        self.log(f'{stage}_loss_diff', loss_diff, sync_dist=True)

        # Calculate standard accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)

        return loss_total

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        """
        Sets up AdamW (crucial for Transformers/Diffusion) and a Cosine learning rate scheduler.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine Annealing gently reduces the learning rate to help the model converge
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=100, # Assuming ~100 epochs, can be tuned via Hydra config
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
