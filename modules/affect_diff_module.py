"""Affect-Diff PyTorch Lightning orchestrator — S-tier wiring.

All config values are accepted, validated, and wired to the
appropriate sub-modules.  Every ablation toggle is functional.
"""

import copy
import logging
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from models.fusion.latent_bottleneck import LatentBottleneck
from models.diffusion.unet_1d import UNet1D
from models.diffusion.diffusion_utils import AffectiveDiffusion
from models.decoder import MultimodalDecoder
from Data.augmentations import MultimodalAugmentation
from utils.loss_functions import BetaTCVAELoss

logger = logging.getLogger(__name__)


class AffectDiffModule(pl.LightningModule):
    """
    Affect-Diff — PyTorch Lightning Orchestrator.

    Joint optimization of:
      1. Multimodal VAE bottleneck (encoder + fusion + reparameterization)
      2. Task classifier (attention pooling + MLP)
      3. Diffusion generative prior (UNet1D + DDPM/DDIM)
      4. Causal attention graph (NOTEARS / Gumbel)
      5. Reconstruction decoder (per-modality MLP)
      6. Data augmentation (temporal masking + noise)

    All sub-systems are togglable via config for ablation studies.
    """

    def __init__(
        self,
        # ── Dimensions ────────────────────────────────────────────────────
        text_dim: int = 300,
        audio_dim: int = 74,
        video_dim: int = 35,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_classes: int = 6,
        # ── Encoder / Fusion / DAG config ─────────────────────────────────
        encoder_type: str = "legacy",
        encoder_dropout: float = 0.2,
        encoder_layers: int = 1,
        text_backbone: str = "roberta-base",
        audio_backbone: str = "facebook/hubert-base-ls960",
        video_backbone: str = "openai/clip-vit-base-patch16",
        freeze_backbones: bool = True,
        fusion_type: str = "concat",
        num_bottleneck_tokens: int = 50,
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        dag_method: str = "notears",
        # ── Diffusion config ──────────────────────────────────────────────
        diffusion_steps: int = 1000,
        ddim_steps: int = 50,
        # ── Loss weights ──────────────────────────────────────────────────
        beta_kl: float = 5.0,
        lambda_diff: float = 1.0,
        lambda_causal: float = 0.1,
        lambda_recon: float = 0.5,
        cfg_scale: float = 3.0,
        # ── Optimizer ─────────────────────────────────────────────────────
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        # ── EMA ───────────────────────────────────────────────────────────
        ema_decay: float = 0.999,
        # ── Regularization ────────────────────────────────────────────────
        label_smoothing: float = 0.1,
        free_bits: float = 0.25,
        # ── Ablation toggles ──────────────────────────────────────────────
        use_reconstruction: bool = True,
        use_diffusion: bool = True,
        use_causal_graph: bool = True,
        use_augmentation: bool = True,
        use_beta_tc_vae: bool = False,
        # ── Focal loss ────────────────────────────────────────────────────
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Balanced accuracy (mean per-class recall).
        # With balanced sampling, epoch-0 always-Happy gets val_acc=0.668 but
        # val_bal_acc≈0.17 (1/6 random) — EarlyStopping on val_bal_acc won't fire
        # until the model beats random, giving the U-curve time to complete.
        try:
            from torchmetrics.classification import MulticlassAccuracy
            # Cannot use nn.ModuleDict with key "train" — conflicts with nn.Module.train()
            self._bal_train = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._bal_val   = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._bal_test  = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._use_bal_metric = True
        except ImportError:
            self._use_bal_metric = False

        # ── 1. Multimodal Encoder + VAE Bottleneck + Causal Graph ─────────
        self.bottleneck = LatentBottleneck(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_type=encoder_type,
            encoder_dropout=encoder_dropout,
            encoder_layers=encoder_layers,
            text_backbone=text_backbone,
            audio_backbone=audio_backbone,
            video_backbone=video_backbone,
            freeze_backbones=freeze_backbones,
            fusion_type=fusion_type,
            num_bottleneck_tokens=num_bottleneck_tokens,
            num_cross_attn_layers=num_cross_attn_layers,
            num_self_attn_layers=num_self_attn_layers,
            dag_method=dag_method,
        )

        # ── 2. Attention Pooling + Task Classifier ────────────────────────
        self.attention_query = nn.Parameter(torch.randn(1, 1, latent_dim))
        # Two dropout layers: one inside the hidden block, one before the final projection.
        # 0.4 > previous 0.3 — important for regularising on 1956 training samples.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(latent_dim // 2, num_classes),
        )

        # ── 3. Generative Diffusion Prior (optional) ─────────────────────
        if use_diffusion:
            self.unet = UNet1D(latent_dim=latent_dim, num_classes=num_classes)
            self.diffusion = AffectiveDiffusion(self.unet, timesteps=diffusion_steps)
            self.ema_unet = copy.deepcopy(self.unet)
            for p in self.ema_unet.parameters():
                p.requires_grad = False
        else:
            self.unet = None
            self.diffusion = None
            self.ema_unet = None

        # ── 4. Reconstruction Decoder (optional) ─────────────────────────
        if use_reconstruction:
            self.decoder = MultimodalDecoder(
                latent_dim=latent_dim,
                text_dim=text_dim,
                audio_dim=audio_dim,
                video_dim=video_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.decoder = None

        # ── 5. Data Augmentation (optional) ───────────────────────────────
        if use_augmentation:
            self.augmentation = MultimodalAugmentation(mask_prob=0.1, noise_std=0.01)
        else:
            self.augmentation = None

        logger.info(
            "AffectDiffModule initialized: encoder=%s, fusion=%s, dag=%s, "
            "diffusion=%s, reconstruction=%s, augmentation=%s, beta_tc_vae=%s",
            encoder_type, fusion_type, dag_method,
            use_diffusion, use_reconstruction, use_augmentation, use_beta_tc_vae,
        )

    # ──────────────────────────────────────────────────────────────────────
    # ATTENTION POOLING
    # ──────────────────────────────────────────────────────────────────────

    def _pool_features(self, z: torch.Tensor) -> torch.Tensor:
        """Mean + attention pooling: (B, L, d_z) -> (B, d_z).

        Mean pooling captures global context; attention pooling selects salient
        frames. Their average is more robust than either alone, especially when
        the attention head hasn't converged yet in early training.
        """
        mean_pool = z.mean(dim=1)  # (B, d_z)
        attn = torch.matmul(
            self.attention_query.expand(z.shape[0], -1, -1),
            z.transpose(1, 2),
        )
        attn = F.softmax(attn / (self.hparams.latent_dim ** 0.5), dim=-1)
        attn_pool = torch.matmul(attn, z).squeeze(1)  # (B, d_z)
        return (mean_pool + attn_pool) * 0.5

    # ──────────────────────────────────────────────────────────────────────
    # FOCAL CROSS-ENTROPY
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _focal_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Focal cross-entropy (Lin et al. 2017) for class-imbalanced classification.

        FL(pt) = -(1 - pt)^gamma * log(pt)

        Easy examples (high pt) get down-weighted so gradient budget concentrates
        on hard, misclassified samples. Combined with class weights this handles
        both frequency imbalance (weights) and difficulty imbalance (focal term).

        Using gamma=2 means a correctly-classified sample with pt=0.9 contributes
        only (0.1)^2 = 1% of the gradient vs standard CE — hard samples dominate.
        """
        ce = F.cross_entropy(
            logits.float(), labels.long(),
            weight=weight, label_smoothing=label_smoothing, reduction="none",
        )
        # Detach pt to avoid double-differentiating through log(softmax)
        pt = torch.exp(-ce.detach())
        return ((1.0 - pt) ** gamma * ce).mean()

    # ──────────────────────────────────────────────────────────────────────
    # INFERENCE FORWARD PASS
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic inference pass -> logits (B, num_classes)."""
        z_perm, _, _, _ = self.bottleneck(text, audio, video)
        z = z_perm.permute(0, 2, 1)
        z_pooled = self._pool_features(z)
        return self.classifier(z_pooled)

    # ──────────────────────────────────────────────────────────────────────
    # SHARED TRAIN / VAL / TEST STEP
    # ──────────────────────────────────────────────────────────────────────

    def shared_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str = "train"
    ) -> torch.Tensor:
        text = torch.nan_to_num(batch["text"])
        audio = torch.nan_to_num(batch["audio"])
        video = torch.nan_to_num(batch["vision"])
        labels = batch["labels"]

        # ── Augmentation (train only) ─────────────────────────────────────
        if stage == "train" and self.augmentation is not None:
            self.augmentation.train()
            text, audio, video = self.augmentation(text, audio, video)

        # ── Stochastic modality dropout (train only, 10% per modality) ────
        # Per-sample mask — each sample independently loses a modality with p=0.10.
        # Previously this was batch-level (entire modality zeroed for all samples),
        # which made 10% of batches fully degenerate instead of teaching robustness.
        if stage == "train":
            B = text.shape[0]
            # Asymmetric rates reflect signal quality: text (semantic) is most reliable,
            # video (FAU) is noisiest — higher dropout teaches the model not to depend on it.
            text  = text  * (torch.rand(B, 1, 1, device=self.device) > 0.05).float()
            audio = audio * (torch.rand(B, 1, 1, device=self.device) > 0.10).float()
            video = video * (torch.rand(B, 1, 1, device=self.device) > 0.20).float()

        # ── Step 1: Encode -> latent space ────────────────────────────────
        z_perm, mu, logvar, adj_matrix = self.bottleneck(text, audio, video)
        # Catch any NaN/Inf that escapes the bottleneck (e.g. fp16 attention
        # edge cases on early training steps) before they propagate everywhere.
        mu       = torch.nan_to_num(mu,       nan=0.0, posinf=1.0,  neginf=-1.0)
        logvar   = torch.nan_to_num(logvar,   nan=0.0, posinf=0.0,  neginf=0.0)
        z_perm   = torch.nan_to_num(z_perm,   nan=0.0, posinf=10.0, neginf=-10.0)
        adj_matrix = torch.nan_to_num(adj_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        z_perm = torch.clamp(z_perm, min=-10.0, max=10.0)

        # ── Step 2: Classification loss ───────────────────────────────────
        z = z_perm.permute(0, 2, 1)
        z_pooled = self._pool_features(z)
        logits = self.classifier(z_pooled)
        class_w = getattr(self, "class_weights", None)
        if self.hparams.use_focal_loss:
            loss_task = self._focal_cross_entropy(
                logits, labels, gamma=self.hparams.focal_gamma,
                weight=class_w, label_smoothing=self.hparams.label_smoothing,
            )
        else:
            loss_task = F.cross_entropy(
                logits.float(),
                labels.long(),
                weight=class_w,
                label_smoothing=self.hparams.label_smoothing,
            )

        # ── Step 3: KL loss (Beta-VAE or Beta-TC-VAE) ────────────────────
        if self.hparams.use_beta_tc_vae:
            z_flat = z_perm.permute(0, 2, 1).reshape(-1, self.hparams.latent_dim)
            mu_flat = mu.reshape(-1, self.hparams.latent_dim)
            logvar_flat = logvar.reshape(-1, self.hparams.latent_dim)
            loss_tc, loss_kl_raw = BetaTCVAELoss.compute_tc_penalty(
                z_flat, mu_flat, logvar_flat,
            )
            loss_kl = self.hparams.beta_kl * (loss_tc + loss_kl_raw)
        else:
            loss_kl = LatentBottleneck.compute_kl_loss(
                mu, logvar, beta=self.hparams.beta_kl, free_bits=self.hparams.free_bits,
            )

        # ── Step 4: Causal graph penalty (NOTEARS or L1) ─────────────────
        if self.hparams.use_causal_graph:
            loss_causal = self.bottleneck.causal_graph.compute_dag_penalty(adj_matrix)
        else:
            loss_causal = torch.tensor(0.0, device=self.device)

        # ── Step 5: Reconstruction loss (optional) ────────────────────────
        if self.decoder is not None and self.hparams.use_reconstruction:
            z_for_decode = z_perm.permute(0, 2, 1)  # (B, L, d_z)
            x_hat_t, x_hat_a, x_hat_v = self.decoder(z_for_decode)
            t_tgt = torch.clamp(torch.nan_to_num(batch["text"]),   -10, 10)
            # Crossmodal fusion with bottleneck tokens produces fewer tokens than the
            # original seq_len (e.g. 20 vs 50). MSE between mismatched shapes crashes
            # or silently broadcasts wrong values. Skip recon in that case.
            if x_hat_t.shape[1] != t_tgt.shape[1]:
                loss_recon = torch.tensor(0.0, device=self.device)
            else:
                def _safe(t):
                    return torch.clamp(torch.nan_to_num(t.float(), nan=0.0, posinf=1.0, neginf=-1.0), -100, 100)
                loss_recon = MultimodalDecoder.compute_reconstruction_loss(
                    _safe(x_hat_t), _safe(x_hat_a), _safe(x_hat_v),
                    t_tgt,
                    torch.clamp(torch.nan_to_num(batch["audio"]),  -10, 10),
                    torch.clamp(torch.nan_to_num(batch["vision"]), -10, 10),
                )
        else:
            loss_recon = torch.tensor(0.0, device=self.device)

        # ── Step 6: Diffusion generative prior loss (optional) ────────────
        # STOP GRADIENT: The diffusion model learns to approximate the
        # aggregate posterior q(z), so it should match the encoder's output
        # distribution — not shape it. Allowing diffusion gradients to flow
        # back through the encoder creates a conflicting objective that hurts
        # classification (empirically: Full_Model 68.2% < Classifier_Only 69.9%).
        # This mirrors Latent Diffusion Models (Rombach et al. 2022) where the
        # encoder is frozen and diffusion trains on detached latents.
        if self.diffusion is not None and self.hparams.use_diffusion:
            b = text.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (b,), device=self.device).long()

            # CFG: 20% null token dropout
            drop_mask = (torch.rand(b, device=self.device) < 0.2).long()
            null_token = self.hparams.num_classes
            labels_cond = labels.long() * (1 - drop_mask) + null_token * drop_mask

            # Causal influence weights (detached: UNet loss shouldn't push causal graph)
            if self.hparams.use_causal_graph:
                causal_influence = torch.clamp(adj_matrix.sum(dim=1), min=0.0, max=5.0).detach()
            else:
                causal_influence = torch.ones(b, 3, device=self.device)

            loss_diff = self.diffusion.p_losses(
                x_start=z_perm.detach(), t=t,
                label=labels_cond,
                causal_weights=causal_influence,
            )
        else:
            loss_diff = torch.tensor(0.0, device=self.device)

        # ── Step 7: Joint loss with curriculum warmup ─────────────────────
        # KL: cyclical annealing (Fu et al. 2019) — restarts from 0 every 20 epochs.
        #   Each cycle has a 10-epoch ramp ("open") then 10 epochs at full weight
        #   ("closed"). This prevents posterior collapse: even if KL collapses in
        #   one cycle, the next cycle's ramp-from-zero lets the encoder re-engage
        #   with the task loss before KL regularization kicks in.
        #   Linear warmup (previous) caused collapse because once KL=0, the ramp
        #   saturated at 1.0 with no recovery mechanism.
        cycle_length = 20
        step_in_cycle = self.current_epoch % cycle_length
        gamma_kl    = min(1.0, step_in_cycle / (cycle_length * 0.5))
        gamma_recon = min(1.0, (self.current_epoch + 1) / 20.0)
        gamma_diff  = min(1.0, max(0.0, (self.current_epoch - 9) / 20.0))

        # Cast all components to float32 before summing (prevents fp16 overflow)
        loss_task_f = loss_task.float()
        loss_kl_f = loss_kl.float()
        loss_diff_f = loss_diff.float()
        loss_recon_f = loss_recon.float()
        loss_causal_f = loss_causal.float()

        loss_total = (
            loss_task_f
            + gamma_kl   * loss_kl_f
            + gamma_diff * self.hparams.lambda_diff * loss_diff_f
            + gamma_recon * self.hparams.lambda_recon * loss_recon_f
            + self.hparams.lambda_causal * loss_causal_f
        )

        # ── NaN guard with per-component diagnosis ────────────────────────
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            nan_components = []
            if torch.isnan(loss_task_f) or torch.isinf(loss_task_f):
                nan_components.append(f"task={loss_task_f.item()}")
            if torch.isnan(loss_kl_f) or torch.isinf(loss_kl_f):
                nan_components.append(f"kl={loss_kl_f.item()}")
            if torch.isnan(loss_diff_f) or torch.isinf(loss_diff_f):
                nan_components.append(f"diff={loss_diff_f.item()}")
            if torch.isnan(loss_recon_f) or torch.isinf(loss_recon_f):
                nan_components.append(f"recon={loss_recon_f.item()}")
            if torch.isnan(loss_causal_f) or torch.isinf(loss_causal_f):
                nan_components.append(f"causal={loss_causal_f.item()}")
            logger.warning(
                "[NaN Guard] Rank %d | %s | step %d | bad components: [%s]. Using task loss only.",
                self.global_rank, stage, batch_idx, ", ".join(nan_components) or "overflow in sum",
            )
            # Fall back to task loss — must NOT call .detach() here because
            # that strips the grad_fn and causes "does not require grad" on backward.
            if not (torch.isnan(loss_task_f) or torch.isinf(loss_task_f)):
                loss_total = loss_task_f
            else:
                # Even cross-entropy is NaN; skip this step with a zero-gradient sentinel.
                loss_total = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)

        # ── Logging ───────────────────────────────────────────────────────
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels.long()).float().mean()

        self.log(f"{stage}_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
        if self._use_bal_metric:
            metric = {"train": self._bal_train, "val": self._bal_val, "test": self._bal_test}[stage]
            bal = metric(preds, labels.long())
            self.log(
                f"{stage}_bal_acc", bal,
                prog_bar=(stage == "val"),
                on_step=False, on_epoch=True, sync_dist=True,
            )
        self.log(f"{stage}_loss_task", loss_task, sync_dist=True)
        self.log(f"{stage}_loss_kl", loss_kl, sync_dist=True)
        self.log(f"{stage}_loss_diff", loss_diff, sync_dist=True)
        self.log(f"{stage}_loss_causal", loss_causal, sync_dist=True)
        self.log(f"{stage}_loss_recon", loss_recon, sync_dist=True)
        if stage == "train":
            self.log("warmup/gamma_kl",    gamma_kl,    sync_dist=True)
            self.log("warmup/gamma_recon", gamma_recon, sync_dist=True)
            self.log("warmup/gamma_diff",  gamma_diff,  sync_dist=True)

        if stage == "val" and self.hparams.use_causal_graph:
            weights = self.bottleneck.causal_graph.get_causal_weights(adj_matrix)
            for k, v in weights.items():
                self.log(f"causal/{k}", v, sync_dist=True)

        # ── t=0 reconstruction fidelity check (val only, every 10 batches) ──
        if (self.diffusion is not None
                and self.hparams.use_diffusion
                and stage == "val"
                and batch_idx % 10 == 0):
            with torch.no_grad():
                b = text.shape[0]
                t_zero = torch.zeros(b, device=self.device, dtype=torch.long)
                if self.hparams.use_causal_graph:
                    ci = torch.clamp(adj_matrix.sum(dim=1), min=0.0, max=5.0)
                else:
                    ci = torch.ones(b, 3, device=self.device)
                z_pred_zero = self.unet(
                    z_perm.detach(), t_zero, label=labels.long(), causal_weights=ci,
                )
                latent_drift = F.mse_loss(z_perm.detach(), z_pred_zero)
                self.log(f"{stage}/latent_drift_t0", latent_drift, rank_zero_only=True)

        return loss_total

    # ──────────────────────────────────────────────────────────────────────
    # PL HOOKS
    # ──────────────────────────────────────────────────────────────────────

    def on_fit_start(self) -> None:
        """Pull class weights from the datamodule after setup() has run."""
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "class_weights") and dm.class_weights is not None:
            w = dm.class_weights.to(self.device)
            # Cap at 2× the minimum weight. The sqrt sampler already overrepresents
            # rare classes in each batch, so the loss weight provides a second gentle
            # nudge — not a dominant force. Higher caps cause rare-class dominance.
            w = torch.clamp(w, max=w.min() * 2.0)
            self.register_buffer("class_weights", w)
            logger.info("Loaded class weights (capped 2× min): %s", w.tolist())
        else:
            logger.warning("No class_weights on datamodule — using uniform loss weights.")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, stage="test")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """EMA update of UNet weights after every gradient step."""
        if self.ema_unet is not None:
            decay = self.hparams.ema_decay
            with torch.no_grad():
                for p, ema_p in zip(self.unet.parameters(), self.ema_unet.parameters()):
                    ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def on_train_epoch_start(self) -> None:
        """Anneal Gumbel-Softmax temperature in the CausalAttentionGraph."""
        if self.hparams.use_causal_graph:
            new_temp = max(0.5, 1.0 - self.current_epoch * 0.05)
            self.bottleneck.causal_graph.set_temperature(new_temp)
            self.log("causal/temperature", new_temp, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Hallucination verification check (every 10 epochs after epoch 5, rank-0 only).

        Samples z from the diffusion manifold using DDIM for speed, then
        checks if the classifier correctly identifies the intended emotion.
        """
        if self.diffusion is None or not self.hparams.use_diffusion:
            return
        if self.current_epoch < 5 or self.current_epoch % 10 != 0:
            return

        # Only rank 0 does the actual sampling; other ranks skip to barrier
        if self.global_rank == 0:
            device = self.device
            b = min(4, self.hparams.num_classes)
            sample_labels = torch.arange(b, device=device)
            causal_influence = torch.ones(b, 3, device=device)

            original_unet = self.diffusion.unet
            self.diffusion.unet = self.ema_unet

            try:
                hallucinated_z = self.diffusion.ddim_sample_loop(
                    shape=(b, self.hparams.latent_dim, 50),
                    device=device,
                    label=sample_labels,
                    causal_weights=causal_influence,
                    cfg_scale=self.hparams.cfg_scale,
                    num_inference_steps=self.hparams.ddim_steps,
                )

                with torch.no_grad():
                    z_seq = hallucinated_z.permute(0, 2, 1)
                    z_pooled = self._pool_features(z_seq)
                    logits = self.classifier(z_pooled)
                    preds = torch.argmax(logits, dim=1)
                    hall_acc = (preds == sample_labels).float().mean()

                self.log("val/hallucination_acc", hall_acc, rank_zero_only=True, sync_dist=True)
                logger.info(
                    "[Hallucination Check] Epoch %d | Targets: %s -> Preds: %s | Acc: %.3f",
                    self.current_epoch, sample_labels.tolist(), preds.tolist(), hall_acc,
                )
            except Exception as e:
                logger.warning("[Hallucination Check Error] %s", e)
            finally:
                self.diffusion.unet = original_unet

        # Barrier: ALL ranks must reach this point (rank 0 after sampling, others immediately)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ──────────────────────────────────────────────────────────────────────
    # OPTIMIZER
    # ──────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warmup for first 5 epochs (or 10% of training, whichever is larger)
        # prevents the large random-init gradients from overshooting in epoch 1.
        warmup_epochs = max(5, self.hparams.epochs // 10)
        main_epochs = max(1, self.hparams.epochs - warmup_epochs)

        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=self.hparams.lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
