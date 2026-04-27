"""Affect-Diff comprehensive ablation & baseline runner for Kaggle.

Self-contained script that trains one experiment at a time and appends
results to a JSON file so partial runs can be resumed.  All ablations
isolate a single claim; three classic baselines provide the external
reference point required for a conference submission.

Ablation families
-----------------
  arch     Core architectural components (causal graph, VAE, Perceiver, …)
  obj      Training objectives (KL warmup, free bits, label smoothing, loss weights)
  robust   Robustness probes run at eval time only (missing modality, noise, masking)
  data     Data-handling choices (NaN sanitization, sequence length)
  baseline External baseline models (TFN, MulT, MISA)

Usage (Kaggle cell)
-------------------
  !python train_ablation_kaggle.py --group arch
  !python train_ablation_kaggle.py --group obj
  !python train_ablation_kaggle.py --group baseline
  !python train_ablation_kaggle.py --name Full_Model          # single run
  !python train_ablation_kaggle.py --analyze                  # print results table
  !python train_ablation_kaggle.py --plot                     # save plots to /kaggle/working/plots/
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from modules.affect_diff_module import AffectDiffModule

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────

PT_PATH   = "/kaggle/input/datasets/ankit58/moesi-aligned/mosei_aligned_seq50_v2.pt"
CKPT_DIR  = "/kaggle/working/ablation_ckpts"
RESULTS_JSON = "/kaggle/working/ablation_results.json"
PLOTS_DIR = "/kaggle/working/plots"

CLASS_NAMES = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]


# ──────────────────────────────────────────────────────────────────────────────
# Full-model reference config (all components active)
# ──────────────────────────────────────────────────────────────────────────────

FULL_MODEL_CFG: Dict[str, Any] = {
    # Paths
    "pt_path":   PT_PATH,
    "ckpt_dir":  CKPT_DIR,
    # Data
    "batch_size":  64,
    "num_workers": 2,
    # Dims
    "text_dim":  300,
    "audio_dim": 74,
    "video_dim": 35,
    "hidden_dim": 128,
    "latent_dim": 128,
    "num_classes": 6,
    # Architecture
    "encoder_type":  "legacy",
    "encoder_layers": 2,
    "encoder_dropout": 0.3,
    "fusion_type": "concat",          # "crossmodal" = Perceiver; "concat" = simpler baseline
    "num_bottleneck_tokens": 20,
    "num_cross_attn_layers": 1,
    "num_self_attn_layers":  1,
    "dag_method": "notears",
    # Diffusion
    "diffusion_steps": 1000,
    "ddim_steps": 50,
    # Loss weights
    "beta_kl":        0.1,
    "lambda_diff":    0.1,
    "lambda_causal":  0.05,
    "lambda_recon":   0.5,
    "cfg_scale":      3.0,
    "ema_decay":      0.999,
    "label_smoothing": 0.1,
    "free_bits":       0.0,
    # Ablation toggles (all ON for full model)
    "use_reconstruction": False,   # keep False until recon is stable
    "use_diffusion":      False,   # keep False until diffusion is stable
    "use_causal_graph":   True,
    "use_augmentation":   True,
    "use_beta_tc_vae":    False,
    "use_focal_loss":     False,
    "focal_gamma":        2.0,
    # Fine-grained ablation flags (all ON for full model)
    "use_vae":                  True,
    "use_stop_gradient":        True,
    "use_cfg":                  True,
    "use_causal_diffusion_cond": True,
    "use_kl_warmup":            True,
    # Optimizer
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "epochs":       100,
    "patience":     25,
    # Trainer
    "precision": "16-mixed",
    "gradient_clip_val": 1.0,
    "devices": 1,
    "seed": 42,
    # Logging
    "experiment_name": "Full_Model",
    "use_wandb": False,
    "wandb_project": "Affect-Diff-CVPR",
}


# ──────────────────────────────────────────────────────────────────────────────
# Ablation catalogue
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── Reference ─────────────────────────────────────────────────────────────
    "Full_Model": {},

    # ── Core architectural ablations ──────────────────────────────────────────

    # 1. Replace causal graph with identity (no cross-modality weighting)
    "No_Causal_Graph": {"use_causal_graph": False},

    # 2. Keep graph structure but remove NOTEARS acyclicity constraint (→ L1 sparsity only)
    "No_NOTEARS": {"dag_method": "gumbel"},

    # 3. Replace Perceiver cross-modal bottleneck with simple concat+MLP
    "No_Perceiver": {"fusion_type": "concat"},

    # 4. Deterministic encoder: z = mu, no reparameterization, KL = 0
    "No_VAE": {"use_vae": False},

    # 5. Remove reconstruction decoder (L_recon = 0)
    "No_Reconstruction": {"use_reconstruction": False},

    # 6. Remove diffusion generative prior (L_diff = 0)
    "No_Diffusion": {"use_diffusion": False},

    # 7. Let diffusion gradients flow back into the encoder (remove stop-gradient)
    "No_Stop_Gradient": {
        "use_diffusion": True,
        "use_stop_gradient": False,
    },

    # 8. Disable classifier-free guidance (no null-token dropout during training)
    "No_CFG": {
        "use_diffusion": True,
        "use_cfg": False,
    },

    # 9. Remove causal conditioning from UNet (timestep + class only, no w)
    "No_Causal_Diffusion_Cond": {
        "use_diffusion": True,
        "use_causal_diffusion_cond": False,
    },

    # ── Training objective ablations ──────────────────────────────────────────

    # 10. Remove cyclical KL warmup (full beta_kl from epoch 0)
    "No_KL_Warmup": {"use_kl_warmup": False},

    # 11. Remove free bits (allow posterior collapse on all dims)
    "No_Free_Bits": {"free_bits": 0.0},

    # 12. Remove label smoothing
    "No_Label_Smoothing": {"label_smoothing": 0.0},

    # 13. Zero out individual loss terms
    "No_L_Causal": {"lambda_causal": 0.0},
    "No_L_KL":     {"beta_kl": 0.0},
    "No_L_Diff":   {"use_diffusion": False},   # same effect as lambda_diff=0
    "No_L_Recon":  {"use_reconstruction": False},

    # ── Hyperparameter sensitivity sweeps ─────────────────────────────────────

    "HP_beta_kl_0.01":  {"beta_kl": 0.01},
    "HP_beta_kl_1.0":   {"beta_kl": 1.0},
    "HP_lambda_diff_0.01": {"use_diffusion": True, "lambda_diff": 0.01},
    "HP_lambda_diff_1.0":  {"use_diffusion": True, "lambda_diff": 1.0},
    "HP_lambda_causal_0.01": {"lambda_causal": 0.01},
    "HP_lambda_causal_0.5":  {"lambda_causal": 0.5},
    "HP_label_smooth_0.0":   {"label_smoothing": 0.0},
    "HP_label_smooth_0.2":   {"label_smoothing": 0.2},
    "HP_free_bits_1.0":      {"free_bits": 1.0},
    "HP_free_bits_4.0":      {"free_bits": 4.0},

    # ── Ablation groups for convenience ───────────────────────────────────────

    # Everything off (sanity floor)
    "Classifier_Only": {
        "use_diffusion":      False,
        "use_causal_graph":   False,
        "use_reconstruction": False,
        "use_augmentation":   False,
        "use_vae":            False,
    },
}

# Group tags for --group filtering
ABLATION_GROUPS: Dict[str, List[str]] = {
    "arch": [
        "Full_Model", "No_Causal_Graph", "No_NOTEARS", "No_Perceiver",
        "No_VAE", "No_Reconstruction", "No_Diffusion", "No_Stop_Gradient",
        "No_CFG", "No_Causal_Diffusion_Cond", "Classifier_Only",
    ],
    "obj": [
        "Full_Model", "No_KL_Warmup", "No_Free_Bits", "No_Label_Smoothing",
        "No_L_Causal", "No_L_KL", "No_L_Diff", "No_L_Recon",
    ],
    "hp": [
        "Full_Model",
        "HP_beta_kl_0.01", "HP_beta_kl_1.0",
        "HP_lambda_diff_0.01", "HP_lambda_diff_1.0",
        "HP_lambda_causal_0.01", "HP_lambda_causal_0.5",
        "HP_label_smooth_0.0", "HP_label_smooth_0.2",
        "HP_free_bits_1.0", "HP_free_bits_4.0",
    ],
    "baseline": ["Baseline_TFN", "Baseline_MulT", "Baseline_MISA"],
    "all": list(ABLATION_CONFIGS.keys()) + ["Baseline_TFN", "Baseline_MulT", "Baseline_MISA"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataModule  (identical to train_prebuilt.py)
# ──────────────────────────────────────────────────────────────────────────────

class _MoseiDataset(Dataset):
    def __init__(self, data_dict: dict) -> None:
        self.vision = data_dict["vision"]
        self.audio  = data_dict["audio"]
        self.text   = data_dict["text"]
        self.labels = data_dict["labels"]

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "vision": self.vision[idx],
            "audio":  self.audio[idx],
            "text":   self.text[idx],
            "labels": self.labels[idx],
        }


class PrebuiltDataModule(pl.LightningDataModule):
    def __init__(self, pt_path: str, batch_size: int = 64, num_workers: int = 2) -> None:
        super().__init__()
        self.pt_path    = pt_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_weights: Optional[torch.Tensor] = None
        self.norm_stats: dict = {}

    def setup(self, stage: Optional[str] = None) -> None:
        data_dict = torch.load(self.pt_path, weights_only=False)
        total = data_dict["labels"].shape[0]
        logger.info("Loaded %d samples from %s", total, self.pt_path)

        rng = torch.Generator()
        rng.manual_seed(42)
        indices = torch.randperm(total, generator=rng).tolist()

        train_end = int(0.70 * total)
        val_end   = int(0.85 * total)
        train_idx = indices[:train_end]
        val_idx   = indices[train_end:val_end]
        test_idx  = indices[val_end:]

        logger.info("Split: train=%d  val=%d  test=%d",
                    len(train_idx), len(val_idx), len(test_idx))

        def _slice(idxs):
            t = torch.tensor(idxs, dtype=torch.long)
            return {k: v[t] for k, v in data_dict.items()}

        def _stats(tensor):
            mean = tensor.mean(dim=(0, 1), keepdim=True)
            std  = tensor.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            return mean, std

        train_raw = _slice(train_idx)
        t_mean, t_std = _stats(train_raw["text"])
        a_mean, a_std = _stats(train_raw["audio"])
        v_mean, v_std = _stats(train_raw["vision"])

        self.norm_stats = {
            "text":   (t_mean, t_std),
            "audio":  (a_mean, a_std),
            "vision": (v_mean, v_std),
        }

        def _norm(d):
            d = {k: v.clone() for k, v in d.items()}
            for key, mean, std in [("text", t_mean, t_std),
                                    ("audio", a_mean, a_std),
                                    ("vision", v_mean, v_std)]:
                d[key] = torch.clamp(
                    torch.nan_to_num((d[key] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
                    min=-10.0, max=10.0,
                )
            return d

        self.train_dataset = _MoseiDataset(_norm(_slice(train_idx)))
        self.val_dataset   = _MoseiDataset(_norm(_slice(val_idx)))
        self.test_dataset  = _MoseiDataset(_norm(_slice(test_idx)))

        labels = self.train_dataset.labels
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=num_classes).float()
        sqrt_inv = 1.0 / counts.clamp(min=1).sqrt()
        self.sample_weights = sqrt_inv[labels]
        self.class_weights  = (sqrt_inv / sqrt_inv.min()).float()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline models
# ──────────────────────────────────────────────────────────────────────────────

class _BaselineBase(pl.LightningModule):
    """Shared plumbing for all three baselines."""

    def __init__(self, lr: float = 5e-4, weight_decay: float = 1e-4,
                 epochs: int = 100, num_classes: int = 6,
                 label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.save_hyperparameters()
        try:
            from torchmetrics.classification import MulticlassAccuracy
            self._bal_val  = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._bal_test = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._use_bal  = True
        except ImportError:
            self._use_bal = False
        self._test_logits: list = []
        self._test_targets: list = []

    def _step(self, batch, stage):
        logits = self(batch["text"], batch["audio"], batch["vision"])
        labels = batch["labels"].long()
        loss   = F.cross_entropy(logits.float(), labels,
                                 label_smoothing=self.hparams.label_smoothing)
        preds  = torch.argmax(logits, dim=1)
        acc    = (preds == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True,  sync_dist=True)
        if self._use_bal and stage in ("val", "test"):
            met = self._bal_val if stage == "val" else self._bal_test
            bal = met(preds, labels)
            self.log(f"{stage}_bal_acc", bal, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_test_epoch_start(self):
        self._test_logits  = []
        self._test_targets = []

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, "test")
        with torch.no_grad():
            logits = self(batch["text"], batch["audio"], batch["vision"])
            self._test_logits.append(logits.cpu())
            self._test_targets.append(batch["labels"].cpu())
        return loss

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        all_logits  = torch.cat(self._test_logits,  dim=0)
        all_targets = torch.cat(self._test_targets, dim=0)
        preds = torch.argmax(all_logits, dim=1)
        num_classes = self.hparams.num_classes
        try:
            from torchmetrics.classification import (
                MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
                MulticlassAUROC, MulticlassConfusionMatrix,
            )
            f1_mac  = MulticlassF1Score(num_classes=num_classes, average="macro")(preds, all_targets).item()
            f1_per  = MulticlassF1Score(num_classes=num_classes, average="none")(preds, all_targets).tolist()
            prec    = MulticlassPrecision(num_classes=num_classes, average="none")(preds, all_targets).tolist()
            rec     = MulticlassRecall(num_classes=num_classes, average="none")(preds, all_targets).tolist()
            auroc   = MulticlassAUROC(num_classes=num_classes)(all_logits, all_targets.long()).item()
            cm      = MulticlassConfusionMatrix(num_classes=num_classes)(preds, all_targets).tolist()
        except Exception:
            f1_mac, f1_per, prec, rec, auroc, cm = 0.0, [], [], [], 0.0, []
        bal_acc = sum(rec) / len(rec) if rec else 0.0
        names   = CLASS_NAMES[:num_classes]
        self._rich_test_metrics: Dict[str, Any] = {
            "macro_f1":         f1_mac,
            "balanced_acc":     bal_acc,
            "auroc":            auroc,
            "per_class_f1":     dict(zip(names, f1_per)),
            "per_class_prec":   dict(zip(names, prec)),
            "per_class_rec":    dict(zip(names, rec)),
            "confusion_matrix": cm,
        }
        self.log("test_macro_f1",    f1_mac,  sync_dist=True)
        self.log("test_balanced_acc", bal_acc, sync_dist=True)
        self.log("test_auroc",        auroc,   sync_dist=True)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr,
                    weight_decay=self.hparams.weight_decay)
        warmup  = max(5, self.hparams.epochs // 10)
        main_ep = max(1, self.hparams.epochs - warmup)
        sched   = SequentialLR(opt, schedulers=[
            LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(opt, T_max=main_ep, eta_min=self.hparams.lr * 0.01),
        ], milestones=[warmup])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── TFN ───────────────────────────────────────────────────────────────────────

class TFNModule(_BaselineBase):
    """Tensor Fusion Network (Zadeh et al., EMNLP 2017).

    Each modality is projected to proj_dim, then the three (proj_dim+1)-vectors
    (with appended 1 for bias) are combined via a 3-way outer product.
    The resulting (proj_dim+1)^3 tensor is flattened and classified.

    Reference: https://arxiv.org/abs/1707.07250
    """

    def __init__(
        self,
        text_dim: int = 300, audio_dim: int = 74, video_dim: int = 35,
        num_classes: int = 6, proj_dim: int = 16, hidden_dim: int = 256,
        dropout: float = 0.3, lr: float = 5e-4, weight_decay: float = 1e-4,
        epochs: int = 100, label_smoothing: float = 0.1,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, epochs=epochs,
                         num_classes=num_classes, label_smoothing=label_smoothing)
        self.save_hyperparameters()
        self.text_enc  = nn.Sequential(
            nn.Linear(text_dim,  proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.video_enc = nn.Sequential(
            nn.Linear(video_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))

        tensor_dim = (proj_dim + 1) ** 3  # e.g. 17^3 = 4913
        self.classifier = nn.Sequential(
            nn.Linear(tensor_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor):
        # Mean-pool temporal dim: (B, L, D) → (B, D)
        t = self.text_enc(text.mean(dim=1))    # (B, proj_dim)
        a = self.audio_enc(audio.mean(dim=1))
        v = self.video_enc(video.mean(dim=1))

        # Append 1 for bias term: (B, proj_dim+1)
        ones = torch.ones(t.shape[0], 1, device=t.device)
        t = torch.cat([t, ones], dim=1)
        a = torch.cat([a, ones], dim=1)
        v = torch.cat([v, ones], dim=1)

        # 3-way outer product via einsum: (B, p+1, p+1, p+1)
        fused = torch.einsum("bi,bj,bk->bijk", t, a, v)
        fused = fused.reshape(fused.shape[0], -1)   # (B, (p+1)^3)
        return self.classifier(fused)


# ── MulT ──────────────────────────────────────────────────────────────────────

class _CrossModalAttn(nn.Module):
    """Single-layer cross-modal attention: source modality attends to target."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_proj  = nn.Linear(q_dim,  q_dim)
        self.kv_proj = nn.Linear(kv_dim, q_dim)
        self.attn = nn.MultiheadAttention(q_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(q_dim)

    def forward(self, q_seq: torch.Tensor, kv_seq: torch.Tensor) -> torch.Tensor:
        """q_seq: (B, Lq, D)   kv_seq: (B, Lkv, D)  →  (B, Lq, D)"""
        Q  = self.q_proj(q_seq)
        KV = self.kv_proj(kv_seq)
        out, _ = self.attn(Q, KV, KV)
        return self.norm(q_seq + out)


class MulTModule(_BaselineBase):
    """Multimodal Transformer (Tsai et al., ACL 2019).

    Six directional cross-modal attention modules (T→A, T→V, A→T, A→V, V→T, V→A)
    followed by a self-attention refinement and mean-pool classifier.

    Reference: https://arxiv.org/abs/1906.00295
    """

    def __init__(
        self,
        text_dim: int = 300, audio_dim: int = 74, video_dim: int = 35,
        num_classes: int = 6, hidden_dim: int = 128, n_heads: int = 4,
        dropout: float = 0.3, lr: float = 5e-4, weight_decay: float = 1e-4,
        epochs: int = 100, label_smoothing: float = 0.1,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, epochs=epochs,
                         num_classes=num_classes, label_smoothing=label_smoothing)
        self.save_hyperparameters()

        self.t_proj = nn.Linear(text_dim,  hidden_dim)
        self.a_proj = nn.Linear(audio_dim, hidden_dim)
        self.v_proj = nn.Linear(video_dim, hidden_dim)

        # 6 directional attentions
        self.ta = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # T queried by A
        self.tv = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # T queried by V
        self.at = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # A queried by T
        self.av = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # A queried by V
        self.vt = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # V queried by T
        self.va = _CrossModalAttn(hidden_dim, hidden_dim, n_heads, dropout)  # V queried by A

        # Self-attn refinement on concatenated cross-modal outputs
        merged_dim = hidden_dim * 6
        self.self_attn = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor):
        T = self.t_proj(text)   # (B, L, H)
        A = self.a_proj(audio)
        V = self.v_proj(video)

        # All six directional attentions
        TA = self.ta(A, T)   # Audio queries Text
        TV = self.tv(V, T)   # Video  queries Text
        AT = self.at(T, A)
        AV = self.av(V, A)
        VT = self.vt(T, V)
        VA = self.va(A, V)

        # Concat cross-modal features: (B, L, 6H)
        merged = torch.cat([TA, TV, AT, AV, VT, VA], dim=-1)

        # Project + mean-pool
        h = self.self_attn(merged).mean(dim=1)  # (B, H)
        return self.classifier(h)


# ── MISA ──────────────────────────────────────────────────────────────────────

def _cmd_loss(h1: torch.Tensor, h2: torch.Tensor, n_moments: int = 5) -> torch.Tensor:
    """Central Moment Discrepancy (Li et al., 2017).

    Matches the first n_moments of the distributions of h1 and h2.
    Used to align the shared subspace representations across modalities.
    """
    loss = F.mse_loss(h1.mean(0), h2.mean(0))
    for p in range(2, n_moments + 1):
        loss = loss + F.mse_loss(
            (h1 - h1.mean(0)).pow(p).mean(0),
            (h2 - h2.mean(0)).pow(p).mean(0),
        )
    return loss


def _orth_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularization: shared ⊥ private."""
    dot = (shared * private).sum(dim=1)  # (B,)
    return dot.pow(2).mean()


class MISAModule(_BaselineBase):
    """MISA — Modality-Invariant and -Specific Representations (Hazarika et al., ACL 2020).

    Each modality is encoded into a shared (modality-invariant) subspace
    and a private (modality-specific) subspace.  CMD aligns the shared reps
    across modalities; an orthogonality penalty keeps the subspaces disentangled.
    Classification uses all six subspace vectors.

    Reference: https://arxiv.org/abs/2005.03545
    """

    def __init__(
        self,
        text_dim: int = 300, audio_dim: int = 74, video_dim: int = 35,
        num_classes: int = 6, hidden_dim: int = 128, sub_dim: int = 64,
        dropout: float = 0.3, lambda_cmd: float = 0.1, lambda_orth: float = 0.1,
        lr: float = 5e-4, weight_decay: float = 1e-4,
        epochs: int = 100, label_smoothing: float = 0.1,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, epochs=epochs,
                         num_classes=num_classes, label_smoothing=label_smoothing)
        self.save_hyperparameters()

        # Shared unimodal encoders (temporal LSTM-like with 1D conv)
        def _enc(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            )

        self.t_enc = _enc(text_dim)
        self.a_enc = _enc(audio_dim)
        self.v_enc = _enc(video_dim)

        # Shared (invariant) heads
        self.t_shared = nn.Linear(hidden_dim, sub_dim)
        self.a_shared = nn.Linear(hidden_dim, sub_dim)
        self.v_shared = nn.Linear(hidden_dim, sub_dim)

        # Private (specific) heads
        self.t_priv = nn.Linear(hidden_dim, sub_dim)
        self.a_priv = nn.Linear(hidden_dim, sub_dim)
        self.v_priv = nn.Linear(hidden_dim, sub_dim)

        # Fusion: 3 shared + 3 private → classifier
        self.classifier = nn.Sequential(
            nn.Linear(sub_dim * 6, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_modalities(self, text, audio, video):
        """Returns (shared, private) for each modality, mean-pooled."""
        ht = self.t_enc(text).mean(1)    # (B, H)
        ha = self.a_enc(audio).mean(1)
        hv = self.v_enc(video).mean(1)

        ts = F.tanh(self.t_shared(ht))
        as_ = F.tanh(self.a_shared(ha))
        vs = F.tanh(self.v_shared(hv))

        tp = F.tanh(self.t_priv(ht))
        ap = F.tanh(self.a_priv(ha))
        vp = F.tanh(self.v_priv(hv))
        return (ts, as_, vs), (tp, ap, vp)

    def forward(self, text, audio, video):
        (ts, as_, vs), (tp, ap, vp) = self._encode_modalities(text, audio, video)
        fused = torch.cat([ts, as_, vs, tp, ap, vp], dim=-1)
        return self.classifier(fused)

    def _step(self, batch, stage):
        text   = batch["text"]
        audio  = batch["audio"]
        video  = batch["vision"]
        labels = batch["labels"].long()

        (ts, as_, vs), (tp, ap, vp) = self._encode_modalities(text, audio, video)
        fused  = torch.cat([ts, as_, vs, tp, ap, vp], dim=-1)
        logits = self.classifier(fused)

        loss_ce = F.cross_entropy(logits.float(), labels,
                                  label_smoothing=self.hparams.label_smoothing)

        # CMD: align shared representations across modalities
        loss_cmd = (
            _cmd_loss(ts, as_) + _cmd_loss(ts, vs) + _cmd_loss(as_, vs)
        ) / 3.0

        # Orthogonality: shared ⊥ private for each modality
        loss_orth = (
            _orth_loss(ts, tp) + _orth_loss(as_, ap) + _orth_loss(vs, vp)
        ) / 3.0

        loss = (loss_ce
                + self.hparams.lambda_cmd  * loss_cmd
                + self.hparams.lambda_orth * loss_orth)

        preds = torch.argmax(logits, dim=1)
        acc   = (preds == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True,  sync_dist=True)
        self.log(f"{stage}_cmd",  loss_cmd, sync_dist=True)
        if self._use_bal and stage in ("val", "test"):
            met = self._bal_val if stage == "val" else self._bal_test
            bal = met(preds, labels)
            self.log(f"{stage}_bal_acc", bal, on_step=False, on_epoch=True, sync_dist=True)
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Rich metrics utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_rich_metrics(model: pl.LightningModule) -> Dict[str, Any]:
    """Extract rich test metrics collected during on_test_epoch_end."""
    return getattr(model, "_rich_test_metrics", {})


def efficiency_stats(model: nn.Module, sample_batch: Dict[str, torch.Tensor],
                     device: str = "cpu") -> Dict[str, Any]:
    """Compute parameter count and inference latency."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = model.to(device).eval()
    text  = sample_batch["text"].to(device)
    audio = sample_batch["audio"].to(device)
    video = sample_batch["vision"].to(device)

    # Warmup
    with torch.no_grad():
        try:
            for _ in range(3):
                _ = model(text, audio, video)
        except Exception:
            pass

    # Timed runs
    t0 = time.perf_counter()
    N  = 10
    with torch.no_grad():
        try:
            for _ in range(N):
                _ = model(text, audio, video)
        except Exception:
            pass
    latency_ms = (time.perf_counter() - t0) / N * 1000

    return {
        "total_params":    total_params,
        "trainable_params": trainable,
        "latency_ms":      round(latency_ms, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Robustness evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_loader(model: pl.LightningModule, loader: DataLoader,
                 device: str, noise_std: float = 0.0,
                 mask_modality: Optional[str] = None,
                 mask_frames_frac: float = 0.0) -> Dict[str, float]:
    model.eval()
    preds_all, labels_all, logits_all = [], [], []
    for batch in loader:
        text  = torch.nan_to_num(batch["text"]).to(device)
        audio = torch.nan_to_num(batch["audio"]).to(device)
        video = torch.nan_to_num(batch["vision"]).to(device)
        labs  = batch["labels"].to(device)

        if mask_modality == "text":
            text  = torch.zeros_like(text)
        elif mask_modality == "audio":
            audio = torch.zeros_like(audio)
        elif mask_modality == "vision":
            video = torch.zeros_like(video)

        if noise_std > 0.0:
            text  = text  + torch.randn_like(text)  * noise_std
            audio = audio + torch.randn_like(audio) * noise_std
            video = video + torch.randn_like(video) * noise_std

        if mask_frames_frac > 0.0:
            L = text.shape[1]
            n_mask = max(1, int(L * mask_frames_frac))
            idx    = torch.randperm(L)[:n_mask]
            text[:, idx]  = 0.0
            audio[:, idx] = 0.0
            video[:, idx] = 0.0

        try:
            logits = model(text, audio, video)
        except Exception:
            continue

        preds_all.append(torch.argmax(logits, 1).cpu())
        labels_all.append(labs.cpu())
        logits_all.append(logits.cpu())

    if not preds_all:
        return {}

    preds  = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    acc    = (preds == labels).float().mean().item()

    try:
        from torchmetrics.classification import MulticlassF1Score
        f1 = MulticlassF1Score(num_classes=6, average="macro")(preds, labels).item()
    except Exception:
        f1 = 0.0

    return {"acc": acc, "macro_f1": f1}


def eval_robustness(model: pl.LightningModule, datamodule: PrebuiltDataModule,
                    device: str = "cuda") -> Dict[str, Any]:
    """Run all robustness probes and return results dict."""
    loader = datamodule.test_dataloader()
    results: Dict[str, Any] = {}

    # Missing modality
    for mod in ("text", "audio", "vision"):
        r = _eval_loader(model, loader, device, mask_modality=mod)
        results[f"missing_{mod}"] = r

    # Gaussian noise sweep
    for std in (0.1, 0.25, 0.5, 1.0, 2.0):
        r = _eval_loader(model, loader, device, noise_std=std)
        results[f"noise_{std}"] = r

    # Temporal frame masking sweep
    for frac in (0.1, 0.25, 0.5, 0.75):
        r = _eval_loader(model, loader, device, mask_frames_frac=frac)
        results[f"frame_mask_{frac}"] = r

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Results persistence
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path: str = RESULTS_JSON) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_result(name: str, result: Dict[str, Any], path: str = RESULTS_JSON) -> None:
    data = load_results(path)
    data[name] = result
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved result for %s → %s", name, path)


# ──────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def _build_affect_diff_model(cfg: Dict[str, Any]) -> AffectDiffModule:
    return AffectDiffModule(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        video_dim=cfg["video_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_classes=cfg["num_classes"],
        encoder_type=cfg["encoder_type"],
        encoder_dropout=cfg.get("encoder_dropout", 0.3),
        encoder_layers=cfg.get("encoder_layers", 2),
        text_backbone="roberta-base",
        audio_backbone="facebook/hubert-base-ls960",
        video_backbone="openai/clip-vit-base-patch16",
        freeze_backbones=True,
        fusion_type=cfg["fusion_type"],
        num_bottleneck_tokens=cfg["num_bottleneck_tokens"],
        num_cross_attn_layers=cfg["num_cross_attn_layers"],
        num_self_attn_layers=cfg["num_self_attn_layers"],
        dag_method=cfg["dag_method"],
        diffusion_steps=cfg["diffusion_steps"],
        ddim_steps=cfg["ddim_steps"],
        beta_kl=cfg["beta_kl"],
        lambda_diff=cfg["lambda_diff"],
        lambda_causal=cfg["lambda_causal"],
        lambda_recon=cfg["lambda_recon"],
        cfg_scale=cfg["cfg_scale"],
        ema_decay=cfg["ema_decay"],
        label_smoothing=cfg["label_smoothing"],
        free_bits=cfg["free_bits"],
        use_reconstruction=cfg["use_reconstruction"],
        use_diffusion=cfg["use_diffusion"],
        use_causal_graph=cfg["use_causal_graph"],
        use_augmentation=cfg["use_augmentation"],
        use_beta_tc_vae=cfg["use_beta_tc_vae"],
        use_focal_loss=cfg.get("use_focal_loss", False),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        use_vae=cfg.get("use_vae", True),
        use_stop_gradient=cfg.get("use_stop_gradient", True),
        use_cfg=cfg.get("use_cfg", True),
        use_causal_diffusion_cond=cfg.get("use_causal_diffusion_cond", True),
        use_kl_warmup=cfg.get("use_kl_warmup", True),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
    )


def _build_baseline_model(name: str, cfg: Dict[str, Any]) -> _BaselineBase:
    shared = dict(
        text_dim=cfg["text_dim"], audio_dim=cfg["audio_dim"], video_dim=cfg["video_dim"],
        num_classes=cfg["num_classes"], hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("encoder_dropout", 0.3),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"], label_smoothing=cfg["label_smoothing"],
    )
    if name == "Baseline_TFN":
        return TFNModule(**{**shared, "proj_dim": 16})
    elif name == "Baseline_MulT":
        return MulTModule(**shared)
    elif name == "Baseline_MISA":
        return MISAModule(**shared)
    raise ValueError(f"Unknown baseline: {name}")


def run_experiment(
    name: str,
    cfg: Dict[str, Any],
    run_robustness: bool = True,
    run_efficiency: bool = True,
    skip_if_done: bool = True,
) -> Dict[str, Any]:
    """Train + test one experiment; return full results dict."""

    if skip_if_done:
        existing = load_results()
        if name in existing:
            logger.info("Skipping %s — already in results JSON", name)
            return existing[name]

    pl.seed_everything(cfg["seed"], workers=True)
    exp_name = cfg["experiment_name"]
    logger.info("=== Starting: %s ===", exp_name)

    datamodule = PrebuiltDataModule(
        pt_path=cfg["pt_path"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    # setup to compute norm_stats for robustness eval
    datamodule.setup()

    is_baseline = name.startswith("Baseline_")
    if is_baseline:
        model = _build_baseline_model(name, cfg)
    else:
        model = _build_affect_diff_model(cfg)

    # Logger
    if cfg.get("use_wandb"):
        try:
            from pytorch_lightning.loggers import WandbLogger
            pl_logger = WandbLogger(project=cfg["wandb_project"], name=exp_name, config=cfg)
        except Exception as e:
            logger.warning("WandbLogger failed (%s) — CSV fallback", e)
            cfg["use_wandb"] = False

    if not cfg.get("use_wandb"):
        from pytorch_lightning.loggers import CSVLogger
        pl_logger = CSVLogger(
            save_dir=os.path.join(cfg["ckpt_dir"], "csv_logs"),
            name=exp_name,
        )

    ckpt_dir = os.path.join(cfg["ckpt_dir"], exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_acc", mode="max", save_top_k=1,
            filename="best-{epoch:02d}-{val_acc:.3f}",
        ),
        EarlyStopping(monitor="val_acc", patience=cfg["patience"], mode="max"),
        LearningRateMonitor(logging_interval="step"),
    ]

    t_train_start = time.time()
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg["devices"],
        precision=cfg["precision"],
        gradient_clip_val=cfg["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        logger=pl_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    train_time_min = (time.time() - t_train_start) / 60.0

    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt or None)
    test_metrics = test_results[0] if test_results else {}

    # Rich metrics from on_test_epoch_end
    rich = compute_rich_metrics(model)

    # Robustness probes
    robustness = {}
    if run_robustness:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Reload best checkpoint for robustness eval
            if best_ckpt:
                state = torch.load(best_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["state_dict"])
            model = model.to(device)
            robustness = eval_robustness(model, datamodule, device=device)
        except Exception as e:
            logger.warning("Robustness eval failed: %s", e)

    # Efficiency stats
    eff = {}
    if run_efficiency:
        try:
            sample = next(iter(datamodule.test_dataloader()))
            eff = efficiency_stats(model.cpu(), sample, device="cpu")
        except Exception as e:
            logger.warning("Efficiency stats failed: %s", e)

    result = {
        "experiment_name":  exp_name,
        "config":           {k: v for k, v in cfg.items() if k not in ("pt_path",)},
        "test_metrics":     test_metrics,
        "rich_metrics":     rich,
        "robustness":       robustness,
        "efficiency":       eff,
        "train_time_min":   round(train_time_min, 1),
        "best_ckpt":        best_ckpt,
    }

    save_result(name, result)
    logger.info("=== Done: %s | test_acc=%.4f | macro_f1=%.4f ===",
                exp_name,
                test_metrics.get("test_acc", float("nan")),
                rich.get("macro_f1", float("nan")))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Sweep runner
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep(names: List[str], base_cfg: Dict[str, Any],
              run_robustness: bool = True, skip_if_done: bool = True) -> None:
    """Run a list of ablation experiments sequentially and print a running table."""
    summary = []

    for name in names:
        overrides = ABLATION_CONFIGS.get(name, {})
        cfg = {**base_cfg, **overrides, "experiment_name": name}

        try:
            result  = run_experiment(name, cfg,
                                     run_robustness=run_robustness,
                                     skip_if_done=skip_if_done)
            acc     = result["test_metrics"].get("test_acc", float("nan"))
            f1      = result["rich_metrics"].get("macro_f1", float("nan"))
            bal_acc = result["rich_metrics"].get("balanced_acc", float("nan"))
            status  = "ok"
        except Exception as e:
            logger.error("Experiment %s crashed: %s", name, e)
            acc = bal_acc = f1 = float("nan")
            status = "CRASH"

        summary.append((name, acc, bal_acc, f1, status))
        _print_table(summary)


def _print_table(rows: List[Tuple]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Experiment':<35} {'Acc':>7} {'BalAcc':>8} {'MacroF1':>9} {'Status':>7}")
    print("-" * 72)
    for name, acc, bal, f1, status in rows:
        def _fmt(v):
            return f"{v:.4f}" if (v == v and v != float("inf")) else "  NaN "
        print(f"{name:<35} {_fmt(acc):>7} {_fmt(bal):>8} {_fmt(f1):>9} {status:>7}")
    print("=" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ──────────────────────────────────────────────────────────────────────────────

def print_analysis(results_path: str = RESULTS_JSON) -> None:
    data = load_results(results_path)
    if not data:
        print("No results found at", results_path)
        return

    rows = []
    for name, r in data.items():
        tm = r.get("test_metrics", {})
        rm = r.get("rich_metrics", {})
        eff = r.get("efficiency", {})
        rows.append({
            "name":       name,
            "acc":        tm.get("test_acc",        float("nan")),
            "bal_acc":    rm.get("balanced_acc",    float("nan")),
            "macro_f1":   rm.get("macro_f1",        float("nan")),
            "auroc":      rm.get("auroc",           float("nan")),
            "params_M":   round(eff.get("total_params", 0) / 1e6, 2),
            "latency_ms": eff.get("latency_ms",    float("nan")),
            "train_min":  r.get("train_time_min",  float("nan")),
        })

    # Sort by macro F1
    rows.sort(key=lambda x: x["macro_f1"] if x["macro_f1"] == x["macro_f1"] else -1, reverse=True)

    hdr = f"{'Experiment':<35} {'Acc':>7} {'BalAcc':>8} {'F1':>7} {'AUROC':>7} {'Params(M)':>10} {'Lat(ms)':>8}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _f(v):
            return f"{v:.4f}" if v == v else "  NaN "
        print(f"{r['name']:<35} {_f(r['acc']):>7} {_f(r['bal_acc']):>8} "
              f"{_f(r['macro_f1']):>7} {_f(r['auroc']):>7} "
              f"{r['params_M']:>10.2f} {_f(r['latency_ms']):>8}")
    print("=" * len(hdr))

    # Per-class F1 table
    print("\n── Per-class F1 ──")
    hdr2 = f"{'Experiment':<35} " + " ".join(f"{c[:4]:>7}" for c in CLASS_NAMES)
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        data_r = data[r["name"]]
        pcf = data_r.get("rich_metrics", {}).get("per_class_f1", {})
        vals = " ".join(f"{pcf.get(c, float('nan')):>7.4f}" for c in CLASS_NAMES)
        print(f"{r['name']:<35} {vals}")


def generate_plots(results_path: str = RESULTS_JSON, out_dir: str = PLOTS_DIR) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import seaborn as sns  # type: ignore
    except ImportError:
        print("matplotlib/seaborn not available — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    data = load_results(results_path)
    if not data:
        print("No results to plot.")
        return

    names  = list(data.keys())
    accs   = [data[n].get("test_metrics", {}).get("test_acc", float("nan")) for n in names]
    f1s    = [data[n].get("rich_metrics", {}).get("macro_f1", float("nan")) for n in names]
    bals   = [data[n].get("rich_metrics", {}).get("balanced_acc", float("nan")) for n in names]

    valid = [(n, a, f, b) for n, a, f, b in zip(names, accs, f1s, bals) if f == f]
    if not valid:
        print("No valid (non-NaN) results to plot.")
        return
    valid.sort(key=lambda x: x[2], reverse=True)
    v_names, v_accs, v_f1s, v_bals = zip(*valid)

    # ── Figure 1: Ablation bar chart (Acc, BalAcc, MacroF1) ───────────────
    fig, ax = plt.subplots(figsize=(max(10, len(v_names) * 0.7), 5))
    x = np.arange(len(v_names))
    w = 0.28
    ax.bar(x - w, v_accs, w, label="Accuracy")
    ax.bar(x,     v_bals, w, label="Balanced Acc")
    ax.bar(x + w, v_f1s,  w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(v_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Main Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_main_metrics.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Per-class F1 heatmap ────────────────────────────────────
    pcf_matrix = []
    pcf_names  = []
    for n in v_names:
        pcf = data[n].get("rich_metrics", {}).get("per_class_f1", {})
        row = [pcf.get(c, float("nan")) for c in CLASS_NAMES]
        if any(v == v for v in row):
            pcf_matrix.append(row)
            pcf_names.append(n)

    if pcf_matrix:
        mat = np.array(pcf_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(10, max(4, len(pcf_names) * 0.35)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(len(pcf_names)))
        ax.set_yticklabels(pcf_names, fontsize=8)
        ax.set_title("Per-class F1 Heatmap")
        plt.colorbar(im, ax=ax)
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] == mat[i, j]:
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_class_f1_heatmap.png"), dpi=150)
        plt.close(fig)

    # ── Figure 3: Robustness curves (noise) for Full_Model ────────────────
    if "Full_Model" in data:
        rob = data["Full_Model"].get("robustness", {})
        noise_stds = [0.1, 0.25, 0.5, 1.0, 2.0]
        noise_f1s  = [rob.get(f"noise_{s}", {}).get("macro_f1", float("nan"))
                      for s in noise_stds]

        if any(v == v for v in noise_f1s):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            ax = axes[0]
            ax.plot(noise_stds, noise_f1s, marker="o")
            ax.set_xlabel("Gaussian noise σ")
            ax.set_ylabel("Macro F1")
            ax.set_title("Robustness to Gaussian Noise")
            ax.grid(alpha=0.3)

            ax = axes[1]
            mods = ["text", "audio", "vision"]
            miss_f1s = [rob.get(f"missing_{m}", {}).get("macro_f1", float("nan"))
                        for m in mods]
            clean_f1 = data["Full_Model"].get("rich_metrics", {}).get("macro_f1", 0)
            bars = ax.bar(mods, miss_f1s, color=["steelblue", "orange", "green"])
            ax.axhline(clean_f1, ls="--", color="red", label=f"Clean F1={clean_f1:.3f}")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Macro F1")
            ax.set_title("Missing-Modality Robustness")
            ax.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "robustness_curves.png"), dpi=150)
            plt.close(fig)

    # ── Figure 4: Confusion matrix for Full_Model ─────────────────────────
    if "Full_Model" in data:
        cm_data = data["Full_Model"].get("rich_metrics", {}).get("confusion_matrix", [])
        if cm_data:
            cm_arr = np.array(cm_data, dtype=float)
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_norm  = cm_arr / (row_sums + 1e-8)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (Full Model, normalized)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
            plt.close(fig)

    # ── Figure 5: Efficiency scatter (params vs F1) ────────────────────────
    param_ms = [data[n].get("efficiency", {}).get("total_params", 0) / 1e6 for n in v_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(param_ms, list(v_f1s), c=list(v_accs), cmap="viridis", s=80)
    for n, x_, y_ in zip(v_names, param_ms, v_f1s):
        ax.annotate(n, (x_, y_), fontsize=6, ha="left", va="bottom")
    plt.colorbar(sc, ax=ax, label="Accuracy")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Efficiency–Performance Trade-off")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "efficiency_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Affect-Diff comprehensive ablation & baseline runner for Kaggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Run mode
    p.add_argument("--name",     default=None,
                   help="Run a single experiment by name")
    p.add_argument("--group",    default=None,
                   choices=list(ABLATION_GROUPS.keys()),
                   help="Run a predefined group of experiments")
    p.add_argument("--list",     action="store_true",
                   help="List all available experiment names and exit")
    p.add_argument("--analyze",  action="store_true",
                   help="Print results table from saved JSON")
    p.add_argument("--plot",     action="store_true",
                   help="Generate all plots from saved JSON")
    p.add_argument("--no_skip",  action="store_true",
                   help="Re-run even if result already in JSON")
    p.add_argument("--no_robust", action="store_true",
                   help="Skip robustness evaluation (faster)")

    # Path overrides
    p.add_argument("--pt_path",  default=PT_PATH)
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--results",  default=RESULTS_JSON)
    p.add_argument("--plots_dir", default=PLOTS_DIR)

    # Common training overrides
    p.add_argument("--epochs",     type=int,   default=FULL_MODEL_CFG["epochs"])
    p.add_argument("--patience",   type=int,   default=FULL_MODEL_CFG["patience"])
    p.add_argument("--batch_size", type=int,   default=FULL_MODEL_CFG["batch_size"])
    p.add_argument("--lr",         type=float, default=FULL_MODEL_CFG["lr"])
    p.add_argument("--seed",       type=int,   default=FULL_MODEL_CFG["seed"])
    p.add_argument("--precision",  default=FULL_MODEL_CFG["precision"],
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--use_wandb",  action="store_true")
    p.add_argument("--wandb_project", default=FULL_MODEL_CFG["wandb_project"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    global RESULTS_JSON, PLOTS_DIR
    RESULTS_JSON = args.results
    PLOTS_DIR    = args.plots_dir

    if args.list:
        all_names = list(ABLATION_CONFIGS.keys()) + ["Baseline_TFN", "Baseline_MulT", "Baseline_MISA"]
        print("\nAvailable experiments:")
        for n in all_names:
            grp = [g for g, ns in ABLATION_GROUPS.items() if n in ns and g != "all"]
            print(f"  {n:<40} groups: {', '.join(grp)}")
        print(f"\nGroups: {', '.join(k for k in ABLATION_GROUPS if k != 'all')}")
        return

    if args.analyze:
        print_analysis(args.results)
        return

    if args.plot:
        generate_plots(args.results, args.plots_dir)
        return

    # Build base config from FULL_MODEL_CFG + CLI overrides
    base_cfg = {**FULL_MODEL_CFG}
    base_cfg.update({
        "pt_path":       args.pt_path,
        "ckpt_dir":      args.ckpt_dir,
        "epochs":        args.epochs,
        "patience":      args.patience,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "seed":          args.seed,
        "precision":     args.precision,
        "use_wandb":     args.use_wandb,
        "wandb_project": args.wandb_project,
        "gradient_clip_val": FULL_MODEL_CFG["gradient_clip_val"],
        "devices": 1,
    })

    skip   = not args.no_skip
    robust = not args.no_robust

    if args.name:
        names = [args.name]
    elif args.group:
        names = ABLATION_GROUPS[args.group]
    else:
        print("Specify --name <exp>, --group <group>, --analyze, or --plot.")
        print("Run with --list to see all available experiments.")
        return

    if len(names) == 1 and names[0] not in ABLATION_CONFIGS and names[0].startswith("Baseline_"):
        # Single baseline run
        cfg = {**base_cfg, "experiment_name": names[0]}
        run_experiment(names[0], cfg, run_robustness=robust, skip_if_done=skip)
    else:
        run_sweep(names, base_cfg, run_robustness=robust, skip_if_done=skip)

    print_analysis(args.results)


if __name__ == "__main__":
    main()
