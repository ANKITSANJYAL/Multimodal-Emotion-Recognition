"""Affect-Diff on CMU-MOSEI Sentiment Analysis (Kaggle runner).

Demonstrates architecture generalizability: same Causal-Diffusion Bridge,
BERT text features (768-dim), standard 22K split, 7-class and binary sentiment.

Data setup (download once to Kaggle, then point the paths below):
  BERT:    https://drive.google.com/file/d/13y2xoO1YlDrJ4Be2X6kjtMzfRBs7tBRg
  COVAREP: https://drive.google.com/file/d/1XpRN8xoEMKxubBHaNyEivgRbnVY2iazu
  FACET:   https://drive.google.com/file/d/1BSjMfKm7FQM8n3HHG5Gn9-dTifULC_Ws

Usage (Kaggle cell):
  !python train_sentiment_kaggle.py --task 7class     # 7-class sentiment
  !python train_sentiment_kaggle.py --task binary     # binary pos/neg
  !python train_sentiment_kaggle.py --task both       # run both (default)
  !python train_sentiment_kaggle.py --analyze         # print results table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics.regression import PearsonCorrCoef, MeanAbsoluteError

from modules.affect_diff_module import AffectDiffModule

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths — adjust to wherever you upload the files on Kaggle
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("/kaggle/input/datasets/ankit58/mosei-sentiment")
BERT_PATH       = DATA_DIR / "BERT_MOSEI.pkl"
COVAREP_PATH    = Path("/kaggle/input/datasets/ankit58/mosei-sentiment/COAVAREP_aligned_MOSEI.pkl")
FACET_PATH      = DATA_DIR / "FACET_aligned_MOSEI.pkl"
# No combined file — always load from separate modality files
COMBINED_PATH   = None

CKPT_DIR        = "/kaggle/working/sentiment_ckpts"
RESULTS_JSON    = "/kaggle/working/sentiment_results.json"

# ── Fine-tune paths (set EMOTION_CACHE_PATH to your uploaded .pt cache) ──────
# Upload mosei_aligned_seq50_v2.pt from data/cache/ as a Kaggle dataset input.
EMOTION_CACHE_PATH    = Path("/kaggle/input/datasets/ankit58/mosei-emotion/mosei_aligned_seq50_v2.pt")
FINETUNE_CKPT_DIR     = "/kaggle/working/finetune_ckpts"
FINETUNE_RESULTS_JSON = "/kaggle/working/finetune_results.json"


# ──────────────────────────────────────────────────────────────────────────────
# Label helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_7class(labels: np.ndarray) -> np.ndarray:
    """Map raw sentiment floats [-3, 3] to integer class 0-6."""
    return np.clip(np.round(labels).astype(int) + 3, 0, 6)

def to_binary(labels: np.ndarray) -> np.ndarray:
    """Map raw sentiment floats to binary: 1=positive (>=0), 0=negative."""
    return (labels >= 0).astype(int)

def class_to_sentiment(cls: np.ndarray) -> np.ndarray:
    """Predicted 7-class index back to sentiment float for MAE/Pearson."""
    return cls.astype(float) - 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Data loading — handles the most common standard MOSEI pickle formats
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def inspect_files():
    """Print top-level structure of all three pkl files to identify keys/format."""
    for name, path in [("BERT", BERT_PATH), ("COVAREP", COVAREP_PATH), ("FACET", FACET_PATH)]:
        if not path.exists():
            print(f"[{name}] NOT FOUND at {path}")
            continue
        d = _load_pkl(path)
        print(f"\n{'='*60}")
        print(f"[{name}]  type={type(d).__name__}")
        if isinstance(d, dict):
            print(f"  top-level keys: {list(d.keys())}")
            for split_key in list(d.keys())[:3]:
                v = d[split_key]
                print(f"  split '{split_key}': type={type(v).__name__}", end="")
                if isinstance(v, dict):
                    print(f"  sub-keys={list(v.keys())}")
                    for sk, sv in list(v.items())[:5]:
                        arr = np.array(sv) if not isinstance(sv, (str, list)) else sv
                        shape = arr.shape if hasattr(arr, 'shape') else (len(arr) if hasattr(arr, '__len__') else '?')
                        print(f"    '{sk}': type={type(sv).__name__}  shape={shape}")
                elif hasattr(v, 'shape'):
                    print(f"  shape={v.shape}  dtype={v.dtype}")
                else:
                    print(f"  len={len(v) if hasattr(v,'__len__') else '?'}")
        elif hasattr(d, 'shape'):
            print(f"  shape={d.shape}  dtype={d.dtype}")
        print()

def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.array(x), dtype=dtype)

def _pad_or_truncate(seq: np.ndarray, max_len: int) -> np.ndarray:
    """Ensure (L, D) → (max_len, D) via padding or truncation."""
    if seq.ndim == 1:
        seq = seq[np.newaxis, :]  # handle (D,) edge case
    L, D = seq.shape
    if L >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - L, D), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

def _discretize(raw: np.ndarray, task: str) -> np.ndarray:
    if task == "7class":
        return to_7class(raw)
    elif task == "binary":
        return to_binary(raw)
    else:
        raise ValueError(f"Unknown task: {task}")


def _extract_cs_features(cs: Any, max_len: int) -> np.ndarray:
    """
    Extract (N, max_len, D) array from a CMU-SDK computational_sequence.
    Iterates .data in sorted segment-ID order for reproducible alignment
    with the BERT flat-index array.
    """
    # Get the underlying data dict
    if hasattr(cs, 'data'):
        data_dict = cs.data
    elif isinstance(cs, dict):
        data_dict = cs
    else:
        raise ValueError(f"Cannot read computational_sequence: {type(cs)}")

    seg_ids = sorted(data_dict.keys())
    feats = []
    for sid in seg_ids:
        entry = data_dict[sid]
        if isinstance(entry, dict) and 'features' in entry:
            f = np.array(entry['features'], dtype=np.float32)
        elif hasattr(entry, 'features'):
            f = np.array(entry.features, dtype=np.float32)
        else:
            f = np.array(entry, dtype=np.float32)
        feats.append(_pad_or_truncate(f, max_len))

    return np.stack(feats, axis=0)   # (N, max_len, D)


def load_mosei_sentiment(
    bert_path: Path,
    covarep_path: Path,
    facet_path: Path,
    combined_path: Optional[Path] = None,
    max_len: int = 50,
    task: str = "7class",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load CMU-MOSEI sentiment from the actual file format:
      BERT_MOSEI.pkl         → {'Data': Tensor(N,768), 'level': Tensor(N)}
      COVAREP_aligned_MOSEI.pkl → computational_sequence (CMU-SDK)
      FACET_aligned_MOSEI.pkl   → computational_sequence (CMU-SDK)

    BERT contains labels ('level') and is flat — no temporal dim.
    Text is unsqueezed to (N, 1, 768) so the model's .mean(dim=1) collapses it.
    COVAREP/FACET are extracted in sorted segment-ID order to align with BERT.
    A reproducible 70/15/15 split is created with seed 42.
    """
    logger.info("Loading BERT …")
    bert_raw  = _load_pkl(bert_path)
    # Handle both Tensor and ndarray
    bert_feats = np.array(bert_raw['Data'], dtype=np.float32)   # (N, 768)
    raw_labels = np.array(bert_raw['level'], dtype=np.float32)  # (N,)
    N = len(raw_labels)
    logger.info("  BERT: %d samples, dim=%d", N, bert_feats.shape[1])

    # Tile BERT sentence embedding to (N, max_len, 768) so all three modalities
    # share the same sequence length for the fusion cat.
    # All max_len positions are identical; the encoder's mean-pool recovers
    # the original sentence vector exactly.
    bert_feats = np.tile(bert_feats[:, np.newaxis, :], (1, max_len, 1))  # (N, max_len, 768)

    logger.info("Loading COVAREP …")
    covarep_cs  = _load_pkl(covarep_path)
    covarep_arr = _extract_cs_features(covarep_cs, max_len)  # (N_cs, max_len, 74)
    logger.info("  COVAREP: %d segs, shape=%s", len(covarep_arr), covarep_arr.shape)

    logger.info("Loading FACET …")
    facet_cs  = _load_pkl(facet_path)
    facet_arr = _extract_cs_features(facet_cs, max_len)      # (N_f, max_len, 35)
    logger.info("  FACET:   %d segs, shape=%s", len(facet_arr), facet_arr.shape)

    # Align counts — use minimum to handle any off-by-one across files
    N = min(N, len(covarep_arr), len(facet_arr))
    bert_feats = bert_feats[:N]
    raw_labels = raw_labels[:N]
    covarep_arr = covarep_arr[:N]
    facet_arr   = facet_arr[:N]
    logger.info("Aligned to %d samples", N)

    # 70 / 15 / 15 split (reproducible)
    rng     = np.random.RandomState(42)
    indices = rng.permutation(N)
    t_end   = int(0.70 * N)
    v_end   = int(0.85 * N)
    splits  = {
        "train": indices[:t_end],
        "val":   indices[t_end:v_end],
        "test":  indices[v_end:],
    }

    out = {}
    for name, idx in splits.items():
        labels_cls = _discretize(raw_labels[idx], task)
        out[name] = {
            "text":       torch.tensor(bert_feats[idx],   dtype=torch.float32),
            "audio":      torch.tensor(covarep_arr[idx],  dtype=torch.float32),
            "vision":     torch.tensor(facet_arr[idx],    dtype=torch.float32),
            "labels":     torch.tensor(labels_cls,        dtype=torch.long),
            "raw_labels": torch.tensor(raw_labels[idx],   dtype=torch.float32),
        }
        logger.info("  %s: %d samples", name, len(idx))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataset and DataModule
# ──────────────────────────────────────────────────────────────────────────────

class _SentimentDataset(Dataset):
    def __init__(self, split_dict: dict) -> None:
        self.text       = split_dict["text"]
        self.audio      = split_dict["audio"]
        self.vision     = split_dict["vision"]
        self.labels     = split_dict["labels"]
        self.raw_labels = split_dict["raw_labels"]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "text":       self.text[idx],
            "audio":      self.audio[idx],
            "vision":     self.vision[idx],
            "labels":     self.labels[idx],
            "raw_labels": self.raw_labels[idx],
        }


class SentimentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str = "7class",
        batch_size: int = 64,
        num_workers: int = 2,
        max_len: int = 50,
    ) -> None:
        super().__init__()
        self.task        = task
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.max_len     = max_len
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        splits = load_mosei_sentiment(
            bert_path=BERT_PATH,
            covarep_path=COVAREP_PATH,
            facet_path=FACET_PATH,
            combined_path=COMBINED_PATH,
            max_len=self.max_len,
            task=self.task,
        )
        self.train_dataset = _SentimentDataset(splits["train"])
        self.val_dataset   = _SentimentDataset(splits["val"])
        self.test_dataset  = _SentimentDataset(splits["test"])

        # Per-class normalised weighting for imbalanced sentiment distribution
        labels = self.train_dataset.labels
        num_cls = int(labels.max().item()) + 1
        counts  = torch.bincount(labels, minlength=num_cls).float().clamp(min=1)
        inv_sqrt = 1.0 / counts.sqrt()
        self.class_weights = (inv_sqrt / inv_sqrt.min()).float()

        # Normalise modalities from training stats
        self._normalize()

    def _normalize(self):
        def _stats(t):
            mean = t.mean(dim=(0, 1), keepdim=True)
            std  = t.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            return mean, std

        # Compute stats only from training set
        stat_map = {}
        for key in ("text", "audio", "vision"):
            t = getattr(self.train_dataset, key).float()
            mean = t.mean(dim=0, keepdim=True)           # (1, L, D) or (1, D)
            std  = t.std(dim=0,  keepdim=True).clamp(min=1e-6)
            stat_map[key] = (mean, std)

        for ds in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for key, (mean, std) in stat_map.items():
                arr = getattr(ds, key).float()
                arr = torch.clamp(torch.nan_to_num(
                    (arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
                    min=-10.0, max=10.0)
                setattr(ds, key, arr)

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
# Sentiment wrapper — overrides test logging to add MAE / Pearson / binary acc
# ──────────────────────────────────────────────────────────────────────────────

SENTIMENT_CLASS_NAMES = {
    7: ["VeryNeg(-3)", "Neg(-2)", "SlightNeg(-1)", "Neutral(0)",
        "SlightPos(+1)", "Pos(+2)", "VeryPos(+3)"],
    2: ["Negative", "Positive"],
}

class SentimentAffectDiff(AffectDiffModule):
    """AffectDiffModule adapted for sentiment: proper metric names + MAE/Pearson."""

    def __init__(self, task: str = "7class", **kwargs):
        # task is not an AffectDiffModule param — store before super().__init__
        self._sentiment_task = task
        super().__init__(**kwargs)
        self._raw_preds: list  = []
        self._raw_labels: list = []

    def test_step(self, batch, batch_idx):
        # Parent accumulates _test_logits / _test_targets; reuse last entry
        loss = super().test_step(batch, batch_idx)
        logits   = self._test_logits[-1]              # (B, C), already on cpu
        pred_cls = torch.argmax(logits, dim=1).float()
        if self._sentiment_task == "7class":
            # 0-6 → -3..+3
            raw_pred = pred_cls - 3.0
        else:
            # binary 0/1 → -1/+1  (enables same MAE/Pearson path)
            raw_pred = pred_cls * 2.0 - 1.0
        self._raw_preds.append(raw_pred)
        self._raw_labels.append(batch["raw_labels"].cpu().float())
        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._raw_preds  = []
        self._raw_labels = []

    def on_test_epoch_end(self):
        # Patch class names before calling super so per-class dicts are labelled correctly
        num_classes = self.hparams.num_classes
        _names = SENTIMENT_CLASS_NAMES.get(num_classes,
                                           [str(i) for i in range(num_classes)])
        import modules.affect_diff_module as _m
        _orig = None
        try:
            # Temporarily monkey-patch the class-name list used by the parent
            import builtins
            _orig_names_7  = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
            # The parent reads class_names from a local variable — we override on_test_epoch_end
            # entirely here instead of relying on super for this.
            if not self._test_logits:
                return
            all_logits  = torch.cat(self._test_logits,  dim=0)
            all_targets = torch.cat(self._test_targets, dim=0)
            preds = torch.argmax(all_logits, dim=1)

            try:
                from torchmetrics.classification import (
                    MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
                    MulticlassAUROC, MulticlassConfusionMatrix,
                )
                f1_mac   = MulticlassF1Score(num_classes=num_classes, average="macro")(preds, all_targets).item()
                f1_per   = MulticlassF1Score(num_classes=num_classes, average="none")(preds, all_targets).tolist()
                prec_per = MulticlassPrecision(num_classes=num_classes, average="none")(preds, all_targets).tolist()
                rec_per  = MulticlassRecall(num_classes=num_classes, average="none")(preds, all_targets).tolist()
                auroc    = MulticlassAUROC(num_classes=num_classes)(all_logits, all_targets.long()).item()
                cm       = MulticlassConfusionMatrix(num_classes=num_classes)(preds, all_targets).tolist()
            except Exception as e:
                logger.warning("Torchmetrics error: %s", e)
                f1_mac, f1_per, prec_per, rec_per, auroc, cm = 0.0, [], [], [], 0.0, []

            bal_acc = sum(rec_per) / len(rec_per) if rec_per else 0.0
            self.log("test_macro_f1",     f1_mac,  sync_dist=True)
            self.log("test_balanced_acc", bal_acc, sync_dist=True)
            self.log("test_auroc",        auroc,   sync_dist=True)
            self.log("test_bal_acc",      bal_acc, sync_dist=True)

            logger.info("Sentiment classification — BalAcc: %.4f  MacroF1: %.4f  AUROC: %.4f",
                        bal_acc, f1_mac, auroc)
            for name, f1 in zip(_names, f1_per):
                logger.info("  %-20s  F1=%.3f", name, f1)

        finally:
            pass

        # Regression metrics (MAE / Pearson / Binary Acc)
        if not self._raw_preds:
            return
        raw_pred  = torch.cat(self._raw_preds,  dim=0)
        raw_label = torch.cat(self._raw_labels, dim=0)

        mae = (raw_pred - raw_label).abs().mean().item()
        try:
            r = PearsonCorrCoef()(raw_pred, raw_label).item()
        except Exception:
            r = float("nan")

        pred_bin = (raw_pred  >= 0).long()
        true_bin = (raw_label >= 0).long()
        bin_acc  = (pred_bin == true_bin).float().mean().item()

        logger.info("Regression — MAE: %.4f  Pearson r: %.4f  Binary Acc: %.4f",
                    mae, r, bin_acc)
        self.log("test_mae",     mae,     sync_dist=True)
        self.log("test_pearson", r,       sync_dist=True)
        self.log("test_bin_acc", bin_acc, sync_dist=True)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment config
# ──────────────────────────────────────────────────────────────────────────────

def make_sentiment_cfg(task: str = "7class") -> dict:
    num_classes = 7 if task == "7class" else 2
    return {
        # Data
        "batch_size":  64,
        "num_workers": 2,
        # Dims — BERT text instead of GloVe
        "text_dim":   768,
        "audio_dim":  74,
        "video_dim":  35,
        "hidden_dim": 128,
        "latent_dim": 128,
        "num_classes": num_classes,
        # Architecture (same as full emotion model)
        "encoder_type":   "legacy",
        "encoder_layers": 2,
        "encoder_dropout": 0.3,
        "fusion_type":    "concat",
        "dag_method":     "notears",
        "num_bottleneck_tokens": 20,
        "num_cross_attn_layers": 1,
        "num_self_attn_layers":  1,
        # Diffusion
        "diffusion_steps": 1000,
        "ddim_steps":      50,
        # Loss weights
        "beta_kl":        0.1,
        "lambda_diff":    0.05,
        "lambda_causal":  0.05,
        "lambda_recon":   0.5,
        "cfg_scale":      3.0,
        "ema_decay":      0.999,
        "label_smoothing": 0.1,
        "free_bits":       0.0,
        # Toggles
        "use_reconstruction": False,
        "use_diffusion":      True,
        "use_causal_graph":   True,
        "use_augmentation":   True,
        "use_beta_tc_vae":    False,
        "use_focal_loss":     True,   # sentiment is mildly imbalanced; keep focal
        "focal_gamma":        1.0,    # milder than emotion task (was 2.0)
        "use_vae":            True,
        "use_stop_gradient":  True,
        "use_cfg":            True,
        "use_causal_diffusion_cond": True,
        "use_kl_warmup":      True,
        # Optimizer
        "lr":           3e-4,    # slightly lower for larger dataset
        "weight_decay": 1e-4,
        "epochs":       60,      # faster convergence with more data
        "patience":     20,
        # Trainer
        "precision":           "16-mixed",
        "gradient_clip_val":   1.0,
        "devices":             1,
        "seed":                42,
        # Logging
        "experiment_name": f"Sentiment_{task}",
        "use_wandb":       False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training runner
# ──────────────────────────────────────────────────────────────────────────────

def run_sentiment_experiment(task: str = "7class") -> dict:
    cfg = make_sentiment_cfg(task)
    pl.seed_everything(cfg["seed"], workers=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    exp_name = cfg["experiment_name"]

    logger.info("=" * 60)
    logger.info("Starting: %s  (task=%s, num_classes=%d, text_dim=%d)",
                exp_name, task, cfg["num_classes"], cfg["text_dim"])

    # Detect GPUs — use all available with DDP
    n_gpus   = max(torch.cuda.device_count(), 1)
    strategy = "ddp" if n_gpus > 1 else "auto"
    workers  = min(4 * n_gpus, 8)
    logger.info("GPUs detected: %d  strategy=%s", n_gpus, strategy)

    # Data
    dm = SentimentDataModule(
        task=task,
        batch_size=cfg["batch_size"],
        num_workers=workers,
    )
    dm.setup()

    # Model  (class_weights NOT passed — AffectDiffModule.on_fit_start pulls
    #         them from self.trainer.datamodule automatically)
    model = SentimentAffectDiff(
        task=task,
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        video_dim=cfg["video_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_classes=cfg["num_classes"],
        encoder_type=cfg["encoder_type"],
        encoder_layers=cfg["encoder_layers"],
        encoder_dropout=cfg["encoder_dropout"],
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
        use_focal_loss=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
        use_vae=cfg["use_vae"],
        use_stop_gradient=cfg["use_stop_gradient"],
        use_cfg=cfg["use_cfg"],
        use_causal_diffusion_cond=cfg["use_causal_diffusion_cond"],
        use_kl_warmup=cfg["use_kl_warmup"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{exp_name}",
        monitor="val_bal_acc", mode="max",
        save_top_k=1, filename="best",
    )
    early_cb = EarlyStopping(
        monitor="val_bal_acc", mode="max",
        patience=cfg["patience"],
    )
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=n_gpus,
        strategy=strategy,
        precision=cfg["precision"],
        gradient_clip_val=cfg["gradient_clip_val"],
        callbacks=[ckpt_cb, early_cb],
        logger=False,
        enable_progress_bar=True,
        deterministic=False,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    train_time = time.time() - t0

    logger.info("Loading best checkpoint: %s", ckpt_cb.best_model_path)
    best = SentimentAffectDiff.load_from_checkpoint(
        ckpt_cb.best_model_path, task=task, strict=False
    )
    test_results = trainer.test(best, datamodule=dm, verbose=True)

    result = {
        "experiment":  exp_name,
        "task":        task,
        "num_classes": cfg["num_classes"],
        "text_dim":    cfg["text_dim"],
        "best_val_bal_acc": float(ckpt_cb.best_model_score or 0),
        "train_time_s":     round(train_time, 1),
        "test":             test_results[0] if test_results else {},
    }
    logger.info("Result: %s", json.dumps(result, indent=2))

    # Append to results JSON
    results = []
    if Path(RESULTS_JSON).exists():
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    results.append(result)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved → %s", RESULTS_JSON)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tune: emotion dataset + checkpoint adapter + experiment runner
# ──────────────────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """Dataset for the emotion .pt cache (no raw_labels — no regression metrics)."""
    def __init__(self, d: dict) -> None:
        self.text   = d["text"]
        self.audio  = d["audio"]
        self.vision = d["vision"]
        self.labels = d["labels"]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "text":   self.text[idx],
            "audio":  self.audio[idx],
            "vision": self.vision[idx],
            "labels": self.labels[idx],
        }


class EmotionFinetuneDataModule(pl.LightningDataModule):
    """Load the pre-built mosei_aligned_seq50_v2.pt cache for emotion fine-tuning.

    Replicates MoseiDataModule.setup() logic: identical 70/15/15 seed-42 split,
    train-stats-only Z-normalization.  Exposes class_weights for on_fit_start.
    """

    def __init__(
        self,
        cache_path: Path,
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.cache_path  = Path(cache_path)
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        logger.info("Loading emotion cache: %s", self.cache_path)
        data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        N = data["labels"].shape[0]
        logger.info("  Total samples: %d", N)

        rng     = torch.Generator()
        rng.manual_seed(42)
        indices = torch.randperm(N, generator=rng).tolist()
        t_end   = int(0.70 * N)
        v_end   = int(0.85 * N)

        def _slice(idx):
            return {k: v[idx] for k, v in data.items()}

        train_raw = _slice(indices[:t_end])

        # Z-normalize from training stats only
        stats: dict = {}
        for key in ("text", "audio", "vision"):
            t    = train_raw[key].float()
            mean = t.mean(dim=0, keepdim=True)
            std  = t.std(dim=0,  keepdim=True).clamp(min=1e-6)
            stats[key] = (mean, std)

        def _norm(d):
            d = {k: v.clone() for k, v in d.items()}
            for key, (mean, std) in stats.items():
                arr = d[key].float()
                arr = torch.clamp(
                    torch.nan_to_num((arr - mean) / std,
                                     nan=0.0, posinf=0.0, neginf=0.0),
                    min=-10.0, max=10.0,
                )
                d[key] = arr
            return d

        self.train_dataset = EmotionDataset(_norm(_slice(indices[:t_end])))
        self.val_dataset   = EmotionDataset(_norm(_slice(indices[t_end:v_end])))
        self.test_dataset  = EmotionDataset(_norm(_slice(indices[v_end:])))

        labels   = self.train_dataset.labels
        num_cls  = int(labels.max().item()) + 1
        counts   = torch.bincount(labels, minlength=num_cls).float().clamp(min=1)
        inv_sqrt = 1.0 / counts.sqrt()
        self.class_weights = (inv_sqrt / inv_sqrt.min()).float()

        logger.info(
            "Emotion split — train: %d  val: %d  test: %d",
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


def adapt_checkpoint(
    ckpt_path: str,
    *,
    source_text_dim: int,
    target_text_dim: int,
    source_num_classes: int,
    target_num_classes: int,
    hidden_dim: int,
    latent_dim: int,
) -> dict:
    """Load a sentiment checkpoint and surgically replace incompatible layers.

    Two layers change when moving sentiment (BERT 768) → emotion (GloVe 300):
      • bottleneck.text_enc.input_proj.0  — Linear(text_dim → hidden_dim)
      • classifier.8                      — Linear(latent_dim//2 → num_classes)

    Everything else (audio/video encoders, VAE, causal graph, diffusion U-Net)
    is kept exactly — those parameters transfer across tasks unchanged.
    """
    logger.info("Loading checkpoint for adaptation: %s", ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)

    new_state: dict = {}
    replaced: list  = []

    text_key = "bottleneck.text_enc.input_proj.0"
    cls_key  = "classifier.8"

    for k, v in state.items():
        if k.startswith(text_key) and source_text_dim != target_text_dim:
            if k.endswith(".weight"):
                new_state[k] = torch.nn.init.kaiming_uniform_(
                    torch.empty(hidden_dim, target_text_dim)
                )
            else:  # .bias
                new_state[k] = torch.zeros(hidden_dim)
            replaced.append(k)

        elif k.startswith(cls_key) and source_num_classes != target_num_classes:
            if k.endswith(".weight"):
                new_state[k] = torch.nn.init.xavier_uniform_(
                    torch.empty(target_num_classes, latent_dim // 2)
                )
            else:  # .bias
                new_state[k] = torch.zeros(target_num_classes)
            replaced.append(k)

        else:
            new_state[k] = v

    logger.info(
        "Adapted %d tensors: %s", len(replaced),
        "  ".join(replaced) if replaced else "(none — dims matched)",
    )
    return new_state


def make_finetune_cfg(source_task: str = "7class") -> dict:
    return {
        "batch_size":  32,
        "num_workers": 2,
        "text_dim":    300,    # GloVe — emotion dataset
        "audio_dim":   74,
        "video_dim":   35,
        "hidden_dim":  128,
        "latent_dim":  128,
        "num_classes": 6,
        "encoder_type":   "legacy",
        "encoder_layers": 2,
        "encoder_dropout": 0.3,
        "fusion_type":    "concat",
        "dag_method":     "notears",
        "num_bottleneck_tokens": 20,
        "num_cross_attn_layers": 1,
        "num_self_attn_layers":  1,
        "diffusion_steps": 1000,
        "ddim_steps":      50,
        "beta_kl":        0.1,
        "lambda_diff":    0.05,
        "lambda_causal":  0.05,
        "lambda_recon":   0.5,
        "cfg_scale":      3.0,
        "ema_decay":      0.999,
        "label_smoothing": 0.1,
        "free_bits":       0.0,
        "use_reconstruction": False,
        "use_diffusion":      True,
        "use_causal_graph":   True,
        "use_augmentation":   True,
        "use_beta_tc_vae":    False,
        "use_focal_loss":     True,
        "focal_gamma":        2.0,
        "use_vae":            True,
        "use_stop_gradient":  True,
        "use_cfg":            True,
        "use_causal_diffusion_cond": True,
        "use_kl_warmup":      True,
        "lr":           1e-4,   # lower LR for fine-tuning
        "weight_decay": 1e-4,
        "epochs":       40,
        "patience":     12,
        "precision":    "16-mixed",
        "gradient_clip_val": 1.0,
        "seed":         42,
        "experiment_name": f"Finetune_{source_task}_to_Emotion",
    }


def run_finetune_experiment(source_task: str = "7class") -> dict:
    """Fine-tune the best sentiment checkpoint on the 6-class emotion task.

    Hypothesis: pre-training on 22 K sentiment samples transfers to emotion
    recognition even though the text modality changes (BERT 768 → GloVe 300).
    The audio/video encoders, causal graph, VAE, and diffusion prior all
    carry over without modification.
    """
    cfg      = make_finetune_cfg(source_task)
    exp_name = cfg["experiment_name"]
    source_ckpt = f"{CKPT_DIR}/Sentiment_{source_task}/best.ckpt"

    if not Path(source_ckpt).exists():
        raise FileNotFoundError(
            f"Sentiment checkpoint not found: {source_ckpt}\n"
            f"Run --task {source_task} first, then --finetune."
        )
    if not EMOTION_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Emotion cache not found: {EMOTION_CACHE_PATH}\n"
            "Upload mosei_aligned_seq50_v2.pt as a Kaggle dataset input."
        )

    pl.seed_everything(cfg["seed"], workers=True)
    os.makedirs(FINETUNE_CKPT_DIR, exist_ok=True)

    n_gpus   = max(torch.cuda.device_count(), 1)
    strategy = "ddp" if n_gpus > 1 else "auto"
    workers  = min(4 * n_gpus, 8)
    logger.info("=" * 60)
    logger.info("Fine-tune: Sentiment_%s → 6-class emotion | GPUs=%d strategy=%s",
                source_task, n_gpus, strategy)

    dm = EmotionFinetuneDataModule(
        cache_path=EMOTION_CACHE_PATH,
        batch_size=cfg["batch_size"],
        num_workers=workers,
    )
    dm.setup()

    # Adapt checkpoint: swap text projection + classifier head
    source_num_classes = 7 if source_task == "7class" else 2
    adapted_state = adapt_checkpoint(
        source_ckpt,
        source_text_dim=768,
        target_text_dim=cfg["text_dim"],
        source_num_classes=source_num_classes,
        target_num_classes=cfg["num_classes"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
    )

    # Build model (identical arch, emotion dims)
    model = AffectDiffModule(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        video_dim=cfg["video_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_classes=cfg["num_classes"],
        encoder_type=cfg["encoder_type"],
        encoder_layers=cfg["encoder_layers"],
        encoder_dropout=cfg["encoder_dropout"],
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
        use_focal_loss=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
        use_vae=cfg["use_vae"],
        use_stop_gradient=cfg["use_stop_gradient"],
        use_cfg=cfg["use_cfg"],
        use_causal_diffusion_cond=cfg["use_causal_diffusion_cond"],
        use_kl_warmup=cfg["use_kl_warmup"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
    )
    # Load adapted weights (strict=False handles any remaining minor mismatches)
    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    if missing:
        logger.info("Missing keys (will be randomly initialized): %s", missing)
    if unexpected:
        logger.info("Unexpected keys (ignored): %s", unexpected)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{FINETUNE_CKPT_DIR}/{exp_name}",
        monitor="val_bal_acc", mode="max",
        save_top_k=1, filename="best",
    )
    early_cb = EarlyStopping(
        monitor="val_bal_acc", mode="max",
        patience=cfg["patience"],
    )
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=n_gpus,
        strategy=strategy,
        precision=cfg["precision"],
        gradient_clip_val=cfg["gradient_clip_val"],
        callbacks=[ckpt_cb, early_cb],
        logger=False,
        enable_progress_bar=True,
        deterministic=False,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    train_time = time.time() - t0

    logger.info("Loading best fine-tune checkpoint: %s", ckpt_cb.best_model_path)
    best = AffectDiffModule.load_from_checkpoint(
        ckpt_cb.best_model_path, strict=False
    )
    test_results = trainer.test(best, datamodule=dm, verbose=True)

    result = {
        "experiment":        exp_name,
        "source_task":       source_task,
        "target_task":       "emotion_6class",
        "pretrain_samples":  22860,
        "finetune_samples":  len(dm.train_dataset),
        "best_val_bal_acc":  float(ckpt_cb.best_model_score or 0),
        "train_time_s":      round(train_time, 1),
        "test":              test_results[0] if test_results else {},
    }
    logger.info("Fine-tune result: %s", json.dumps(result, indent=2))

    results = []
    if Path(FINETUNE_RESULTS_JSON).exists():
        with open(FINETUNE_RESULTS_JSON) as f:
            results = json.load(f)
    results.append(result)
    with open(FINETUNE_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved → %s", FINETUNE_RESULTS_JSON)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Analysis helper
# ──────────────────────────────────────────────────────────────────────────────

def analyze_results():
    if not Path(RESULTS_JSON).exists():
        print("No results file found.")
        return
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    print(f"\n{'Experiment':<28} {'Task':<8} {'Val-BalAcc':>10} "
          f"{'Test-Acc':>10} {'MAE':>7} {'Pearson':>8} {'Bin-Acc':>8}")
    print("-" * 85)
    for r in results:
        t = r.get("test", {})
        print(f"{r['experiment']:<28} {r['task']:<8} "
              f"{r['best_val_bal_acc']:>10.4f} "
              f"{t.get('test_bal_acc', t.get('test_acc', 0)):>10.4f} "
              f"{t.get('test_mae', 0):>7.4f} "
              f"{t.get('test_pearson', 0):>8.4f} "
              f"{t.get('test_bin_acc', 0):>8.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Affect-Diff sentiment training + emotion fine-tuning on Kaggle"
    )
    parser.add_argument("--task", choices=["7class", "binary", "both"],
                        default="both",
                        help="Which sentiment task(s) to train (or which pretrained "
                             "checkpoint(s) to fine-tune from when --finetune is set)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune the best --task checkpoint(s) on CMU-MOSEI "
                             "6-class emotion. Requires a prior --task run AND "
                             "EMOTION_CACHE_PATH pointing to mosei_aligned_seq50_v2.pt")
    parser.add_argument("--analyze", action="store_true",
                        help="Print results table and exit")
    parser.add_argument("--inspect", action="store_true",
                        help="Print pkl file structure and exit (sentiment paths only)")
    args = parser.parse_args()

    if args.inspect:
        inspect_files()
        sys.exit(0)

    if args.analyze:
        analyze_results()
        sys.exit(0)

    if args.finetune:
        # Fine-tune from pre-trained sentiment checkpoint(s) → 6-class emotion
        if args.task in ("7class", "both"):
            run_finetune_experiment("7class")
        if args.task in ("binary", "both"):
            run_finetune_experiment("binary")
        # Print fine-tune summary
        if Path(FINETUNE_RESULTS_JSON).exists():
            with open(FINETUNE_RESULTS_JSON) as f:
                ft_results = json.load(f)
            print(f"\n{'Experiment':<35} {'Val-BalAcc':>10} {'Test-BalAcc':>12} {'MacroF1':>9}")
            print("-" * 72)
            for r in ft_results:
                t = r.get("test", {})
                print(f"{r['experiment']:<35} "
                      f"{r['best_val_bal_acc']:>10.4f} "
                      f"{t.get('test_bal_acc', t.get('test_balanced_acc', 0)):>12.4f} "
                      f"{t.get('test_macro_f1', 0):>9.4f}")
        sys.exit(0)

    # Default: train sentiment
    if args.task in ("7class", "both"):
        run_sentiment_experiment("7class")

    if args.task in ("binary", "both"):
        run_sentiment_experiment("binary")

    analyze_results()
