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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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

DATA_DIR        = Path("/kaggle/input/mosei-sentiment")   # dataset you create
BERT_PATH       = DATA_DIR / "bert_embeddings.pkl"
COVAREP_PATH    = DATA_DIR / "covarep.pkl"
FACET_PATH      = DATA_DIR / "facet.pkl"
# Fallback: combined single pickle (try this first)
COMBINED_PATH   = DATA_DIR / "mosei_sentiment.pkl"

CKPT_DIR        = "/kaggle/working/sentiment_ckpts"
RESULTS_JSON    = "/kaggle/working/sentiment_results.json"


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

def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.array(x), dtype=dtype)

def _pad_or_truncate(seq: np.ndarray, max_len: int) -> np.ndarray:
    """Ensure (L, D) → (max_len, D) via padding or truncation."""
    L, D = seq.shape
    if L >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - L, D), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

def load_mosei_sentiment(
    bert_path: Path,
    covarep_path: Path,
    facet_path: Path,
    combined_path: Optional[Path] = None,
    max_len: int = 50,
    task: str = "7class",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load CMU-MOSEI sentiment data from the standard pickle format.

    Supports three common layouts:
      1. Single combined pickle: {split: {text, audio, vision, labels}}
      2. Separate modality pickles with same split structure
      3. CMU-SDK format: {split: {seg_id: {features: ..., labels: ...}}}

    Returns dict: {split: {text, audio, vision, labels, raw_labels}}
      - labels     : long tensor for classification (0-6 for 7class, 0-1 for binary)
      - raw_labels : float tensor of raw sentiment values for MAE/Pearson
    """
    # ── Try combined file first ──────────────────────────────────────────────
    if combined_path and combined_path.exists():
        logger.info("Loading combined file: %s", combined_path)
        data = _load_pkl(combined_path)
        return _parse_combined(data, max_len, task)

    # ── Separate modality files ──────────────────────────────────────────────
    logger.info("Loading separate modality files …")
    bert    = _load_pkl(bert_path)
    covarep = _load_pkl(covarep_path)
    facet   = _load_pkl(facet_path)
    return _merge_modalities(bert, covarep, facet, max_len, task)


def _parse_combined(data: dict, max_len: int, task: str) -> dict:
    """Parse {split: {text/audio/vision/labels}} format."""
    out = {}
    split_map = {"train": "train", "dev": "val", "valid": "val",
                 "validation": "val", "test": "test"}
    for raw_key, split in split_map.items():
        if raw_key not in data:
            continue
        d = data[raw_key]
        text   = _extract_modality(d, ["text", "bert", "bert_embeddings"], max_len)
        audio  = _extract_modality(d, ["audio", "covarep"], max_len)
        vision = _extract_modality(d, ["vision", "video", "facet"], max_len)
        raw    = _extract_labels(d)
        labels = _discretize(raw, task)
        out[split] = {
            "text":       _to_tensor(text),
            "audio":      _to_tensor(audio),
            "vision":     _to_tensor(vision),
            "labels":     torch.tensor(labels, dtype=torch.long),
            "raw_labels": torch.tensor(raw, dtype=torch.float32),
        }
        logger.info("  %s: %d samples", split, len(labels))
    return out


def _merge_modalities(bert, covarep, facet, max_len, task) -> dict:
    """Merge three separate modality dicts into a combined split dict."""
    out = {}
    split_map = {"train": "train", "dev": "val", "valid": "val", "test": "test"}
    for raw_key, split in split_map.items():
        if raw_key not in bert:
            continue
        b = bert[raw_key]
        c = covarep[raw_key]
        f = facet[raw_key]
        text   = _extract_modality(b, ["text", "features", "data"], max_len)
        audio  = _extract_modality(c, ["audio", "features", "data"], max_len)
        vision = _extract_modality(f, ["vision", "features", "data"], max_len)
        raw    = _extract_labels(b)
        labels = _discretize(raw, task)
        out[split] = {
            "text":       _to_tensor(text),
            "audio":      _to_tensor(audio),
            "vision":     _to_tensor(vision),
            "labels":     torch.tensor(labels, dtype=torch.long),
            "raw_labels": torch.tensor(raw, dtype=torch.float32),
        }
        logger.info("  %s: %d samples", split, len(labels))
    return out


def _extract_modality(d: Any, keys: List[str], max_len: int) -> np.ndarray:
    """Try multiple key names; handle (N,L,D) or SDK dict-of-dicts format."""
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                arr = np.array(d[k])
                if arr.ndim == 3:
                    return np.stack([_pad_or_truncate(arr[i], max_len)
                                     for i in range(len(arr))])
                return arr
        # SDK format: dict of seg_id → {features: arr}
        segs = []
        for seg_id, seg_data in d.items():
            if isinstance(seg_data, dict) and "features" in seg_data:
                feats = np.array(seg_data["features"])
                if feats.ndim == 2:
                    segs.append(_pad_or_truncate(feats, max_len))
        if segs:
            return np.stack(segs)
    raise ValueError(f"Cannot parse modality — tried keys {keys}")


def _extract_labels(d: Any) -> np.ndarray:
    """Extract raw regression labels from various formats."""
    for k in ["labels", "label", "sentiment", "y"]:
        if isinstance(d, dict) and k in d:
            arr = np.array(d[k]).squeeze()
            return arr.astype(np.float32)
    # SDK: labels in each segment
    if isinstance(d, dict):
        vals = []
        for seg_id, seg_data in d.items():
            if isinstance(seg_data, dict) and "labels" in seg_data:
                vals.append(float(np.array(seg_data["labels"]).squeeze()))
        if vals:
            return np.array(vals, dtype=np.float32)
    raise ValueError("Cannot extract labels")


def _discretize(raw: np.ndarray, task: str) -> np.ndarray:
    if task == "7class":
        return to_7class(raw)
    elif task == "binary":
        return to_binary(raw)
    else:
        raise ValueError(f"Unknown task: {task}")


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

        def _apply(ds, stats):
            for key, mean, std in stats:
                arr = getattr(ds, key).float()
                arr = torch.clamp(torch.nan_to_num(
                    (arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
                    min=-10.0, max=10.0)
                setattr(ds, key, arr)

        stats = [
            ("text",   *_stats(self.train_dataset.text.float())),
            ("audio",  *_stats(self.train_dataset.audio.float())),
            ("vision", *_stats(self.train_dataset.vision.float())),
        ]
        for ds in [self.train_dataset, self.val_dataset, self.test_dataset]:
            _apply(ds, stats)

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

class SentimentAffectDiff(AffectDiffModule):
    """AffectDiffModule adapted for sentiment: proper metric names + MAE/Pearson."""

    def __init__(self, task: str = "7class", **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self._raw_preds: list  = []
        self._raw_labels: list = []

    def test_step(self, batch, batch_idx):
        loss = super().test_step(batch, batch_idx)
        with torch.no_grad():
            logits = self(batch["text"], batch["audio"], batch["vision"])
            pred_cls = torch.argmax(logits, dim=1).cpu().float()
            # convert back to sentiment scale for regression metrics
            self._raw_preds.append(pred_cls - 3.0)
            self._raw_labels.append(batch["raw_labels"].cpu().float())
        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._raw_preds  = []
        self._raw_labels = []

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if not self._raw_preds:
            return
        preds  = torch.cat(self._raw_preds,  dim=0)
        labels = torch.cat(self._raw_labels, dim=0)

        mae  = (preds - labels).abs().mean().item()
        try:
            r = PearsonCorrCoef()(preds, labels).item()
        except Exception:
            r = float("nan")

        # Binary accuracy: >=0 = positive
        pred_bin  = (preds  >= 0).long()
        true_bin  = (labels >= 0).long()
        bin_acc   = (pred_bin == true_bin).float().mean().item()

        logger.info("Sentiment test — MAE: %.4f  Pearson r: %.4f  Binary Acc: %.4f",
                    mae, r, bin_acc)
        self.log("test_mae",      mae,     sync_dist=True)
        self.log("test_pearson",  r,       sync_dist=True)
        self.log("test_bin_acc",  bin_acc, sync_dist=True)


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

    # Data
    dm = SentimentDataModule(
        task=task,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    dm.setup()

    # Model
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
        dag_method=cfg["dag_method"],
        diffusion_steps=cfg["diffusion_steps"],
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
        class_weights=dm.class_weights,
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
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=cfg["devices"],
        precision=cfg["precision"],
        gradient_clip_val=cfg["gradient_clip_val"],
        callbacks=[ckpt_cb, early_cb, lr_cb],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["7class", "binary", "both"],
                        default="both")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        sys.exit(0)

    if args.task in ("7class", "both"):
        run_sentiment_experiment("7class")

    if args.task in ("binary", "both"):
        run_sentiment_experiment("binary")

    analyze_results()
