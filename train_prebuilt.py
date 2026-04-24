"""Affect-Diff training script for Kaggle (pre-built dataset).

Loads directly from a pre-aligned .pt file — no mmsdk alignment step.
Configured via argparse so it works without Hydra on Kaggle.

Usage (single run):
    python train_prebuilt.py

Usage (specific ablation):
    python train_prebuilt.py --ablation Classifier_Only

Usage (sweep all ablations):
    python train_prebuilt.py --run_all_ablations

Dataset path (Kaggle):
    /kaggle/input/datasets/ankit58/moesi-aligned/mosei_aligned_seq50_v2.pt
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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
# Default config (mirrors config.yaml — edit here to change baseline)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    # Paths
    "pt_path": "/kaggle/input/datasets/ankit58/moesi-aligned/mosei_aligned_seq50_v2.pt",
    "ckpt_dir": "/kaggle/working/checkpoints",
    # Data
    "batch_size": 64,
    "num_workers": 2,
    # Model dims
    "text_dim": 300,
    "audio_dim": 74,
    "video_dim": 35,
    "hidden_dim": 512,
    "latent_dim": 256,
    "num_classes": 6,
    # Encoder / fusion / DAG
    "encoder_type": "legacy",
    "fusion_type": "concat",   # ablations: concat = crossmodal accuracy, ~2× faster to converge
    "num_bottleneck_tokens": 50,
    "num_cross_attn_layers": 2,
    "num_self_attn_layers": 2,
    "dag_method": "notears",
    # Diffusion
    "diffusion_steps": 1000,
    "ddim_steps": 50,
    # Loss weights
    "beta_kl": 0.1,           # was 0.5 — lighter KL so task gradient dominates
    "lambda_diff": 0.1,       # if diffusion re-enabled, keep it light (ablations: 0.1 is best)
    "lambda_causal": 0.05,
    "lambda_recon": 0.5,
    "cfg_scale": 3.0,
    "ema_decay": 0.999,
    "label_smoothing": 0.1,
    "free_bits": 0.0,         # 0 = no floor; with beta_kl=0.1 and cyclical annealing, no collapse
    # Ablation toggles — start simple; reconstruction caused recon=inf, diffusion adds noise
    "use_reconstruction": False,
    "use_diffusion": False,
    "use_causal_graph": True,
    "use_augmentation": True,
    "use_beta_tc_vae": False,
    # Optimizer
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "epochs": 100,
    "patience": 20,
    # Trainer
    "precision": "16-mixed",
    "gradient_clip_val": 1.0,
    "devices": 1,
    "seed": 42,
    # Logging
    "experiment_name": "AffectDiff_Kaggle",
    "use_wandb": False,
    "wandb_project": "Affect-Diff-CVPR",
}

# ──────────────────────────────────────────────────────────────────────────────
# Ablation experiment definitions
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Component ablations
    "Full_Model":           {},
    "No_Diffusion":         {"use_diffusion": False},
    "No_Causal":            {"use_causal_graph": False},
    "No_Reconstruction":    {"use_reconstruction": False},
    "No_Augmentation":      {"use_augmentation": False},
    "Classifier_Only":      {
        "use_diffusion": False,
        "use_causal_graph": False,
        "use_reconstruction": False,
        "use_augmentation": False,
    },
    "Concat_Fusion":        {"fusion_type": "concat"},
    "Gumbel_DAG":           {"dag_method": "gumbel"},
    # Hyperparameter sweeps
    "HP_beta_kl_0.01":      {"beta_kl": 0.01},
    "HP_beta_kl_0.1":       {"beta_kl": 0.1},
    "HP_beta_kl_1.0":       {"beta_kl": 1.0},
    "HP_lambda_diff_0.1":   {"lambda_diff": 0.1},
    "HP_lambda_diff_1.0":   {"lambda_diff": 1.0},
    "HP_lambda_causal_0.01":{"lambda_causal": 0.01},
    "HP_lambda_causal_0.5": {"lambda_causal": 0.5},
    "HP_label_smoothing_0.0":  {"label_smoothing": 0.0},
    "HP_label_smoothing_0.2":  {"label_smoothing": 0.2},
    "HP_free_bits_0.0":     {"free_bits": 0.0},
    "HP_free_bits_1.0":     {"free_bits": 1.0},
    "HP_free_bits_4.0":     {"free_bits": 4.0},
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataModule (no alignment — loads directly from .pt)
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
    """DataModule that loads directly from a pre-built .pt file.

    Performs the same 70/15/15 split and Z-normalization as MoseiDataModule,
    but skips the mmsdk alignment step entirely.
    """

    def __init__(self, pt_path: str, batch_size: int = 64, num_workers: int = 2) -> None:
        super().__init__()
        self.pt_path     = pt_path
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.class_weights: torch.Tensor | None = None
        self.norm_stats: dict = {}

    def setup(self, stage: str | None = None) -> None:
        data_dict = torch.load(self.pt_path, weights_only=False)
        total = data_dict["labels"].shape[0]
        logger.info("Loaded %d samples from %s", total, self.pt_path)

        # Fixed-seed 70/15/15 split — identical to MoseiDataModule
        rng = torch.Generator()
        rng.manual_seed(42)
        indices = torch.randperm(total, generator=rng).tolist()

        train_end = int(0.70 * total)
        val_end   = int(0.85 * total)
        train_idx = indices[:train_end]
        val_idx   = indices[train_end:val_end]
        test_idx  = indices[val_end:]

        logger.info(
            "Split: train=%d  val=%d  test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        def _slice(idxs: list) -> dict:
            t = torch.tensor(idxs, dtype=torch.long)
            return {k: v[t] for k, v in data_dict.items()}

        def _stats(tensor: torch.Tensor):
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

        def _norm(d: dict) -> dict:
            d = {k: v.clone() for k, v in d.items()}
            # Clamp after normalization: the .pt file's nan_to_num replaces original +inf
            # feature values with 3.4e38 (float32 max). After dividing by a small std,
            # those survive as huge finite numbers. MSE against those causes recon=inf
            # because (3.4e28)^2 overflows float32. Clamp to [-10, 10] neutralises them.
            for key, mean, std in [("text", t_mean, t_std), ("audio", a_mean, a_std), ("vision", v_mean, v_std)]:
                d[key] = torch.clamp(
                    torch.nan_to_num((d[key] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
                    min=-10.0, max=10.0,
                )
            return d

        self.train_dataset = _MoseiDataset(_norm(_slice(train_idx)))
        self.val_dataset   = _MoseiDataset(_norm(_slice(val_idx)))
        self.test_dataset  = _MoseiDataset(_norm(_slice(test_idx)))

        # Sqrt inverse-frequency class weights — normalised so minimum class weight = 1.0.
        # Full inverse-frequency ({0:0.25, 3:9.9}) gave a 39x ratio between Happy and Fear,
        # which drove the model to predict rare classes everywhere (val_acc dropped to 2.9%).
        # Sqrt reduces the ratio to ~6x, still compensating for imbalance without overpowering
        # the task gradient.
        labels = self.train_dataset.labels
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=num_classes).float()
        inv_freq = labels.shape[0] / (num_classes * counts.clamp(min=1))
        sqrt_w = inv_freq.sqrt()
        self.class_weights = sqrt_w / sqrt_w.min()  # min class = 1.0
        logger.info(
            "Class weights (sqrt, min-normalised): %s",
            {i: f"{w:.3f}" for i, w in enumerate(self.class_weights.tolist())},
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Training function (single run)
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Train + test one experiment. Returns test metrics dict."""
    pl.seed_everything(cfg["seed"], workers=True)

    exp_name = cfg["experiment_name"]
    logger.info("=== Starting experiment: %s ===", exp_name)

    # DataModule
    datamodule = PrebuiltDataModule(
        pt_path=cfg["pt_path"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # Model
    model = AffectDiffModule(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        video_dim=cfg["video_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_classes=cfg["num_classes"],
        encoder_type=cfg["encoder_type"],
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
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
    )

    # Logger
    if cfg["use_wandb"]:
        try:
            from pytorch_lightning.loggers import WandbLogger
            pl_logger = WandbLogger(
                project=cfg["wandb_project"],
                name=exp_name,
                config=cfg,
            )
        except Exception as e:
            logger.warning("WandbLogger failed (%s) — falling back to CSVLogger", e)
            cfg["use_wandb"] = False

    if not cfg["use_wandb"]:
        from pytorch_lightning.loggers import CSVLogger
        pl_logger = CSVLogger(
            save_dir=os.path.join(cfg["ckpt_dir"], "csv_logs"),
            name=exp_name,
        )

    # Callbacks
    ckpt_dir = os.path.join(cfg["ckpt_dir"], exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_acc",
            mode="max",
            save_top_k=2,
            filename="affect-diff-{epoch:02d}-{val_acc:.3f}",
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=cfg["patience"],
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
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
    results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    test_metrics = results[0] if results else {}
    logger.info("=== Finished %s | test metrics: %s ===", exp_name, test_metrics)
    return test_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Ablation sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_sweep(base_cfg: Dict[str, Any], ablation_names: list[str]) -> None:
    """Run multiple ablation experiments and print a summary table."""
    results_table = []

    for name in ablation_names:
        overrides = ABLATION_CONFIGS.get(name, {})
        cfg = {**base_cfg, **overrides, "experiment_name": f"AblationStudy_{name}"}
        try:
            metrics = run_experiment(cfg)
            test_acc = metrics.get("test_acc", float("nan"))
        except Exception as e:
            logger.error("Experiment %s crashed: %s", name, e)
            test_acc = float("nan")

        results_table.append((name, test_acc))

        # Print running summary after each experiment
        print("\n" + "=" * 60)
        print("ABLATION RESULTS SO FAR")
        print("=" * 60)
        print(f"{'Experiment':<35} {'Test Acc':>10}")
        print("-" * 60)
        for exp_name, acc in results_table:
            acc_str = f"{acc:.4f}" if acc == acc else "CRASHED"
            print(f"{exp_name:<35} {acc_str:>10}")
        print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Affect-Diff training for Kaggle (pre-built .pt dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run mode
    p.add_argument("--ablation", default=None, choices=list(ABLATION_CONFIGS.keys()),
                   help="Run a specific ablation (applies its overrides on top of defaults)")
    p.add_argument("--run_all_ablations", action="store_true",
                   help="Run every ablation experiment sequentially and print summary")
    p.add_argument("--ablation_subset", nargs="+", default=None,
                   choices=list(ABLATION_CONFIGS.keys()),
                   help="Run a subset of ablations (space-separated names)")

    # Paths
    p.add_argument("--pt_path", default=DEFAULTS["pt_path"])
    p.add_argument("--ckpt_dir", default=DEFAULTS["ckpt_dir"])
    p.add_argument("--experiment_name", default=DEFAULTS["experiment_name"])

    # Data
    p.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--num_workers",  type=int,   default=DEFAULTS["num_workers"])

    # Architecture overrides (useful for quick one-off experiments)
    p.add_argument("--fusion_type",   default=DEFAULTS["fusion_type"],
                   choices=["concat", "crossmodal"])
    p.add_argument("--dag_method",    default=DEFAULTS["dag_method"],
                   choices=["notears", "gumbel"])
    p.add_argument("--hidden_dim",    type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--latent_dim",    type=int, default=DEFAULTS["latent_dim"])

    # Loss weights
    p.add_argument("--beta_kl",          type=float, default=DEFAULTS["beta_kl"])
    p.add_argument("--lambda_diff",      type=float, default=DEFAULTS["lambda_diff"])
    p.add_argument("--lambda_causal",    type=float, default=DEFAULTS["lambda_causal"])
    p.add_argument("--lambda_recon",     type=float, default=DEFAULTS["lambda_recon"])
    p.add_argument("--label_smoothing",  type=float, default=DEFAULTS["label_smoothing"])
    p.add_argument("--free_bits",        type=float, default=DEFAULTS["free_bits"])

    # Ablation toggles
    p.add_argument("--no_diffusion",     action="store_true")
    p.add_argument("--no_causal",        action="store_true")
    p.add_argument("--no_reconstruction",action="store_true")
    p.add_argument("--no_augmentation",  action="store_true")

    # Training
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--patience",       type=int,   default=DEFAULTS["patience"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--precision",      default=DEFAULTS["precision"],
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--devices",        type=int,   default=DEFAULTS["devices"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])

    # Logging
    p.add_argument("--use_wandb",      action="store_true", help="Log to W&B (needs WANDB_API_KEY)")
    p.add_argument("--wandb_project",  default=DEFAULTS["wandb_project"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build base config from defaults + CLI overrides
    cfg: Dict[str, Any] = {**DEFAULTS}
    cli_overrides = {
        "pt_path":           args.pt_path,
        "ckpt_dir":          args.ckpt_dir,
        "experiment_name":   args.experiment_name,
        "batch_size":        args.batch_size,
        "num_workers":       args.num_workers,
        "fusion_type":       args.fusion_type,
        "dag_method":        args.dag_method,
        "hidden_dim":        args.hidden_dim,
        "latent_dim":        args.latent_dim,
        "beta_kl":           args.beta_kl,
        "lambda_diff":       args.lambda_diff,
        "lambda_causal":     args.lambda_causal,
        "lambda_recon":      args.lambda_recon,
        "label_smoothing":   args.label_smoothing,
        "free_bits":         args.free_bits,
        "use_diffusion":     not args.no_diffusion,
        "use_causal_graph":  not args.no_causal,
        "use_reconstruction":not args.no_reconstruction,
        "use_augmentation":  not args.no_augmentation,
        "epochs":            args.epochs,
        "patience":          args.patience,
        "lr":                args.lr,
        "weight_decay":      args.weight_decay,
        "precision":         args.precision,
        "devices":           args.devices,
        "seed":              args.seed,
        "use_wandb":         args.use_wandb,
        "wandb_project":     args.wandb_project,
    }
    cfg.update(cli_overrides)

    # Decide run mode
    if args.run_all_ablations:
        run_ablation_sweep(cfg, list(ABLATION_CONFIGS.keys()))

    elif args.ablation_subset:
        run_ablation_sweep(cfg, args.ablation_subset)

    elif args.ablation:
        overrides = ABLATION_CONFIGS[args.ablation]
        cfg.update(overrides)
        cfg["experiment_name"] = f"AblationStudy_{args.ablation}"
        run_experiment(cfg)

    else:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
