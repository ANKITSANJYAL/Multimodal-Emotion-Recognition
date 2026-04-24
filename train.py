"""Affect-Diff training entry point.

Wires all config.yaml values to the model, trainer, and data modules.
Uses Hydra for configuration management with structured config validation.
"""

import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from Data.mosei_datamodule import MoseiDataModule
from modules.affect_diff_module import AffectDiffModule

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra changes cwd — resolve paths relative to the original project root
    orig_cwd = hydra.utils.get_original_cwd()

    # Print config for logs
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Extended timeout for massive data alignment
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # 1. Reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # 2. DataModule — use absolute paths so Hydra cwd change doesn't break things
    datamodule = MoseiDataModule(
        data_dir=os.path.join(orig_cwd, cfg.data.data_dir),
        cache_dir=os.path.join(orig_cwd, cfg.data.cache_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seq_len=cfg.data.seq_len,
    )

    # 3. Model — ALL config values wired through
    model = AffectDiffModule(
        # Dimensions
        text_dim=cfg.model.text_dim,
        audio_dim=cfg.model.audio_dim,
        video_dim=cfg.model.video_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        num_classes=cfg.model.num_classes,
        # Encoder / Fusion / DAG
        encoder_type=cfg.model.encoder_type,
        text_backbone=cfg.model.text_backbone,
        audio_backbone=cfg.model.audio_backbone,
        video_backbone=cfg.model.video_backbone,
        freeze_backbones=cfg.model.freeze_backbones,
        fusion_type=cfg.model.fusion_type,
        num_bottleneck_tokens=cfg.model.num_bottleneck_tokens,
        num_cross_attn_layers=cfg.model.num_cross_attn_layers,
        num_self_attn_layers=cfg.model.num_self_attn_layers,
        dag_method=cfg.model.dag_method,
        # Diffusion
        diffusion_steps=cfg.diffusion.steps,
        ddim_steps=cfg.diffusion.ddim_steps,
        # Loss weights
        beta_kl=cfg.model.beta_kl,
        lambda_diff=cfg.model.lambda_diff,
        lambda_causal=cfg.model.lambda_causal,
        lambda_recon=cfg.model.lambda_recon,
        cfg_scale=cfg.model.cfg_scale,
        # Optimizer
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        epochs=cfg.trainer.epochs,
        # EMA
        ema_decay=cfg.model.ema_decay,
        # Ablation toggles
        use_reconstruction=cfg.model.use_reconstruction,
        use_diffusion=cfg.model.use_diffusion,
        use_causal_graph=cfg.model.use_causal_graph,
        use_augmentation=cfg.model.use_augmentation,
        use_beta_tc_vae=cfg.model.use_beta_tc_vae,
        # Regularization
        label_smoothing=cfg.model.label_smoothing,
        free_bits=cfg.model.free_bits,
    )

    # 4. Logger + Callbacks
    wandb_logger = WandbLogger(project="Affect-Diff-CVPR", name=cfg.experiment_name)

    # Per-experiment checkpoint directory to prevent cross-contamination
    ckpt_dir = os.path.join(orig_cwd, "checkpoints", cfg.experiment_name)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            filename="affect-diff-{epoch:02d}-{val_acc:.3f}",
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=cfg.trainer.patience,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # 5. Trainer — all config values wired
    # Use find_unused_parameters=True for ablation runs that disable components
    if cfg.trainer.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.trainer.devices,
        strategy=strategy,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # 6. Train
    trainer.fit(model, datamodule=datamodule)

    # 7. Test
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
