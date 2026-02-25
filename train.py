import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch

from Data.mosei_datamodule import MoseiDataModule
from modules.affect_diff_module import AffectDiffModule

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print the config so it's recorded in the SLURM output logs
    print(OmegaConf.to_yaml(cfg))

    # 1. Initialize PyTorch Lightning Seed for Reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # 2. Setup DataModule
    datamodule = MoseiDataModule(
        data_dir=cfg.data.data_dir,
        cache_dir=cfg.data.cache_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    # 3. Setup Model
    model = AffectDiffModule(
        text_dim=cfg.model.text_dim,
        audio_dim=cfg.model.audio_dim,
        video_dim=cfg.model.video_dim,
        latent_dim=cfg.model.latent_dim,
        diffusion_steps=cfg.diffusion.steps,
        lr=cfg.trainer.lr,
        beta_kl=cfg.model.beta_kl,
        lambda_diff=cfg.model.lambda_diff
    )

    # Log parameter count for debugging
    if torch.distributed.is_initialized():
        print(f"Rank {torch.distributed.get_rank()} parameter count: {sum(p.numel() for p in model.parameters())}")

    # 4. Setup Callbacks (Weights & Biases, Checkpointing)
    wandb_logger = WandbLogger(project="Affect-Diff-CVPR", name=cfg.experiment_name)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="affect-diff-{epoch:02d}-{val_acc:.3f}"
        ),
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]

    # 5. Setup PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.trainer.devices,          # E.g., 8 GPUs
        strategy="ddp_find_unused_parameters_true" if cfg.trainer.devices > 1 else "auto", # Distributed Data Parallel
        precision=32,                         # Switch to full precision for maximum stability in Counterfactual Hallucination
        gradient_clip_val=1.0,                # DeepMind standard: prevent exploding gradients
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5
    )

    # 6. Train!
    trainer.fit(model, datamodule=datamodule)

    # 7. Evaluate on Test Set
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()
