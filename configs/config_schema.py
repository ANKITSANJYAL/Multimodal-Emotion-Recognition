"""Structured configuration schema with dataclass validation.

Provides compile-time type checking and runtime validation for all
configuration parameters via OmegaConf structured configs.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    batch_size: int = 64
    num_workers: int = 8
    seq_len: int = 50


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    num_classes: int = 6
    text_dim: int = 300
    audio_dim: int = 74
    video_dim: int = 35
    hidden_dim: int = 512
    latent_dim: int = 256
    beta_kl: float = 5.0
    lambda_diff: float = 1.0
    lambda_causal: float = 0.1
    lambda_recon: float = 0.5
    cfg_scale: float = 3.0
    ema_decay: float = 0.999

    # Encoder backbone selection
    encoder_type: Literal["legacy", "foundation"] = "legacy"
    text_backbone: str = "roberta-base"
    audio_backbone: str = "facebook/hubert-base-ls960"
    video_backbone: str = "openai/clip-vit-base-patch16"
    freeze_backbones: bool = True

    # Fusion method
    fusion_type: Literal["concat", "crossmodal"] = "crossmodal"
    num_bottleneck_tokens: int = 50
    num_cross_attn_layers: int = 2
    num_self_attn_layers: int = 2

    # DAG learning
    dag_method: Literal["gumbel", "notears"] = "notears"

    # Component toggles for ablation studies
    use_reconstruction: bool = True
    use_diffusion: bool = True
    use_causal_graph: bool = True
    use_augmentation: bool = True
    use_beta_tc_vae: bool = False


@dataclass
class DiffusionConfig:
    """Diffusion generative prior configuration."""
    steps: int = 1000
    schedule: str = "cosine"
    sampler: Literal["ddpm", "ddim"] = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0


@dataclass
class TrainerConfig:
    """Training and optimization configuration."""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    devices: int = 2
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    patience: int = 50
    log_every_n_steps: int = 5


@dataclass
class EvalConfig:
    """Evaluation and inference configuration."""
    checkpoint_path: str = "checkpoints/best.ckpt"
    dissonance_threshold: float = 0.4
    cfg_scale: float = 4.0
    use_ddim: bool = True
    ddim_steps: int = 50


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    experiment_name: str = "AffectDiff_Baseline_Joint_Training"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
