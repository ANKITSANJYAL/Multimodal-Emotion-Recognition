# Affect-Diff: Multimodal Emotion Recognition via Causal-Diffusion Bridge

**CISC 6080 Capstone Project — Fordham University**
Ankit Sanjyal · asanjyal56@fordham.edu

---

## Overview

Affect-Diff addresses the class-imbalance problem in multimodal emotion recognition on CMU-MOSEI. Standard fusion models collapse to majority-class predictions (Happy = 65.9% of labels), producing zero F1 on Fear, Disgust, and Surprise. Affect-Diff jointly trains three mechanisms:

1. **NOTEARS Causal Graph** — learns a directed acyclic graph over {Text, Audio, Video}; column-sum weights gate each modality before fusion and condition the diffusion denoiser
2. **β-VAE Bottleneck** — compresses the fused representation to a 128-D stochastic latent with free-bits KL regularization
3. **Stop-Gradiented DDPM Prior** — a 1D U-Net learns the class-conditional latent distribution without conflicting with classification gradients

**Results on CMU-MOSEI 6-class emotion recognition (3,292 aligned segments):**

| Method | Val-BalAcc ↑ | Test-Acc |
|--------|-------------|----------|
| TFN (2017) | 0.248 | 0.667 |
| MulT (2019) | 0.278 | 0.626 |
| MISA (2020) | 0.278 | 0.633 |
| MMIM (2021) | 0.266 | 0.679 |
| TETFN (2022) | 0.324 | 0.600 |
| **Affect-Diff (ours)** | **0.384 ± 0.000** | **0.642** |

Val-BalAcc = macro-average recall across all 6 classes. Stable across 3 seeds (42, 43, 44).

---

## Repository Structure

```
Affect-Diff/
├── configs/                    # Training hyperparameters and sweep configs
│   ├── config.yaml
│   ├── config_schema.py
│   └── sweeps/
│
├── Data/                       # Dataset loading and preprocessing
│   ├── mosei_datamodule.py     # PyTorch Lightning DataModule for CMU-MOSEI
│   ├── augmentations.py
│   └── preprocessors/          # Feature extraction from raw CMU-MOSEI SDK
│
├── models/                     # Model architecture
│   ├── causal_graph.py         # NOTEARS DAG module
│   ├── decoder.py              # VAE decoder
│   ├── diffusion/              # 1D U-Net denoiser (unet_1d.py, diffusion_utils.py)
│   ├── encoders/               # Per-modality encoders (text, audio, video)
│   └── fusion/                 # Crossmodal transformer + VAE bottleneck
│
├── modules/
│   └── affect_diff_module.py   # PyTorch Lightning training module (loss, optimiser, metrics)
│
├── utils/
│   ├── loss_functions.py       # Focal loss, free-bits KL, NOTEARS penalty
│   ├── metrics.py              # Balanced accuracy, macro-F1
│   └── visualization.py
│
├── scripts/                    # Cluster and sweep runners
│   ├── run_slurm.sh
│   ├── run_ablation.sh
│   ├── run_ablation_slurm.sh
│   ├── run_ablation_sweep.py
│   └── run_inference.py
│
├── train.py                    # Main training entry point (local/HPC)
├── train_prebuilt.py           # Training from pre-built .pt cache (Kaggle)
├── train_ablation_kaggle.py    # Full ablation suite for Kaggle
├── train_sentiment_kaggle.py   # Sentiment generalizability experiment
├── eval.py                     # Evaluation and metric reporting
├── generate_figures.py         # Reproduces all paper figures from logs
│
├── results/
│   ├── ablation_results.json   # All ablation and baseline run metrics
│   └── sentiment_results.json  # Sentiment generalizability metrics
│
├── logs/                       # CSV training logs (one subfolder per run)
│   ├── Full_Model/
│   ├── No_{Diffusion,Stop_Gradient,NOTEARS,Causal_Graph,VAE}/
│   ├── Baseline_{TFN,MulT,MISA,MMIM,TETFN}/
│   └── seeds/                  # Seeds 43 and 44
│
├── figures/                    # Generated figures (PNG + PDF)
├── paper/
│   └── affect-diff.tex         # LaTeX paper source
└── presentation/
    ├── build_slides.py         # Builds presentation.html + presentation.pdf
    ├── presentation.html       # Self-contained HTML slides (12 slides)
    └── presentation.pdf
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate affect-diff
# or
pip install -r requirements.txt
```

---

## Data

Download or build the CMU-MOSEI pre-aligned cache and place it at:

```
Data/mosei_aligned_seq50_v2.pt
```

This `.pt` file contains GloVe 300-d text, COVAREP 74-d audio, and FACET 35-d video features for 3,292 strictly tri-modally aligned segments with 6-class Ekman emotion labels. Split is 70/15/15 (train/val/test), seed 42. Z-normalization is computed from the training split only.

**Why 3,292 of 23,453?** Strict temporal alignment across all three modalities (GloVe word-level, COVAREP frame-level, FACET frame-level) discards segments where any modality has a missing or misaligned array. This retains ~14% of the corpus. The sentiment experiment on 22,860 segments confirms the architecture scales: 7× more data yields +90% balanced accuracy.

To rebuild from raw CMU-MOSEI SDK files:
```bash
python Data/preprocessors/extract_audio_features.py
python Data/preprocessors/extract_video_features.py
```

---

## Training

### Full model
```bash
python train.py
# or from pre-built cache (Kaggle / no SDK)
python train_prebuilt.py
```

### Ablation suite
```bash
bash scripts/run_ablation.sh          # local, sequential
bash scripts/run_ablation_slurm.sh    # SLURM cluster, parallel
# on Kaggle:
python train_ablation_kaggle.py --task all
```

### Sentiment generalizability
```bash
python train_sentiment_kaggle.py --task both
python train_sentiment_kaggle.py --task both --finetune  # fine-tune to emotion task
```

---

## Reproducing Figures

All figures are generated from the CSV logs in `logs/`:

```bash
python generate_figures.py
```

Output goes to `figures/`.

---

## Rebuilding the Presentation

```bash
python presentation/build_slides.py
```

Requires Google Chrome on macOS. Output: `presentation/presentation.html` and `presentation/presentation.pdf`.

---

## Architecture

```
Text (GloVe 300d) ─┐
Audio (COVAREP 74d)─┼─► Encoders ─► NOTEARS Causal Graph ─► Crossmodal Fusion
Video (FACET 35d) ─┘   (BiLSTM)     h(A)=tr(e^(A∘A))−3=0   (Transformer)
                                                                     │
                                                              β-VAE Bottleneck
                                                             z = μ + ε·σ ∈ ℝ¹²⁸
                                                                     │
                                                     ┌───────────────┤
                                                     │               │
                                               Classifier      DDPM Prior
                                              (6 emotions)  sg(z) → 1D U-Net
                                                     │               │
                                                  L_task   L_diff + L_KL + λ·h(A)
```

`sg(z)` (stop-gradient) decouples the diffusion loss from the encoder, allowing both objectives to train without gradient conflict.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 128 |
| β (KL weight) | 0.1 |
| Free-bits λ | 2.0 |
| γ_diff | 0.1 |
| λ_DAG | 0.01 |
| Diffusion timesteps | 1000 |
| Optimizer | AdamW, lr=3×10⁻⁴ |
| Batch size | 64 |
| Max epochs | 60 (patience=15) |

---

## Tests

```bash
pytest tests/
```
