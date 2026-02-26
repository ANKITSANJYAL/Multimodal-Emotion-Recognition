# Ongoing Updates

# Multimodal Emotion Recognition (Affect-Diff)

## Overview
This repository implements a multimodal emotion recognition system using text, audio, and video features, a variational bottleneck, and a diffusion generative prior. It includes explainability via causal graphs and counterfactual interventions.

## Setup
- Install dependencies: `pip install -r requirements.txt`
- Prepare CMU-MOSEI data in the `data/Data` directory.
- Run feature extraction scripts in `Data/preprocessors/` if needed.

## Training
- Edit `configs/config.yaml` as needed.
- Run training: `python train.py`

## Evaluation
- Run evaluation: `python eval.py`

## Structure
- `Data/`: Data modules, augmentations, preprocessors
- `models/`: Encoders, fusion, diffusion, causal graph
- `modules/`: PyTorch Lightning training logic
- `utils/`: Metrics, visualization, loss functions
- `scripts/`: Cluster and feature extraction scripts
- `tests/`: Unit tests

## Notes
- Ensure all import paths are case-sensitive and match your filesystem.
- For more details, see code comments and docstrings.
