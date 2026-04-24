#!/bin/bash
# SLURM run script for Affect-Diff cluster training
# Usage: sbatch scripts/run_slurm.sh

#SBATCH --job-name=affect_diff
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gpus=2
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load cuda/11.8
source ~/envs/affect_diff/bin/activate

# Ensure logs directory exists
mkdir -p logs

# Run training with Hydra config
python train.py
