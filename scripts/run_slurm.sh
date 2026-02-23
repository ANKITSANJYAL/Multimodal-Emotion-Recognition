#!/bin/bash
# SLURM run script for cluster execution
# Example usage: sbatch run_slurm.sh

#SBATCH --job-name=affect_diff
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load cuda/11.3
source ~/envs/affect_diff/bin/activate
python modules/affect_diff_module.py --config configs/config.yaml
