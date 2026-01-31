#!/bin/bash
# Quick start script for training

echo "========================================"
echo "Trimodal Emotion Recognition Training"
echo "========================================"

# Check if preprocessing has been done
if [ ! -f "data/metadata.csv" ]; then
    echo "Error: metadata.csv not found!"
    echo "Please run preprocessing first: python preprocess.py"
    exit 1
fi

# Default parameters
NUM_GPUS=2
BATCH_SIZE=16
EPOCHS=20
LR=1e-4

echo ""
echo "Training Configuration:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Batch Size per GPU: $BATCH_SIZE"
echo "  - Global Batch Size: $((NUM_GPUS * BATCH_SIZE))"
echo "  - Epochs: $EPOCHS"
echo "  - Learning Rate: $LR"
echo ""

# Create output directory
mkdir -p outputs

# Launch training
echo "Launching distributed training..."
torchrun --nproc_per_node=$NUM_GPUS src/train_ddp.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --output_dir outputs

echo ""
echo "Training complete! Check outputs/ for results."
