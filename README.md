# Robust Trimodal Emotion Recognition (RAVDESS)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project implements a **State-of-the-Art (SOTA) Trimodal Emotion Recognition System** using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system combines three modalities:
- **Audio**: Processed using Wav2Vec2 transformer
- **Video**: Analyzed using VideoMAE (Masked Autoencoder for Video)
- **Text**: Extracted via ASR and processed with DistilRoBERTa

The architecture employs a late-fusion transformer approach with multi-GPU training support via PyTorch Distributed Data Parallel (DDP).

## Key Features

- ✅ **All-Transformer Architecture**: Uses pre-trained HuggingFace transformers for all three modalities
- ✅ **Multi-GPU Training**: Distributed Data Parallel (DDP) for efficient training on 2x NVIDIA Tesla V100 GPUs
- ✅ **Robustness Testing**: Supports audio noise injection and video frame occlusion augmentation
- ✅ **Subject-Independent Split**: Training on actors 1-20, validation/test on actors 21-24
- ✅ **8 Emotion Classes**: neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Architecture

### Model Components

| Modality | Model | Hidden Dim | Parameters |
|----------|-------|-----------|-----------|
| Audio | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | 768 | 95M |
| Video | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | 768 | 86M |
| Text | [distilroberta-base](https://huggingface.co/distilroberta-base) | 768 | 82M |

**Fusion Strategy**: Late fusion with concatenation (768 × 3 = 2304 dims) followed by an MLP classifier.

### Training Strategy

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: Cosine Annealing
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 16 per GPU (Global batch size = 32)
- **Augmentation**: 
  - Audio: Gaussian noise injection (50% probability)
  - Video: Frame dropout (50% probability, 30% of frames)

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 2x NVIDIA GPUs (recommended: Tesla V100 with 32GB VRAM)

### Setup

```bash
# Clone the repository
git clone https://github.com/ANKITSANJYAL/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project uses the [RAVDESS dataset](https://zenodo.org/record/1188976) which contains:
- 24 professional actors (12 male, 12 female)
- 8 emotional expressions
- Video + Audio recordings

Dataset structure after preprocessing:
```
data/
├── videos/
│   ├── Actor_01/
│   │   ├── 01-01-01-01-01-01-01.mp4
│   │   └── ...
│   └── Actor_24/
├── audio_extracted/
│   ├── Actor_01/
│   │   ├── 01-01-01-01-01-01-01.wav
│   │   └── ...
│   └── Actor_24/
└── metadata.csv
```

## Usage

### Step 1: Data Preprocessing

Extract audio from videos and generate metadata:

```bash
python preprocess.py
```

This will:
- Extract audio from all video files (16kHz, mono)
- Parse filenames to extract emotion labels
- Create `data/metadata.csv` with train/val/test splits
- Apply subject-independent splitting strategy

### Step 2: Training

Run distributed training on 2 GPUs:

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 src/train_ddp.py \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-4 \
    --output_dir outputs

# Alternative: Specify custom paths
torchrun --nproc_per_node=2 src/train_ddp.py \
    --base_dir /path/to/project \
    --metadata_csv /path/to/metadata.csv \
    --batch_size 16 \
    --epochs 20
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 16 | Batch size per GPU |
| `--epochs` | 20 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 0.01 | Weight decay for AdamW |
| `--gradient_checkpointing` | False | Enable gradient checkpointing to save VRAM |
| `--num_workers` | 4 | Number of dataloader workers |
| `--output_dir` | outputs | Directory for checkpoints and logs |

### Step 3: Evaluation

After training, the following outputs are saved in `outputs/`:

- `best_model.pth`: Best model checkpoint
- `confusion_matrix.png`: Confusion matrix visualization
- `training_curves.png`: Loss and accuracy curves
- `classification_report.txt`: Detailed per-class metrics

## Results

Expected performance metrics:

| Split | Samples | Accuracy |
|-------|---------|----------|
| Train | ~1200 | 85%+ |
| Val | ~200 | 75%+ |
| Test | ~200 | 70%+ |

*Note: Actual results may vary depending on augmentation and training configuration.*

## Project Structure

```
Multimodal-Emotion-Recognition/
├── data/                          # Dataset directory
│   ├── videos/                    # Extracted video files
│   ├── audio_extracted/           # Extracted audio files
│   └── metadata.csv              # Dataset metadata
├── src/                          # Source code
│   ├── dataset.py                # PyTorch Dataset implementation
│   ├── model.py                  # Trimodal architecture
│   ├── train_ddp.py              # DDP training script
│   └── utils.py                  # Helper functions
├── outputs/                      # Training outputs
│   ├── best_model.pth           # Best model checkpoint
│   ├── confusion_matrix.png     # Visualization
│   └── classification_report.txt
├── preprocess.py                 # Data preprocessing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Multi-GPU Training Details

This project uses **DistributedDataParallel (DDP)** instead of DataParallel for optimal performance:

### Why DDP?
- **Faster**: Separate processes per GPU (avoids Python GIL)
- **More Efficient**: Overlaps computation and communication
- **Scalable**: Works across multiple nodes

### How it Works
1. **Process Spawning**: `torchrun` spawns one process per GPU
2. **Data Partitioning**: Each GPU receives different data slices via `DistributedSampler`
3. **Gradient Synchronization**: Gradients are averaged across GPUs using all-reduce
4. **Model Replication**: Each process has its own model replica

### Memory Efficiency
- **Gradient Checkpointing**: Enable with `--gradient_checkpointing` flag
- **Mixed Precision**: Can be added for further optimization (future work)

## Research Applications

This codebase is designed for research on:
- **Robustness**: Test model performance under noisy conditions
- **Ablation Studies**: Evaluate individual modality contributions
- **Feature Analysis**: Extract and visualize learned representations
- **Transfer Learning**: Fine-tune on custom emotion datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ravdess-trimodal-2026,
  author = {Your Name},
  title = {Robust Trimodal Emotion Recognition using RAVDESS},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ANKITSANJYAL/Multimodal-Emotion-Recognition}
}
```

### RAVDESS Dataset Citation

```bibtex
@article{livingstone2018ryerson,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PloS one},
  volume={13},
  number={5},
  pages={e0196391},
  year={2018},
  publisher={Public Library of Science}
}
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `--batch_size`
- Enable `--gradient_checkpointing`
- Reduce number of frames or audio duration in `dataset.py`

**2. DDP Initialization Fails**
- Check that `MASTER_ADDR` and `MASTER_PORT` are set correctly
- Ensure all GPUs are accessible
- Verify NCCL backend is available

**3. Slow Data Loading**
- Increase `--num_workers`
- Check disk I/O performance
- Consider caching preprocessed data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RAVDESS dataset by Livingstone & Russo
- HuggingFace Transformers library
- PyTorch team for DDP implementation

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: your.email@example.com
- GitHub: [@ANKITSANJYAL](https://github.com/ANKITSANJYAL)

---

**Built with ❤️ using PyTorch and Transformers**