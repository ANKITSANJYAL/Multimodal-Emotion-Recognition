import pytest
import torch
from Data.mosei_datamodule import MoseiDataset

def test_mosei_dataset_shapes():
    """
    Tests that the dataset correctly serves padded sequences
    with the expected temporal and feature dimensions.
    """
    # Create dummy aligned data mimicking our caching script
    num_samples = 10
    seq_len = 50
    dummy_data = {
        'vision': torch.randn(num_samples, seq_len, 42),   # VisualFacet42
        'audio': torch.randn(num_samples, seq_len, 74),    # COVAREP
        'text': torch.randn(num_samples, seq_len, 300),    # GloVe
        'labels': torch.randint(0, 6, (num_samples,))      # 6 Emotion Classes
    }

    dataset = MoseiDataset(dummy_data)

    # Check length
    assert len(dataset) == num_samples

    # Check individual item extraction and shapes
    item = dataset[0]
    assert item['vision'].shape == (seq_len, 42)
    assert item['audio'].shape == (seq_len, 74)
    assert item['text'].shape == (seq_len, 300)
    assert isinstance(item['labels'].item(), int)

def test_dataloader_batching():
    """Ensures PyTorch DataLoader stacks the items correctly into batches."""
    num_samples = 32
    seq_len = 50
    dummy_data = {
        'vision': torch.randn(num_samples, seq_len, 42),
        'audio': torch.randn(num_samples, seq_len, 74),
        'text': torch.randn(num_samples, seq_len, 300),
        'labels': torch.randint(0, 6, (num_samples,))
    }

    dataset = MoseiDataset(dummy_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

    batch = next(iter(loader))
    assert batch['vision'].shape == (8, seq_len, 42)
    assert batch['labels'].shape == (8,)
