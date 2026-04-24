"""Tests for the MoseiDataset and DataLoader integration."""

import pytest
import torch
from Data.mosei_datamodule import MoseiDataset


class TestMoseiDataset:
    """Tests for the CMU-MOSEI dataset wrapper."""

    def test_dataset_shapes(self):
        """Dataset serves padded sequences with correct dimensions."""
        num_samples, seq_len = 10, 50
        dummy_data = {
            "vision": torch.randn(num_samples, seq_len, 35),  # FAU 35-d
            "audio": torch.randn(num_samples, seq_len, 74),   # COVAREP 74-d
            "text": torch.randn(num_samples, seq_len, 300),   # GloVe 300-d
            "labels": torch.randint(0, 6, (num_samples,)),
        }

        dataset = MoseiDataset(dummy_data)
        assert len(dataset) == num_samples

        item = dataset[0]
        assert item["vision"].shape == (seq_len, 35)
        assert item["audio"].shape == (seq_len, 74)
        assert item["text"].shape == (seq_len, 300)
        assert isinstance(item["labels"].item(), int)

    def test_dataloader_batching(self):
        """DataLoader correctly stacks items into batches."""
        num_samples, seq_len = 32, 50
        dummy_data = {
            "vision": torch.randn(num_samples, seq_len, 35),
            "audio": torch.randn(num_samples, seq_len, 74),
            "text": torch.randn(num_samples, seq_len, 300),
            "labels": torch.randint(0, 6, (num_samples,)),
        }

        dataset = MoseiDataset(dummy_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

        batch = next(iter(loader))
        assert batch["vision"].shape == (8, seq_len, 35)
        assert batch["audio"].shape == (8, seq_len, 74)
        assert batch["text"].shape == (8, seq_len, 300)
        assert batch["labels"].shape == (8,)

    def test_single_sample(self):
        """Dataset handles single-sample edge case."""
        dummy_data = {
            "vision": torch.randn(1, 50, 35),
            "audio": torch.randn(1, 50, 74),
            "text": torch.randn(1, 50, 300),
            "labels": torch.tensor([3]),
        }
        dataset = MoseiDataset(dummy_data)
        assert len(dataset) == 1
        item = dataset[0]
        assert item["labels"].item() == 3
