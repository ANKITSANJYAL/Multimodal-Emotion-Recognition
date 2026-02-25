import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
# NOTE: Importing the heavy `mmsdk` package at module import time causes an
# ImportError on systems without it installed. We import it lazily inside
# `prepare_data()` only when alignment from raw .csd files is required. This
# allows the module to be imported when a cached aligned tensor exists.

class MoseiDataset(Dataset):
    """
    Standard PyTorch Dataset for aligned CMU-MOSEI data.
    Expects pre-aligned and padded tensors.
    """
    def __init__(self, data_dict):
        super().__init__()
        self.vision = data_dict['vision']
        self.audio = data_dict['audio']
        self.text = data_dict['text']
        self.labels = data_dict['labels']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            'vision': self.vision[idx],
            'audio': self.audio[idx],
            'text': self.text[idx],
            'labels': self.labels[idx]
        }

class MoseiDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CMU-MOSEI.
    Handles the heavy lifting of aligning .csd files using mmsdk
    and caching them for high-throughput GPU training.
    """
    def __init__(self, data_dir: str, cache_dir: str, batch_size: int = 32, num_workers: int = 8, seq_len: int = 50):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.cache_path = os.path.join(self.cache_dir, f'mosei_aligned_seq{self.seq_len}.pt')

    def prepare_data(self):
        """
        Executed only on 1 GPU. Aligns .csd files and caches them.
        """
        if os.path.exists(self.cache_path):
            print(f"Found cached aligned data at {self.cache_path}. Skipping alignment.")
            return

        print("Loading raw .csd files via CMU mmsdk...")
        dataset_recipe = {
            'vision': os.path.join(self.data_dir, 'CMU_MOSEI_VisualFacet42.csd'),
            'audio': os.path.join(self.data_dir, 'CMU_MOSEI_COVAREP.csd'),
            'text': os.path.join(self.data_dir, 'CMU_MOSEI_TimestampedWordVectors.csd'),
            'labels': os.path.join(self.data_dir, 'CMU_MOSEI_Labels.csd')
        }

        # Lazy import: try to find a callable constructor for an mmdataset-like
        # object in the installed mmsdk package. Provide a helpful error if it's
        # not installed or has an unexpected API.
        try:
            from mmsdk import mmdatasdk
            mmdataset = mmdatasdk.mmdataset
        except (ImportError, AttributeError):
            raise ImportError(
                "Could not locate 'mmdataset' in the 'mmsdk' package. \n"
                "Please ensure mmsdk is installed and accessible in the 'emotion_rec' environment."
            )

        dataset = mmdataset(dataset_recipe)

        # Align all modalities to the label timestamps
        print("Aligning modalities to labels (this may take a while)...")
        dataset.align('labels', collapse_functions=[self._avg_collapse, self._avg_collapse, self._avg_collapse])

        # Extract and pad sequences
        print("Extracting tensors and applying sequence padding...")
        aligned_data = self._extract_tensors(dataset)

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(aligned_data, self.cache_path)
        print(f"Data successfully cached to {self.cache_path}")

    def setup(self, stage=None):
        """
        Executed on every GPU. Loads the cached tensors and creates train/val/test splits.
        """
        # Load cached tensors
        data_dict = torch.load(self.cache_path)

        # CMU-MOSEI standard split ratios (approx: 70% train, 10% val, 20% test)
        total_samples = data_dict['labels'].shape[0]
        indices = torch.randperm(total_samples).tolist()

        train_bound = int(0.7 * total_samples)
        val_bound = int(0.8 * total_samples)

        train_idx = indices[:train_bound]
        val_idx = indices[train_bound:val_bound]
        test_idx = indices[val_bound:]

        self.train_dataset = MoseiDataset(self._slice_dict(data_dict, train_idx))
        self.val_dataset = MoseiDataset(self._slice_dict(data_dict, val_idx))
        self.test_dataset = MoseiDataset(self._slice_dict(data_dict, test_idx))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    # --- Helper Methods ---

    def _avg_collapse(self, intervals, features):
        """Collapses multiple high-frequency frames into a single vector by averaging."""
        if features.shape[0] == 0:
            return np.zeros((1, features.shape[1]))
        return np.mean(features, axis=0, keepdims=True)

    def _extract_tensors(self, dataset):
        """Iterates through the aligned mmdataset and converts to padded PyTorch tensors."""
        vision_list, audio_list, text_list, label_list = [], [], [], []

        # The aligned dataset keys are video IDs
        for vid in dataset.computational_sequences['labels'].data.keys():
            v = dataset.computational_sequences['vision'].data[vid]['features']
            a = dataset.computational_sequences['audio'].data[vid]['features']
            t = dataset.computational_sequences['text'].data[vid]['features']
            l = dataset.computational_sequences['labels'].data[vid]['features']

            # Truncate or Pad to self.seq_len
            v_pad = self._pad_sequence(v, self.seq_len)
            a_pad = self._pad_sequence(a, self.seq_len)
            t_pad = self._pad_sequence(t, self.seq_len)

            # Correct MoSEI Header: [sentiment, happy, sad, angry, fear, disgust, surprise]
            # Primary emotion is the argmax of the 6 intensity scores (indices 1-6)
            if len(l) > 0:
                intensities = l[0][1:7] # Extract indices 1 to 6
                # If all intensities are zero, default to a neutral category or handle
                # But for MoSEI, we'll take the max intensity as the label
                l_val = np.argmax(intensities) if np.sum(intensities) > 0 else 0
            else:
                l_val = 0

            vision_list.append(v_pad)
            audio_list.append(a_pad)
            text_list.append(t_pad)
            label_list.append(l_val)

        # 1. Convert to tensors with NaN-to-Zero robustness
        v_tensor = torch.nan_to_num(torch.tensor(np.stack(vision_list), dtype=torch.float32))
        a_tensor = torch.nan_to_num(torch.tensor(np.stack(audio_list), dtype=torch.float32))
        t_tensor = torch.nan_to_num(torch.tensor(np.stack(text_list), dtype=torch.float32))
        l_tensor = torch.tensor(label_list, dtype=torch.long)

        # 2. Research-Grade Stability: Global Modal-wise Normalization
        # This prevents gradient explosions from unnormalized features hitting the VAE
        def normalize(tensor):
            mask = (tensor.abs().sum(dim=-1) > 0)
            if mask.sum() == 0: return tensor
            mean = tensor[mask].mean()
            std = tensor[mask].std() + 1e-8
            return (tensor - mean) / std

        v_tensor = normalize(v_tensor)
        a_tensor = normalize(a_tensor)
        t_tensor = normalize(t_tensor)

        return {
            'vision': v_tensor,
            'audio': a_tensor,
            'text': t_tensor,
            'labels': l_tensor
        }

    def _pad_sequence(self, seq, target_len):
        """Pads or truncates a sequence to the target length."""
        seq = np.array(seq)
        if len(seq.shape) == 1:
            seq = np.expand_dims(seq, axis=0)

        current_len = seq.shape[0]
        if current_len >= target_len:
            return seq[:target_len, :]
        else:
            pad_len = target_len - current_len
            pad_matrix = np.zeros((pad_len, seq.shape[1]))
            return np.vstack((seq, pad_matrix))

    def _slice_dict(self, data_dict, indices):
        """Helper to slice the data dictionary using a list of indices."""
        return {k: v[indices] for k, v in data_dict.items()}
