import logging
import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Increment when preprocessing logic changes to invalidate old caches.
_CACHE_VERSION = "v2"


class MoseiDataset(Dataset):
    """
    Standard PyTorch Dataset for aligned CMU-MOSEI data.
    Expects pre-aligned, padded, and Z-normalized tensors.
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

    Changes vs v1:
      - REPAIR instead of PRUNE mismatched interval/feature arrays (recovers ~85% of
        previously discarded segments by truncating to min(len(intervals), len(features))).
      - DROP zero-intensity segments instead of mislabelling them as class-0 (Happy).
      - Z-NORMALIZE each modality using training-split statistics only (no leakage).
      - FIXED 70/15/15 split (was incorrectly 70/10/20).
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        seq_len: int = 50,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.cache_path = os.path.join(
            self.cache_dir, f"mosei_aligned_seq{self.seq_len}_{_CACHE_VERSION}.pt"
        )
        # Normalization stats populated in setup(); exposed for inference code.
        self.norm_stats: dict = {}

    # ------------------------------------------------------------------
    # PREPARE DATA  (rank-0 only; aligns .csd files and caches tensors)
    # ------------------------------------------------------------------

    def prepare_data(self):
        if os.path.exists(self.cache_path):
            logger.info("Found cached aligned data at %s. Skipping alignment.", self.cache_path)
            return

        logger.info("Loading raw .csd files via CMU mmsdk...")
        from mmsdk import mmdatasdk  # lazy import — only needed during alignment

        dataset_recipe = {
            'vision': os.path.join(self.data_dir, 'CMU_MOSEI_VisualFacet42.csd'),
            'audio':  os.path.join(self.data_dir, 'CMU_MOSEI_COVAREP.csd'),
            'text':   os.path.join(self.data_dir, 'CMU_MOSEI_TimestampedWordVectors.csd'),
            'labels': os.path.join(self.data_dir, 'CMU_MOSEI_Labels.csd'),
        }

        dataset = mmdatasdk.mmdataset(dataset_recipe)

        # --- 1. Unify video IDs across all four sequences --------------------
        logger.info("Unifying dataset indices...")
        dataset.unify()

        # --- 2. REPAIR mismatched entries (not prune) -----------------------
        # Previously, any segment where len(intervals) != len(features) was
        # dropped entirely.  This discarded ~85% of the corpus.  We instead
        # truncate both arrays to min(len(intervals), len(features)), keeping
        # the segment valid.
        logger.info("Repairing interval/feature mismatches (truncate to min length)...")
        repaired = 0
        for seq_key in dataset.computational_sequences.keys():
            seq_data = dataset.computational_sequences[seq_key].data
            for vid in list(seq_data.keys()):
                d = seq_data[vid]
                n_int = len(d['intervals'])
                n_feat = len(d['features'])
                if n_int != n_feat:
                    min_len = min(n_int, n_feat)
                    if min_len == 0:
                        # Zero-length after repair → still unusable; drop.
                        del seq_data[vid]
                    else:
                        seq_data[vid]['intervals'] = seq_data[vid]['intervals'][:min_len]
                        seq_data[vid]['features']  = seq_data[vid]['features'][:min_len]
                        repaired += 1

        logger.info("Repaired %d mismatched entries.", repaired)

        # --- 3. Extract and pad sequences -----------------------------------
        logger.info("Extracting tensors with sequence padding (seq_len=%d)...", self.seq_len)
        aligned_data = self._extract_tensors(dataset)

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(aligned_data, self.cache_path)
        logger.info(
            "Data cached to %s  (%d samples)",
            self.cache_path, aligned_data['labels'].shape[0],
        )

    # ------------------------------------------------------------------
    # SETUP  (every GPU; loads cache, splits, normalizes)
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        """
        Loads cached tensors, splits into train/val/test (70/15/15),
        then Z-normalises each modality using training-split statistics.

        The split uses a fixed random permutation (seed=42) so all GPU
        ranks produce identical splits and results are reproducible across
        resume runs.

        Normalization is done AFTER splitting to avoid leaking val/test
        statistics into the encoder.  Stats are stored in self.norm_stats
        for use by the inference / eval pipeline.
        """
        data_dict = torch.load(self.cache_path, weights_only=False)
        total_samples = data_dict['labels'].shape[0]
        logger.info("Loaded %d samples from cache.", total_samples)

        # Fixed-seed permutation — identical across all ranks and runs.
        rng = torch.Generator()
        rng.manual_seed(42)
        indices = torch.randperm(total_samples, generator=rng).tolist()

        train_bound = int(0.70 * total_samples)
        val_bound   = int(0.85 * total_samples)   # FIX: 70/15/15 (was 70/10/20)

        train_idx = indices[:train_bound]
        val_idx   = indices[train_bound:val_bound]
        test_idx  = indices[val_bound:]

        logger.info(
            "Split: train=%d  val=%d  test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        # --- Z-normalize using training-split statistics only ---------------
        train_raw = self._slice_dict(data_dict, train_idx)
        t_mean, t_std = self._feature_stats(train_raw['text'])
        a_mean, a_std = self._feature_stats(train_raw['audio'])
        v_mean, v_std = self._feature_stats(train_raw['vision'])

        self.norm_stats = {
            'text':   (t_mean, t_std),
            'audio':  (a_mean, a_std),
            'vision': (v_mean, v_std),
        }

        def _apply_norm(d):
            d = {k: v.clone() for k, v in d.items()}
            d['text']   = (d['text']   - t_mean) / t_std
            d['audio']  = (d['audio']  - a_mean) / a_std
            d['vision'] = (d['vision'] - v_mean) / v_std
            return d

        self.train_dataset = MoseiDataset(_apply_norm(self._slice_dict(data_dict, train_idx)))
        self.val_dataset   = MoseiDataset(_apply_norm(self._slice_dict(data_dict, val_idx)))
        self.test_dataset  = MoseiDataset(_apply_norm(self._slice_dict(data_dict, test_idx)))

        # --- Class weights for cross-entropy (inverse frequency, training split only) ---
        # MOSEI is heavily imbalanced: Happy ~66%, Fear ~2%.  Without weighting,
        # the model converges to predicting Happy for everything.
        train_labels = self.train_dataset.labels
        num_classes  = int(train_labels.max().item()) + 1
        counts = torch.bincount(train_labels, minlength=num_classes).float()
        # weight_c = N_train / (num_classes * count_c)  — standard balanced weighting
        self.class_weights = train_labels.shape[0] / (num_classes * counts.clamp(min=1))
        logger.info(
            "Class weights (balanced): %s",
            {i: f"{w:.3f}" for i, w in enumerate(self.class_weights.tolist())},
        )

    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_stats(tensor: torch.Tensor):
        """
        Compute per-feature mean and std from (N, L, D) tensor.
        Returns (mean, std) each of shape (1, 1, D) for broadcasting.
        Std is clamped to ≥1e-6 to avoid divide-by-zero on constant features.
        """
        mean = tensor.mean(dim=(0, 1), keepdim=True)       # (1, 1, D)
        std  = tensor.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        return mean, std

    def _extract_tensors(self, dataset) -> dict:
        """
        Iterates through the aligned mmdataset and converts every valid
        segment to padded PyTorch tensors.

        Label construction (FIX):
          - Emotion intensities are indices 1-6 of the label vector:
            [sentiment, happy, sad, angry, fear, disgust, surprise]
          - argmax gives the dominant emotion (0=happy … 5=surprise).
          - Segments where ALL intensities are zero are DROPPED (they are
            genuinely unlabelled neutral clips).  Previously they were
            silently assigned to class-0 (Happy), poisoning that class.
        """
        vision_list, audio_list, text_list, label_list = [], [], [], []
        skipped_missing = 0
        skipped_neutral = 0

        for vid in dataset.computational_sequences['labels'].data.keys():
            # All four modalities must be present.
            if (vid not in dataset.computational_sequences['vision'].data or
                    vid not in dataset.computational_sequences['audio'].data or
                    vid not in dataset.computational_sequences['text'].data):
                skipped_missing += 1
                continue

            v = dataset.computational_sequences['vision'].data[vid]['features']
            a = dataset.computational_sequences['audio'].data[vid]['features']
            t = dataset.computational_sequences['text'].data[vid]['features']
            l = dataset.computational_sequences['labels'].data[vid]['features']

            if len(l) == 0:
                skipped_missing += 1
                continue

            # MOSEI label layout: [sentiment, happy, sad, angry, fear, disgust, surprise]
            intensities = l[0][1:7]
            if np.sum(intensities) == 0:
                # Drop truly un-labelled neutral segments rather than mis-assigning
                # them to class-0 (Happy).
                skipped_neutral += 1
                continue

            l_val = int(np.argmax(intensities))   # 0=happy … 5=surprise

            v_pad = self._pad_sequence(v, self.seq_len)
            a_pad = self._pad_sequence(a, self.seq_len)
            t_pad = self._pad_sequence(t, self.seq_len)

            vision_list.append(v_pad)
            audio_list.append(a_pad)
            text_list.append(t_pad)
            label_list.append(l_val)

        logger.info(
            "Extraction complete: %d kept | %d missing-modality skipped | "
            "%d zero-intensity (neutral) skipped",
            len(label_list), skipped_missing, skipped_neutral,
        )

        # Sanitize NaN/Inf by zero-imputation (treat missing frames as uninformative).
        v_tensor = torch.nan_to_num(torch.tensor(np.stack(vision_list), dtype=torch.float32))
        a_tensor = torch.nan_to_num(torch.tensor(np.stack(audio_list),  dtype=torch.float32))
        t_tensor = torch.nan_to_num(torch.tensor(np.stack(text_list),   dtype=torch.float32))
        l_tensor = torch.tensor(label_list, dtype=torch.long)

        return {'vision': v_tensor, 'audio': a_tensor, 'text': t_tensor, 'labels': l_tensor}

    @staticmethod
    def _pad_sequence(seq, target_len: int) -> np.ndarray:
        """Zero-pad or truncate a feature sequence to target_len time steps."""
        seq = np.array(seq)
        if seq.ndim == 1:
            seq = seq[np.newaxis, :]         # (1, D)
        current_len = seq.shape[0]
        if current_len >= target_len:
            return seq[:target_len, :]
        pad = np.zeros((target_len - current_len, seq.shape[1]), dtype=seq.dtype)
        return np.vstack((seq, pad))

    @staticmethod
    def _slice_dict(data_dict: dict, indices) -> dict:
        """Slice every tensor in data_dict by index list."""
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return {k: v[idx_tensor] for k, v in data_dict.items()}
