"""Affect-Diff evaluation entry point.

Runs standardized evaluation on the CMU-MOSEI test set with:
  - Standard metrics: Accuracy, Macro-F1, per-class F1
  - MOSEI metrics: MAE, Pearson Correlation, Acc-2, Acc-7
  - Causal Sensitivity: interventional effect size per modality
  - Dissonance detection: sarcasm/hidden emotion flagging
"""

import logging

import hydra
import torch
from omegaconf import DictConfig

from Data.mosei_datamodule import MoseiDataModule
from modules.affect_diff_module import AffectDiffModule
from utils.metrics import AffectiveCausalMetrics

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate the Affect-Diff model on CMU-MOSEI test set."""
    logger.info("Initializing Evaluation Pipeline for Affect-Diff...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    datamodule = MoseiDataModule(
        data_dir=cfg.data.data_dir,
        cache_dir=cfg.data.cache_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seq_len=cfg.data.seq_len,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 2. Load model
    checkpoint_path = cfg.eval.checkpoint_path
    model = AffectDiffModule.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # 3. Metrics
    metrics = AffectiveCausalMetrics(num_classes=cfg.model.num_classes, device=device)

    total_acc, total_f1 = 0.0, 0.0
    total_cs_audio, total_cs_video, total_cs_text = 0.0, 0.0, 0.0
    all_logits, all_labels = [], []
    num_batches = 0

    logger.info("Running Test Set & Counterfactual Interventions...")

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Factual forward pass
            logits = model(batch["text"], batch["audio"], batch["vision"])
            all_logits.append(logits)
            all_labels.append(batch["labels"])

            # Standard metrics
            std = metrics.compute_standard_metrics(logits, batch["labels"])
            total_acc += std["Accuracy"]
            total_f1 += std["Macro_F1"]

            # Causal Sensitivity per modality (uses DDIM healing internally)
            if cfg.model.use_diffusion and model.diffusion is not None:
                cs_audio = metrics.calculate_causal_sensitivity(
                    model, batch, target_modality="audio",
                    cfg_scale=cfg.eval.cfg_scale,
                )
                cs_video = metrics.calculate_causal_sensitivity(
                    model, batch, target_modality="video",
                    cfg_scale=cfg.eval.cfg_scale,
                )
                cs_text = metrics.calculate_causal_sensitivity(
                    model, batch, target_modality="text",
                    cfg_scale=cfg.eval.cfg_scale,
                )
                total_cs_audio += cs_audio
                total_cs_video += cs_video
                total_cs_text += cs_text

            num_batches += 1

    # 4. Aggregate metrics
    avg_acc = total_acc / max(num_batches, 1)
    avg_f1 = total_f1 / max(num_batches, 1)

    # Compute full-dataset MOSEI metrics (Acc-2, Acc-7, MAE, Corr, per-class F1, confusion)
    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)
    full_metrics = metrics.compute_full_metrics(all_logits_cat, all_labels_cat)

    # 5. Print results
    print("\n" + "=" * 60)
    print(f"{'AFFECT-DIFF CVPR 2026 EVALUATION RESULTS':^60}")
    print("=" * 60)
    print(f"  Dataset:                 CMU-MOSEI")
    print(f"  Accuracy (Acc-7):        {full_metrics['acc_7']:.4f}")
    print(f"  Accuracy (Acc-2):        {full_metrics['acc_2']:.4f}")
    print(f"  Macro F1-Score:          {full_metrics['macro_f1']:.4f}")
    print(f"  MAE:                     {full_metrics['mae']:.4f}")
    print(f"  Pearson Correlation:     {full_metrics['corr']:.4f}")
    print("-" * 60)
    print(f"  Per-Class F1:")
    for cls_name, f1_val in full_metrics["per_class_f1"].items():
        print(f"    {cls_name:12s}:  {f1_val:.4f}")
    print("-" * 60)

    if cfg.model.use_diffusion and model.diffusion is not None:
        avg_cs_audio = total_cs_audio / max(num_batches, 1)
        avg_cs_video = total_cs_video / max(num_batches, 1)
        avg_cs_text = total_cs_text / max(num_batches, 1)
        print(f"  Causal Sensitivity (Text):   {avg_cs_text:.4f}")
        print(f"  Causal Sensitivity (Audio):  {avg_cs_audio:.4f}")
        print(f"  Causal Sensitivity (Video):  {avg_cs_video:.4f}")

    print("=" * 60)
    print("Interpretation: Higher CS = model relied heavily on that modality.")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
