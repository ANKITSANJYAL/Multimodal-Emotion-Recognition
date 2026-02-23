import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from Data.mosei_datamodule import MoseiDataModule
from modules.affect_diff_module import AffectDiffModule
from utils.metrics import AffectiveCausalMetrics

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def evaluate(cfg: DictConfig):
    print("Initializing Evaluation Pipeline for Affect-Diff...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    datamodule = MoseiDataModule(
        data_dir=cfg.data.data_dir,
        cache_dir=cfg.data.cache_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 2. Load Trained Model Checkpoint
    checkpoint_path = cfg.eval.checkpoint_path # e.g., "checkpoints/best_model.ckpt"
    model = AffectDiffModule.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # 3. Setup Metrics
    metrics = AffectiveCausalMetrics(num_classes=cfg.model.num_classes, device=device)

    total_acc, total_f1 = 0.0, 0.0
    total_cs_audio, total_cs_video = 0.0, 0.0
    num_batches = 0

    print("\nRunning Test Set & Counterfactual Interventions...")

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Factual Forward Pass
            logits = model(batch['text'], batch['audio'], batch['vision'])

            # Standard Metrics
            std_metrics = metrics.compute_standard_metrics(logits, batch['labels'])
            total_acc += std_metrics['Accuracy']
            total_f1 += std_metrics['Macro_F1']

            # Causal Interventions (The CVPR Novelty)
            cs_audio = metrics.calculate_causal_sensitivity(model, batch, target_modality='audio')
            cs_video = metrics.calculate_causal_sensitivity(model, batch, target_modality='video')

            total_cs_audio += cs_audio
            total_cs_video += cs_video
            num_batches += 1

    # 4. Average and Print CVPR-Ready Results
    avg_acc = total_acc / num_batches
    avg_f1 = total_f1 / num_batches
    avg_cs_audio = total_cs_audio / num_batches
    avg_cs_video = total_cs_video / num_batches

    print("\n" + "="*50)
    print(f"{'CVPR EVALUATION RESULTS':^50}")
    print("="*50)
    print(f"Dataset:              CMU-MOSEI")
    print(f"Accuracy:             {avg_acc:.4f}")
    print(f"Macro F1-Score:       {avg_f1:.4f}")
    print("-" * 50)
    print(f"Explainability Metrics (Causal Sensitivity)")
    print(f"CS_Audio Ablation:    {avg_cs_audio:.4f}")
    print(f"CS_Video Ablation:    {avg_cs_video:.4f}")
    print("="*50)
    print("Interpretation: A higher CS score means the model heavily")
    print("relied on that modality to formulate its factual prediction.")

if __name__ == "__main__":
    evaluate()
