"""Affect-Diff evaluation metrics: standard, MOSEI benchmark, and causal.

Implements:
  - Standard: Accuracy, Macro-F1
  - CMU-MOSEI benchmark: MAE, Pearson Correlation, Acc-2 (binary), Acc-7
  - Per-class F1 and confusion matrix
  - Causal Sensitivity (interventional effect size)
  - Dissonance detection (sarcasm proxy)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, Accuracy

logger = logging.getLogger(__name__)


class AffectiveCausalMetrics:
    """
    Pipeline role: EVALUATION METRICS (standard + causal)

    Computes:
      1. Standard metrics  — Accuracy, Macro-F1  (leaderboard numbers)
      2. Causal Sensitivity (CS) — interventional effect size via diffusion healing
      3. Dissonance detection  — flags sarcasm / hidden emotion

    All methods expect the full AffectDiffModule as `model` so they can
    access both model.bottleneck and model.diffusion.
    """
    def __init__(self, num_classes=6, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # ──────────────────────────────────────────────────────────────────────
    # STANDARD METRICS
    # ──────────────────────────────────────────────────────────────────────

    def compute_standard_metrics(self, logits, labels):
        """
        Input:  logits (B, num_classes)
                labels (B,)  long
        Output: dict {"Accuracy": float, "Macro_F1": float}
        """
        preds = torch.argmax(logits, dim=1)
        acc = self.acc_metric(preds, labels)
        f1  = self.f1_metric(preds, labels)
        return {"Accuracy": acc.item(), "Macro_F1": f1.item()}

    # ──────────────────────────────────────────────────────────────────────
    # CAUSAL SENSITIVITY
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def calculate_causal_sensitivity(self, model, batch, target_modality='audio', cfg_scale=3.0):
        """
        Computes CS_m = || P(y|x_factual) - P(y|do(x_m=0), healed) ||_1

        Steps:
          1. Factual forward pass → P_factual
          2. Zero the target modality (do-intervention)
          3. Encode the ablated input → z_ablated
          4. Partially noise z_ablated to t=T/2 (mid-noise level)
          5. CFG-guided reverse diffusion from t=T/2 → z_CF
             (the diffusion model "heals" the missing modality using
              knowledge from the remaining modalities + label conditioning)
          6. Classify z_CF → P_CF
          7. CS = mean over batch of ||P_factual - P_CF||_1

        Input:  model           AffectDiffModule
                batch           dict with keys 'text','audio','vision','labels'
                target_modality str  in {'audio', 'video', 'text'}
                cfg_scale       float  guidance strength
        Output: float — Causal Sensitivity scalar in [0, 2]
        """
        model.eval()
        text   = batch['text'].to(self.device)    # (B, L, 300)
        audio  = batch['audio'].to(self.device)   # (B, L,  74)
        video  = batch['vision'].to(self.device)  # (B, L,  35)
        labels = batch['labels'].to(self.device).long()  # (B,)

        # ── 1. Factual pass ───────────────────────────────────────────────
        # bottleneck returns (z_perm, mu, logvar, adj_matrix)
        z_factual, _, _, adj_matrix = model.bottleneck(text, audio, video)
        # z_factual: (B, d_z, L) — need (B, L, d_z) for pooling
        z_factual_seq = z_factual.permute(0, 2, 1)          # (B, L, d_z)
        z_factual_pooled = model._pool_features(z_factual_seq)  # (B, d_z)
        p_factual = F.softmax(model.classifier(z_factual_pooled), dim=1)  # (B, K)

        causal_influence = torch.clamp(adj_matrix.sum(dim=1), 0.0, 5.0)  # (B, 3)

        # ── 2. Intervention: do(x_m = 0) ─────────────────────────────────
        if target_modality == 'audio':
            z_ablated, _, _, _ = model.bottleneck(text, torch.zeros_like(audio), video)
        elif target_modality == 'video':
            z_ablated, _, _, _ = model.bottleneck(text, audio, torch.zeros_like(video))
        else:  # text
            z_ablated, _, _, _ = model.bottleneck(torch.zeros_like(text), audio, video)
        # z_ablated: (B, d_z, L)

        # ── 3. Generative healing via partial reverse diffusion ───────────
        # Noise z_ablated to t=T/2 (half-way destroyed), then denoise back
        # using CFG conditioned on the ground-truth label.
        # This samples from p̃(z | y, remaining_modalities) — a principled
        # counterfactual that is tighter than naive zero-ablation.
        b = text.shape[0]
        t_mid = torch.full((b,), model.diffusion.timesteps // 2,
                           device=self.device, dtype=torch.long)
        z_noisy = model.diffusion.q_sample(z_ablated, t_mid)   # (B, d_z, L)

        z_cf = z_noisy
        for i in reversed(range(model.diffusion.timesteps // 2)):
            t_curr = torch.full((b,), i, device=self.device, dtype=torch.long)
            z_cf = model.diffusion.p_sample(
                z_cf, t_curr, i,
                label=labels,
                causal_weights=causal_influence,
                cfg_scale=cfg_scale
            )
        # z_cf: (B, d_z, L)

        # ── 4. Counterfactual prediction ──────────────────────────────────
        z_cf_seq    = z_cf.permute(0, 2, 1)               # (B, L, d_z)
        z_cf_pooled = model._pool_features(z_cf_seq)       # (B, d_z)
        p_cf = F.softmax(model.classifier(z_cf_pooled), dim=1)  # (B, K)

        # ── 5. L1 distance = 2 * Total Variation ─────────────────────────
        causal_sensitivity = torch.norm(p_factual - p_cf, p=1, dim=1).mean()
        return causal_sensitivity.item()

    # ──────────────────────────────────────────────────────────────────────
    # DISSONANCE DETECTION
    # ──────────────────────────────────────────────────────────────────────

    def detect_dissonance(self, logits_factual, logits_text_only, threshold=0.4):
        """
        Flags samples where text-only prediction strongly disagrees with
        the full multimodal prediction — a proxy for sarcasm or suppressed emotion.

        Input:  logits_factual   (B, K)  — full multimodal logits
                logits_text_only (B, K)  — text-only logits (audio=0, video=0)
                threshold        float   — L1 distance threshold
        Output: (B,) bool tensor — True where ||P_full - P_text||_1 > threshold
        """
        p_factual = F.softmax(logits_factual,   dim=1)
        p_text    = F.softmax(logits_text_only, dim=1)
        divergence = torch.norm(p_factual - p_text, p=1, dim=1)
        return divergence > threshold

    # ──────────────────────────────────────────────────────────────────────
    # FULL EVALUATION METRICS (MOSEI BENCHMARK)
    # ──────────────────────────────────────────────────────────────────────

    def compute_full_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        regression_preds: Optional[torch.Tensor] = None,
        regression_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics for CMU-MOSEI.

        Includes standard classification metrics plus MOSEI benchmark metrics.

        Args:
            logits: Model output logits (N, num_classes).
            labels: Ground-truth class labels (N,) long.
            regression_preds: Optional continuous predictions for MAE/Corr.
            regression_targets: Optional continuous targets for MAE/Corr.

        Returns:
            Dictionary with all metrics.
        """
        preds = torch.argmax(logits, dim=1)
        labels_long = labels.long()
        results: Dict[str, Any] = {}

        # ── Standard Accuracy & Macro-F1 ──────────────────────────────────
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        f1_macro = F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device)
        results["accuracy"] = acc_metric(preds, labels_long).item()
        results["macro_f1"] = f1_macro(preds, labels_long).item()

        # ── Per-Class F1 ──────────────────────────────────────────────────
        f1_per_class = F1Score(
            task="multiclass", num_classes=self.num_classes, average="none",
        ).to(self.device)
        per_class_f1 = f1_per_class(preds, labels_long)
        class_names = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
        results["per_class_f1"] = {
            name: per_class_f1[i].item()
            for i, name in enumerate(class_names[: self.num_classes])
        }

        # ── Acc-7 (7-class or num_classes accuracy) ───────────────────────
        results["acc_7"] = results["accuracy"]

        # ── Acc-2 (binary: positive vs non-positive sentiment) ────────────
        # Convention: class 0 = Happy → positive; rest → non-positive
        binary_preds = (preds == 0).long()
        binary_labels = (labels_long == 0).long()
        acc2 = Accuracy(task="binary").to(self.device)
        results["acc_2"] = acc2(binary_preds, binary_labels).item()

        # ── Confusion Matrix ──────────────────────────────────────────────
        confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=self.device,
        )
        for p, t in zip(preds, labels_long):
            confusion[t, p] += 1
        results["confusion_matrix"] = confusion.cpu().numpy().tolist()

        # ── MOSEI Regression Metrics (if continuous targets available) ────
        if regression_preds is not None and regression_targets is not None:
            reg_p = regression_preds.float().cpu().numpy()
            reg_t = regression_targets.float().cpu().numpy()

            # MAE
            results["mae"] = float(np.mean(np.abs(reg_p - reg_t)))

            # Pearson Correlation
            if len(reg_p) > 1 and np.std(reg_p) > 1e-8 and np.std(reg_t) > 1e-8:
                results["corr"] = float(np.corrcoef(reg_p, reg_t)[0, 1])
            else:
                results["corr"] = 0.0
        else:
            # Use softmax probabilities as proxy for continuous prediction
            probs = F.softmax(logits, dim=1)
            # Weighted sum of class indices as continuous prediction
            class_indices = torch.arange(self.num_classes, device=self.device, dtype=torch.float)
            continuous_preds = (probs * class_indices.unsqueeze(0)).sum(dim=1)
            continuous_targets = labels.float()

            results["mae"] = F.l1_loss(continuous_preds, continuous_targets).item()

            cp = continuous_preds.cpu().numpy()
            ct = continuous_targets.cpu().numpy()
            if len(cp) > 1 and np.std(cp) > 1e-8 and np.std(ct) > 1e-8:
                results["corr"] = float(np.corrcoef(cp, ct)[0, 1])
            else:
                results["corr"] = 0.0

        logger.info(
            "Full metrics: Acc=%.4f, F1=%.4f, Acc-2=%.4f, Acc-7=%.4f, MAE=%.4f, Corr=%.4f",
            results["accuracy"], results["macro_f1"],
            results["acc_2"], results["acc_7"],
            results["mae"], results["corr"],
        )

        return results
