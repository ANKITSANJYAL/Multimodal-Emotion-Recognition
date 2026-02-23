import torch
import torch.nn.functional as F
from torchmetrics import F1Score, Accuracy

class AffectiveCausalMetrics:
    """
    Computes standard classification metrics alongside our novel
    Counterfactual Sensitivity and Dissonance scores for CVPR.
    """
    def __init__(self, num_classes=6, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    def compute_standard_metrics(self, logits, labels):
        """Standard metrics for the leaderboard."""
        preds = torch.argmax(logits, dim=1)
        acc = self.acc_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)
        return {"Accuracy": acc.item(), "Macro_F1": f1.item()}

    @torch.no_grad()
    def calculate_causal_sensitivity(self, model, batch, target_modality='audio'):
        """
        Calculates how much the prediction relies on a specific modality
        by generating a diffusion-based counterfactual.
        """
        model.eval()
        text, audio, video = batch['text'], batch['audio'], batch['vision']

        # 1. Factual Prediction P(Y | X)
        z_factual, _, _ = model.bottleneck(text, audio, video)
        p_factual = F.softmax(model.classifier(z_factual), dim=1)

        # 2. Intervention: do(X_m = 0)
        # We zero out the raw features of the target modality
        if target_modality == 'audio':
            audio_cf = torch.zeros_like(audio)
            z_ablated, _, _ = model.bottleneck(text, audio_cf, video)
        elif target_modality == 'video':
            video_cf = torch.zeros_like(video)
            z_ablated, _, _ = model.bottleneck(text, audio, video_cf)
        else:
            text_cf = torch.zeros_like(text)
            z_ablated, _, _ = model.bottleneck(text_cf, audio, video)

        # 3. Generative Healing (The Novelty)
        # Instead of classifying the ablated latent space, we use our Diffusion
        # model to "hallucinate" a plausible affective state using the remaining modalities.
        # We add noise to t=T/2 (partial destruction) and denoise to heal the missing gap.

        b = text.shape[0]
        t_mid = torch.full((b,), model.diffusion.timesteps // 2, device=self.device, dtype=torch.long)

        # Partially noise the ablated latent
        z_noisy = model.diffusion.q_sample(z_ablated, t_mid)

        # Denoise back to t=0 (The model "fills in" the missing affective gap)
        z_counterfactual = z_noisy
        for i in reversed(range(0, model.diffusion.timesteps // 2)):
            t_curr = torch.full((b,), i, device=self.device, dtype=torch.long)
            z_counterfactual = model.diffusion.p_sample(z_counterfactual, t_curr, i)

        # 4. Counterfactual Prediction
        p_counterfactual = F.softmax(model.classifier(z_counterfactual), dim=1)

        # 5. Calculate L1 Distance between distributions (Causal Sensitivity)
        # Shape: (Batch,) -> Mean scalar
        causal_sensitivity = torch.norm(p_factual - p_counterfactual, p=1, dim=1).mean()

        return causal_sensitivity.item()

    def detect_dissonance(self, logits_factual, logits_text_only, threshold=0.4):
        """
        Flags "Sarcasm" or "Hidden Emotion" by comparing the multimodal
        prediction against a text-only prediction.
        """
        p_factual = F.softmax(logits_factual, dim=1)
        p_text = F.softmax(logits_text_only, dim=1)

        # Jensen-Shannon Divergence or simple L1
        divergence = torch.norm(p_factual - p_text, p=1, dim=1)

        # Boolean mask of batches where modalities strongly disagree with text
        is_dissonant = divergence > threshold
        return is_dissonant
