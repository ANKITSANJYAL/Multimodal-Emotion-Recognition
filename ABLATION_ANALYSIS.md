# Ablation Study Analysis — Affect-Diff (CVPR 2026)
## Date: March 31, 2026

---

## 1. Executive Summary

**25/25 experiments completed.** Results reveal a critical architectural flaw: the VAE's free-bits mechanism creates an irreducible KL loss floor of **256.0** that constitutes **99.3% of total loss**, drowning out all task-relevant gradient signal. As a result, auxiliary losses (diffusion, causal, reconstruction) actively **hurt** classification rather than help it.

**Key finding:** `Classifier_Only` (69.85%) **beats** `Full_Model` (68.18%) by +1.67% — the paper's core contribution (generative causal disentanglement) is currently a net negative.

---

## 2. Complete Results Table

### Component Ablation

| Experiment | Epochs | Best val_acc | Test Acc | KL Loss | Task Loss | Δ vs Full |
|---|---|---|---|---|---|---|
| **COMP_Full_Model** | 47 | 0.718 | **0.6818** | 256.00 | 1.113 | baseline |
| COMP_No_Diffusion | 48 | 0.715 | **0.6985** | 256.00 | 1.159 | **+1.67%** |
| COMP_No_Causal | 52 | 0.718 | 0.6682 | 256.00 | 1.200 | -1.36% |
| COMP_No_Reconstruction | 48 | 0.721 | 0.6697 | 256.01 | 1.200 | -1.21% |
| COMP_No_Augmentation | 71 | 0.724 | 0.6652 | 256.00 | 1.362 | -1.66% |
| **COMP_Classifier_Only** | 52 | **0.721** | **0.6985** | 0.00 | 1.166 | **+1.67%** |
| COMP_Concat_Fusion | 44 | 0.724 | 0.6985 | 256.00 | 1.145 | +1.67% |
| COMP_Gumbel_DAG | 60 | 0.724 | 0.6439 | 256.00 | 1.260 | -3.79% |
| COMP_Beta_TC_VAE | 0 | — | CRASHED | — | — | OOM |

### Hyperparameter Sensitivity

| Experiment | Epochs | Best val_acc | Test Acc | KL Loss | Task Loss |
|---|---|---|---|---|---|
| HP_beta_kl_0.01 | 61 | 0.730 | 0.6742 | 5.12 | 1.214 |
| HP_beta_kl_0.1 | 48 | 0.712 | 0.6939 | 51.20 | 1.135 |
| HP_beta_kl_1.0 | 48 | 0.718 | 0.6985 | 512.00 | 1.149 |
| HP_beta_kl_5.0 | 59 | 0.718 | 0.6727 | 2560.01 | 1.196 |
| **HP_lambda_diff_0.1** | 42 | 0.706 | **0.7076** | 256.00 | 1.121 |
| HP_lambda_diff_1.0 | 41 | 0.718 | 0.6712 | 256.01 | 1.122 |
| HP_lambda_diff_2.0 | 41 | 0.718 | 0.6803 | 256.00 | 1.132 |
| HP_lambda_causal_0.01 | 46 | 0.712 | 0.6818 | 256.00 | 1.119 |
| HP_lambda_causal_0.1 | 48 | 0.718 | 0.6848 | 256.00 | 1.165 |
| HP_lambda_causal_0.5 | 48 | 0.715 | 0.6985 | 256.00 | 1.133 |
| HP_label_smoothing_0.0 | 45 | 0.724 | 0.6939 | 256.00 | 0.935 |
| HP_label_smoothing_0.05 | 41 | 0.715 | 0.6879 | 256.00 | 1.040 |
| HP_label_smoothing_0.2 | 53 | 0.727 | 0.6727 | 256.00 | 1.328 |
| HP_free_bits_0.0 | 43 | 0.718 | 0.6455 | 0.09 | 1.266 |
| HP_free_bits_1.0 | 49 | 0.715 | 0.6909 | 128.00 | 1.158 |
| HP_free_bits_4.0 | 53 | 0.727 | 0.6500 | 512.85 | 1.210 |

---

## 3. Root Cause: The Free-Bits KL Floor Problem

### The Math

```
KL_floor = beta_kl × latent_dim × free_bits
         = 0.5 × 256 × 2.0
         = 256.0
```

This matches the observed `loss_kl = 256.00` in **every experiment** with free_bits=2.0.

### What's Happening

The `compute_kl_loss` function applies `torch.clamp(kl_per_dim, min=free_bits)` — each of the 256 latent dimensions has its KL clamped to at least 2.0 nats, then multiplied by beta=0.5, giving an irreducible floor of 256.0.

When the posterior **perfectly matches the prior** (i.e., the VAE has collapsed and encodes zero information), the raw KL per dimension is 0.0 — but free_bits clamps it to 2.0, producing a phantom loss of 256.0.

### Consequence

| Loss Component | Value | % of Total |
|---|---|---|
| `loss_kl` (floor) | 256.0 | **99.3%** |
| `loss_task` | 1.11 | 0.4% |
| `loss_diff` | 0.64 | 0.2% |
| `loss_recon` | 0.44 | 0.2% |
| `loss_causal` | 0.04 | 0.0% |
| **Total** | **257.7** | 100% |

**The optimizer is spending 99.3% of its gradient budget trying to reduce an irreducible constant.** The task loss that actually determines classification quality gets <0.5% of gradient signal.

### Verification: KL scales linearly with beta and free_bits

| Experiment | beta_kl | Expected KL Floor | Actual KL |
|---|---|---|---|
| HP_beta_kl_0.01 | 0.01 | 0.01 × 256 × 2.0 = 5.12 | **5.12** ✓ |
| HP_beta_kl_0.1 | 0.1 | 0.1 × 256 × 2.0 = 51.2 | **51.20** ✓ |
| HP_beta_kl_1.0 | 1.0 | 1.0 × 256 × 2.0 = 512.0 | **512.00** ✓ |
| HP_beta_kl_5.0 | 5.0 | 5.0 × 256 × 2.0 = 2560.0 | **2560.01** ✓ |
| HP_free_bits_0.0 | 0.5 | 0.5 × 256 × 0.0 = 0.0 | **0.09** ✓ (near zero) |
| HP_free_bits_1.0 | 0.5 | 0.5 × 256 × 1.0 = 128.0 | **128.00** ✓ |
| HP_free_bits_4.0 | 0.5 | 0.5 × 256 × 4.0 = 512.0 | **512.85** ✓ |

**Every KL value is exactly the irreducible floor.** The posterior has collapsed to the prior in all cases — free_bits is creating phantom loss, not preventing collapse.

---

## 4. Why Auxiliary Losses Hurt

1. **Gradient competition**: With KL=256 dominating, the optimizer focuses on the constant floor gradient rather than reducing task loss
2. **Diffusion loss adds noise**: The DDPM loss (0.64) tries to model z-space dynamics, but z has collapsed to noise — diffusion is modeling garbage
3. **Reconstruction loss is detached**: With a collapsed posterior, reconstruction gradients also optimize a degenerate latent space
4. **Best result (HP_lambda_diff_0.1)**: Reducing diffusion weight from 0.5→0.1 gives best accuracy (70.76%) because it reduces gradient interference

---

## 5. Key Insights from Results

### What Works
- **Causal graph matters** (No_Causal drops to 66.82% vs 68.18% Full → causal helps +1.4%)
- **Augmentation helps** (No_Augmentation drops to 66.52% → augmentation helps +1.7%)
- **NOTEARS >> Gumbel** (Gumbel_DAG at 64.39% vs Full 68.18% → NOTEARS clearly better)
- **Lower diffusion weight helps** (lambda_diff=0.1 → 70.76%, best overall)
- **CrossModal fusion = Concat** (both at 69.85%, CrossModal isn't helping yet)

### What Doesn't Work Yet
- **Diffusion hurts** (No_Diffusion = Classifier_Only = 69.85% > Full 68.18%)
- **Reconstruction hurts** (No_Reconstruction 66.97% < No_Diffusion 69.85%)
- **Free_bits is catastrophic** (creates constant loss floor, masks all signal)
- **Beta_TC_VAE still OOMs** (needs batch_size reduction or gradient checkpointing)

---

## 6. Prioritized Action Plan for Google-Level Research

### Phase 1: Fix the KL Floor (CRITICAL — Expected Impact: +5-8%)

**Problem**: Free-bits clamp creates irreducible loss floor that drowns task signal.

**Fix**: Replace current free-bits implementation with **stop-gradient free-bits** that prevents collapse without creating phantom loss:

```python
# CURRENT (broken): Creates floor loss even when KL is below free_bits
kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

# FIXED: Only contribute to loss when KL exceeds free_bits threshold
kl_per_dim = torch.where(
    kl_per_dim > free_bits,
    kl_per_dim,
    free_bits * torch.ones_like(kl_per_dim)  # Constant → zero gradient
)
# OR better — the Kingma formulation:
kl_per_dim = torch.max(kl_per_dim, free_bits * torch.ones_like(kl_per_dim).detach())
```

**Even better — use KL annealing instead of free bits:**
```python
# Cyclical KL annealing (Fu et al., 2019)
kl_weight = min(1.0, (step % cycle_length) / (cycle_length * proportion))
loss_kl = kl_weight * beta_kl * raw_kl
```

### Phase 2: Proper Multi-Task Loss Balancing (Expected Impact: +3-5%)

**Problem**: Fixed loss weights mean KL/diffusion/recon dominate or underperform.

**Options (pick one):**

1. **Uncertainty Weighting** (Kendall et al., 2018):
   ```python
   # Learnable log-variance per task
   loss = sum(0.5 * exp(-log_var_i) * loss_i + 0.5 * log_var_i)
   ```

2. **GradNorm** (Chen et al., 2018):
   - Dynamically reweight losses to equalize gradient magnitudes

3. **PCGrad** (Yu et al., 2020):
   - Project conflicting task gradients to avoid interference

### Phase 3: Fix Diffusion Training (Expected Impact: +2-4%)

Currently the diffusion model is training on a collapsed latent space. Once KL is fixed:

1. **Two-stage training**: Train encoder+classifier first, THEN add diffusion
2. **Diffusion as inference-time augmentation**: Train diffusion, use it to generate augmented z during test-time
3. **Increase diffusion capacity**: Current UNet1D may be too small for 256-d latent

### Phase 4: Stronger Encoders (Expected Impact: +5-10%)

Current: GloVe (300d), COVAREP (74d), FAU (35d) — all legacy features.

**Upgrade to foundation models:**
- Text: RoBERTa-base (768d) — already in config, just needs `encoder_type: "foundation"`
- Audio: HuBERT-base (768d)
- Video: CLIP ViT-B/16 (768d)

This alone could push accuracy from ~70% to ~80%+ based on MOSEI literature.

### Phase 5: Statistical Rigor (Required for Top Venue)

1. **Multi-seed experiments** (3-5 seeds) with mean ± std reporting
2. **Significance testing** (paired t-test or Wilcoxon) for ablation claims
3. **Confidence intervals** in all tables
4. **Effect size** reporting (Cohen's d)

### Phase 6: Additional Experiments for Paper

1. **Qualitative analysis**:
   - Visualize learned causal graph (adjacency matrix)
   - t-SNE of latent space (collapsed vs disentangled)
   - Attention heatmaps from cross-modal fusion

2. **Generalization**:
   - CMU-MOSI (smaller dataset, binary sentiment)
   - IEMOCAP (dyadic conversation, 4 emotions)

3. **Ablation completeness**:
   - Latent dim sweep (64, 128, 256, 512)
   - Number of diffusion steps (100, 500, 1000)
   - Warmup schedule length (10, 20, 50 epochs)

---

## 7. Immediate Next Steps (This Week)

1. **Fix `compute_kl_loss`** — Remove free-bits floor, use proper KL annealing
2. **Decouple loss logging** — Log `loss_task` as monitor metric instead of total loss
3. **Re-run Full Model + Classifier_Only** with fixed KL to verify improvement
4. **Fix Beta_TC_VAE OOM** — Reduce batch_size to 32 for this variant
5. **Add multi-seed support** to sweep infrastructure

---

## 8. Target Performance

| Metric | Current | After KL Fix | After Full Pipeline | SOTA (MOSEI) |
|---|---|---|---|---|
| 6-class Acc | 70.8% | ~75-77% | ~80-83% | ~85% |
| Weighted F1 | ~0.68 | ~0.73-0.75 | ~0.78-0.81 | ~0.83 |

The gap to SOTA is bridgeable with: fixed KL + foundation encoders + proper multi-task balancing.
