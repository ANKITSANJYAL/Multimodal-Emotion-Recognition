# Affect-Diff Training Guide
**Last updated: 2026-04-24**

---

## Current Status

| Run | val_acc | Problem |
|-----|---------|---------|
| Initial | 0.029 | 3 compounding bugs: 39× class weights, recon=inf, KL=0 |
| After fixes | 0.666 | Always-Happy baseline (model not learning) |
| Focal loss attempt | 0.535 | Focal + class weights amplified rare-class gradient 4× |
| Latest | 0.582 | Overfitting: train=0.750 vs val=0.582 (16.8pt gap) |

**Realistic ceiling with pre-extracted GloVe/COVAREP/FAU features: ~74–78%.**
State-of-the-art papers on CMU-MOSEI with these exact features report ~76%.

---

## What Was Just Changed (April 24 session)

### 1. `encoder_layers=1` wired through the full chain
**Where:** `train_prebuilt.py` DEFAULTS, `affect_diff_module.py`, `latent_bottleneck.py`, all three encoder constructors.
**Why:** 3-layer encoders + hidden_dim=256 = ~8M params on 1956 training samples. That's ~4000 params per sample — severe overfitting territory. 1 layer + hidden_dim=128 + latent_dim=64 = ~750K params.

### 2. `latent_dim` reduced from 128 → 64
**Where:** `train_prebuilt.py` DEFAULTS.
**Why:** Smaller bottleneck = stronger regularization on the classifier pathway.

### 3. Dynamic Modality Gating added to `LatentBottleneck`
**Where:** `models/fusion/latent_bottleneck.py` — new `self.modality_gate` MLP.
**Why:** Text (GloVe semantic polarity) >> Audio (COVAREP arousal) >> Video (FAU facial actions — very noisy for MOSEI). Uniform weighting hurts. The gate is a tiny 384→64→3 MLP that learns per-sample modality importance. Softmax output × 3 preserves activation scale at init.

### 4. Asymmetric modality dropout
**Where:** `modules/affect_diff_module.py` shared_step.
**Before:** uniform 10% dropout for all three modalities.
**After:** text=5%, audio=10%, video=20%.
**Why:** Forces the classifier to rely more on text/audio and become robust to missing video.

### 5. Focal loss CLI bug fixed
**Before:** `--no_focal_loss` default caused focal to be ON in every CLI run (overriding `DEFAULTS["use_focal_loss"]=False`).
**After:** `--use_focal_loss` flag — focal is OFF by default, opt-in only.

---

## Before Every New Run — Clear Checkpoints

Stale checkpoints cause `ckpt_path="best"` to load weights from a previous run.
Always run this cell first:

```python
import shutil, os
shutil.rmtree("/kaggle/working/checkpoints", ignore_errors=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
print("Checkpoints cleared.")
```

---

## Recommended Experiment Sequence

### Step 1 — Baseline with new architecture (run this first)
```bash
python train_prebuilt.py --epochs 80 --patience 20
```
Expect: val_acc 0.66–0.72. The train/val gap should be ≤10pt now (was 16.8pt).

### Step 2 — Try crossmodal attention fusion
```bash
python train_prebuilt.py --fusion_type crossmodal --epochs 80 --patience 20
```
Expect: ~+1–3pt over concat if text/audio patterns interact usefully.
Watch for: slower convergence (needs more epochs).

### Step 3 — Increase encoder complexity if Step 1 val_acc > 0.70
Only do this if the gap between train and val is ≤8pt:
```bash
python train_prebuilt.py --encoder_layers 2 --hidden_dim 128 --epochs 100 --patience 25
```

### Step 4 — Try beta_kl sweep
```bash
python train_prebuilt.py --beta_kl 0.01 --epochs 80   # near-deterministic encoder
python train_prebuilt.py --beta_kl 0.5  --epochs 80   # stronger regularization
```
Lower beta_kl → encoder acts more like a standard Transformer (less generative), which usually helps discriminative accuracy.

### Step 5 — Disable KL entirely (pure discriminative)
```bash
python train_prebuilt.py --beta_kl 0.0 --free_bits 0.0 --epochs 80
```
This turns the VAE into a deterministic encoder. Often best for classification-only.

---

## Things to Try Manually (In Order of Expected Impact)

### HIGH IMPACT

#### A. Separate per-modality classifier heads (late fusion)
**What:** Instead of fusing T+A+V into one latent, give each modality its own pooling + small classifier head. Final prediction = learnable weighted average of 3 heads.

**Where to add:** `modules/affect_diff_module.py`

**How:**
```python
# In __init__, add 3 unimodal classifiers:
self.text_cls = nn.Linear(hidden_dim, num_classes)
self.audio_cls = nn.Linear(hidden_dim, num_classes)
self.video_cls = nn.Linear(hidden_dim, num_classes)
self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # T>A>V prior

# In shared_step, after bottleneck:
mu_t, _, t_feat, a_feat, v_feat = self.bottleneck.encode(text, audio, video)
w = torch.softmax(self.fusion_weight, dim=0)
logits_t = self.text_cls(t_feat.mean(1))
logits_a = self.audio_cls(a_feat.mean(1))
logits_v = self.video_cls(v_feat.mean(1))
logits = w[0]*logits_t + w[1]*logits_a + w[2]*logits_v
```
**Why high impact:** Prevents noisy video features from corrupting the text signal in the shared latent. Each modality's classifier trains on clean unimodal features. The weight vector learns text dominates.

#### B. Disable the VAE entirely (deterministic encoder)
**What:** Set `beta_kl=0` and `free_bits=0`. The reparameterization trick still runs but the KL term is zero — effectively a deterministic Transformer encoder with latent regularization only from the task loss.

**How:** `python train_prebuilt.py --beta_kl 0.0 --free_bits 0.0`

**Why:** For a 6-class discriminative task with 1956 samples, generative modeling (VAE/diffusion) is hard to justify. The classification accuracy is the ceiling — the VAE adds noise that can hurt.

#### C. Text-only baseline (sanity check)
**What:** Zero out audio and video entirely, train text-only.
```python
# In shared_step, before bottleneck:
audio = torch.zeros_like(audio)
video = torch.zeros_like(video)
```
This tells you the text ceiling. If text-only gets 0.72 and the full model gets 0.68, the other modalities are hurting.

---

### MEDIUM IMPACT

#### D. Label smoothing sweep
Current: `label_smoothing=0.05`.
With 6 classes and only ~1956 samples, 0.1 often helps.
Try: `--label_smoothing 0.0`, `--label_smoothing 0.1`, `--label_smoothing 0.15`.

#### E. Weight decay increase
Current: `weight_decay=1e-4`.
With this few samples, try: `--weight_decay 5e-4` or `--weight_decay 1e-3`.

#### F. Learning rate reduction
Current: `lr=5e-4`.
After warmup kicks in this is fine, but try `--lr 2e-4` if training is unstable.

#### G. Longer training with stronger patience
```bash
python train_prebuilt.py --epochs 150 --patience 30
```
The cyclical KL annealing has 20-epoch cycles — you need at least 3 full cycles (60 epochs) to see convergence.

#### H. Increase batch size (if GPU memory allows)
Current: 64. Try `--batch_size 128`. Larger batches stabilize gradient estimates and help with class-imbalanced data (more rare-class samples per batch).

---

### LOW IMPACT / RISKY

#### I. Enable diffusion (after getting stable val_acc > 0.70)
Only add diffusion AFTER the classifier is converging well:
```bash
python train_prebuilt.py --use_diffusion  # (need to add this flag — currently hardcoded False)
```
Diffusion on 1956 samples is very likely to overfit the generative component. Use `lambda_diff=0.01` (very weak) if you try it.

#### J. Enable reconstruction
Only after fixing val_acc > 0.70:
```bash
python train_prebuilt.py --use_reconstruction
```
The recon=inf bug is fixed (data is clamped to [-10, 10]), but reconstruction on noisy FAU features may not help classification.

#### K. Foundation models (RoBERTa/HuBERT/CLIP)
Only if you have ~4h+ compute per run:
```bash
python train_prebuilt.py --encoder_type foundation --encoder_layers 1 --freeze_backbones True
```
RoBERTa text features are much stronger than GloVe for sentiment. Expected gain: +3–5pt on text-heavy classes.

---

## Diagnosing Your Runs

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `val_acc ≈ 0.666` after 5+ epochs | Model predicts all-Happy | Class weights too weak or KL=0 collapse |
| `val_acc ≈ 0.03` | Predicting rare class everywhere | Class weights too strong (>5×) or focal loss on tiny data |
| `train_acc >> val_acc` by >10pt | Overfitting | Reduce encoder_layers, hidden_dim, increase weight_decay |
| `KL = 0` in logs | Posterior collapse | Already fixed by cyclical annealing; check free_bits=0 |
| `recon = inf` in logs | Unnormalized outlier features | Fixed in _norm() — verify you're using train_prebuilt.py |
| Loss goes NaN after epoch 1 | fp16 overflow | Switch to `--precision bf16-mixed` on A100; or `--precision 32` |
| `test_loss_diff > 0` when `use_diffusion=False` | Stale checkpoint | Clear /kaggle/working/checkpoints before running |

---

## Architecture Summary (Current)

```
GloVe(300) → TextEncoder(1 layer, d=128) ──┐
COVAREP(74) → AudioEncoder(1 layer, d=128) ─→ ModalityGate(3×128→64→3) → gated feats
FAU(35) → VideoEncoder(1 layer, d=128) ────┘
                                             ↓
                             CausalGraph(notears, 3×3)
                                             ↓
                         Concat+MLP fusion → hidden(128)
                                             ↓
                            VAE heads (mu/logvar, d=64)
                                             ↓
                        MeanPool + AttnPool blend → z_pooled(64)
                                             ↓
                         Classifier (64→64→32→6, dropout=0.4)
```

Total parameters: ~750K (down from ~8M).

---

## Key Constants to Know

- **Training samples:** 1956 (70% of 2795)
- **Val/Test samples:** 419 / 420
- **Class distribution:** Happy 66.2%, Sad 17%, Angry 10%, Fear 1.7%, Disgust 1.8%, Surprise 2.9%
- **Always-Happy baseline:** 66.2% val accuracy (just predicting class 0 always)
- **State-of-art with these features:** ~76% (MulT, 2019)
- **Realistic target with this codebase:** ~72–75% with careful tuning
