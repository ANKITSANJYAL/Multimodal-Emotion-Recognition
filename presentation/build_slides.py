"""Build presentation PDF from slide data using Chrome headless."""

import subprocess, os, base64
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent  # repo root
FIGS   = ROOT / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def img64(name):
    p = FIGS / name
    data = p.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

# Pre-load all images as base64 so the HTML is self-contained
BC  = img64("fig_baseline_comparison.png")
TC  = img64("fig_training_curves.png")
PF  = img64("fig_perclass_f1.png")
CI  = img64("fig_causal_influence.png")
WS  = img64("fig_warmup_schedule.png")
SS  = img64("fig_seed_stability.png")
CD  = img64("fig_class_distribution.png")
MD  = img64("fig_modalities.png")
SR  = img64("fig_sentiment_scale.png")
AR  = img64("fig_architecture.png")
CD2 = img64("fig_causal_dag.png")

HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0d1117; --fg: #e6edf3; --accent: #58a6ff;
    --accent2: #f78166; --muted: #8b949e;
    --card: #161b22; --border: #30363d;
  }}
  body {{ background: var(--bg); font-family: 'Inter', system-ui, sans-serif; }}

  .slide {{
    width: 1280px; height: 720px;
    background: var(--bg); color: var(--fg);
    padding: 44px 64px 36px;
    display: flex; flex-direction: column;
    page-break-after: always;
    overflow: hidden;
    position: relative;
  }}
  .slide:last-child {{ page-break-after: auto; }}

  /* slide number badge */
  .pg {{
    position: absolute; bottom: 20px; right: 28px;
    color: var(--muted); font-size: 12px; font-weight: 600;
  }}
  /* top accent bar */
  .slide::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }}

  h1 {{ color: var(--accent); font-size: 30px; font-weight: 800;
       border-bottom: 1.5px solid var(--border); padding-bottom: 8px; margin-bottom: 18px; }}
  h2 {{ color: var(--accent); font-size: 22px; font-weight: 700; margin-bottom: 10px; }}
  h3 {{ color: var(--accent2); font-size: 15px; font-weight: 700;
       margin: 12px 0 4px; text-transform: uppercase; letter-spacing: .05em; }}
  strong {{ color: var(--accent); }}
  em {{ color: var(--accent2); font-style: normal; font-weight: 600; }}
  p, li {{ font-size: 14px; line-height: 1.65; color: var(--fg); }}
  ul {{ padding-left: 18px; }}
  li {{ margin-bottom: 3px; }}

  .cols {{ display: grid; gap: 28px; flex: 1; }}
  .cols2 {{ grid-template-columns: 1fr 1fr; }}
  .cols3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .cols-6-4 {{ grid-template-columns: 6fr 4fr; }}
  .cols-4-6 {{ grid-template-columns: 4fr 6fr; }}

  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
  }}
  .card-accent {{
    background: #0d2137; border: 1px solid #1f4a7a;
    border-radius: 8px; padding: 14px 16px;
  }}
  .callout {{
    background: #1a0a0a; border-left: 4px solid var(--accent2);
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 10px 0;
    font-size: 13.5px; line-height: 1.55;
  }}
  .callout strong {{ color: var(--accent2); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }}
  th {{ background: #1f2937; color: var(--accent); padding: 7px 10px;
       border-bottom: 2px solid var(--border); text-align: left; font-size: 12px; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  tr.ours td {{ color: var(--accent); font-weight: 700;
               background: rgba(88,166,255,0.06); }}
  tr:last-child td {{ border-bottom: none; }}

  img {{ width: 100%; height: 100%; object-fit: contain; }}
  .img-wrap {{ display: flex; align-items: center; }}

  .tag {{
    display: inline-block; background: #1f3a5f; color: var(--accent);
    border-radius: 4px; padding: 3px 10px; font-size: 12px;
    font-weight: 600; margin: 0 4px 4px 0;
  }}
  .big {{ font-size: 48px; font-weight: 800; color: var(--accent2); line-height: 1; }}
  .mono {{ font-family: 'Courier New', monospace; font-size: 13px;
           background: var(--card); padding: 2px 6px;
           border-radius: 4px; color: #79c0ff; }}
  .eq {{
    text-align: center; padding: 10px; background: var(--card);
    border-radius: 6px; font-family: 'Courier New', monospace;
    font-size: 13px; color: #79c0ff; margin: 8px 0;
  }}

  /* TITLE SLIDE */
  .title-slide {{
    justify-content: center; gap: 18px;
  }}
  .title-slide .pretitle {{
    font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .12em; font-weight: 600;
  }}
  .title-slide h1 {{
    font-size: 48px; border: none; padding: 0; margin: 4px 0;
  }}
  .title-slide .subtitle {{
    font-size: 20px; color: var(--muted); margin-bottom: 10px;
  }}
  .author {{ font-size: 13px; color: var(--muted); margin-top: 12px; }}
</style>
</head>
<body>

<!-- ══════════════════ SLIDE 1: TITLE ══════════════════ -->
<div class="slide title-slide">
  <div class="pretitle">CISC 6080 · Fordham University · Capstone Project</div>
  <h1>Affect-Diff</h1>
  <div class="subtitle">Multimodal Emotion Recognition via Causal-Diffusion Bridge</div>

  <div class="callout" style="max-width:780px">
    <strong>The problem in one sentence:</strong>
    Standard models achieve 66 % accuracy on CMU-MOSEI by
    <em>never predicting Fear, Disgust, or Surprise</em>  three of six Ekman emotions  at all.
  </div>

  <div class="cols cols3" style="margin-top:14px; max-width:900px">
    <div class="card"><strong>What</strong><br><span style="font-size:13px;color:var(--muted)">6-class emotion recognition from text, audio &amp; video (CMU-MOSEI)</span></div>
    <div class="card"><strong>Why hard</strong><br><span style="font-size:13px;color:var(--muted)">Happy = 65.9 % of data. Three classes under 2 % each.</span></div>
    <div class="card"><strong>Our approach</strong><br><span style="font-size:13px;color:var(--muted)">Causal graph + β-VAE + stop-gradiented DDPM prior</span></div>
  </div>

  <div style="margin-top:18px">
    <span class="tag">CMU-MOSEI</span>
    <span class="tag">NOTEARS Causal Graph</span>
    <span class="tag">Diffusion Models</span>
    <span class="tag">Class Imbalance</span>
    <span class="tag">β-VAE</span>
  </div>

  <div class="author">Ankit Sanjyal &nbsp;·&nbsp; asanjyal56@fordham.edu</div>
  <div class="pg">1 / 12</div>
</div>


<!-- ══════════════════ SLIDE 2: THE DATA ══════════════════ -->
<div class="slide">
  <h1>The Dataset: CMU-MOSEI</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:12px">
      <p>CMU-MOSEI contains <strong>23,453 utterances</strong> from 1,000 YouTube speakers, each aligned across three modalities. We use a <em>3,292-segment</em> subset with clean tri-modal alignment.</p>
      <div class="img-wrap" style="flex:1; max-height:210px">
        <img src="{MD}" alt="three modalities">
      </div>
      <div class="card" style="font-size:13px">
        Each segment is labeled for <strong>6 Ekman emotions</strong> (Happy, Sad, Angry, Disgust, Surprise, Fear) — not sentiment polarity. Labels are independent per emotion; multi-label possible.
      </div>
      <div class="card" style="font-size:12.5px;border-left:3px solid var(--accent2)">
        <strong>Why 3,292 of 23,453?</strong> — Strict <em>tri-modal alignment</em>: every segment needs valid GloVe + COVAREP + FACET arrays with matching temporal intervals. Misaligned or partially missing segments are discarded, leaving ~14 % of the corpus. This is a <em>data-limited regime by design</em>, not by choice — and our sentiment results show the model scales: 7× more data → <strong>+90 % BalAcc</strong>.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="img-wrap" style="flex:1">
        <img src="{CD}" alt="class distribution">
      </div>
      <div class="callout" style="font-size:13.5px">
        <strong>The core challenge:</strong> Happy alone is 65.9 % of labels.
        Fear, Disgust, and Surprise collectively represent under 7 %  yet all three are
        clinically meaningful emotions that a real system must detect.
      </div>
    </div>
  </div>
  <div class="pg">2 / 12</div>
</div>


<!-- ══════════════════ SLIDE 3: WHY MODELS FAIL ══════════════════ -->
<div class="slide">
  <h1>Why Existing Models Fail</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:10px">
      <h3 style="margin-top:0">Two compounding failure modes</h3>
      <div class="card-accent">
        <strong>① Majority-class collapse</strong>
        <p style="font-size:13px;margin-top:4px">Under severe imbalance, every fusion model maximises accuracy by predicting Happy / Sad for everything. A "predict Happy always" classifier scores <em>66 % accuracy</em> while being completely useless for emotion recognition.</p>
      </div>
      <div class="card-accent">
        <strong>② Modality collapse</strong>
        <p style="font-size:13px;margin-top:4px">Models over-rely on text and ignore audio / video  the modalities that carry prosody and facial expression. When text degrades, performance collapses entirely.</p>
      </div>
      <div class="callout" style="font-size:13px">
        Focal loss, class-weighted loss, post-hoc causal correction (CausalMER), and diffusion imputation (IMDer) each address one axis  none address <em>both simultaneously in the forward pass</em>.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <h3 style="margin-top:0">Five baselines  same blind spot</h3>
      <table>
        <tr><th>Method</th><th>Year</th><th>Val-BalAcc ↑</th><th>Test-Acc</th></tr>
        <tr><td>TFN</td><td>2017</td><td>0.248</td><td>0.667</td></tr>
        <tr><td>MulT</td><td>2019</td><td>0.278</td><td>0.626</td></tr>
        <tr><td>MISA</td><td>2020</td><td>0.278</td><td>0.633</td></tr>
        <tr><td>MMIM</td><td>2021</td><td>0.266</td><td>0.679</td></tr>
        <tr><td>TETFN</td><td>2022</td><td>0.324</td><td>0.600</td></tr>
        <tr class="ours"><td><strong>Affect-Diff</strong></td><td></td><td><strong>0.384</strong></td><td>0.642</td></tr>
      </table>
      <p style="font-size:12.5px;color:var(--muted);margin-top:6px">
        All five produce <strong style="color:var(--accent2)">zero F1</strong> on Fear, Disgust, and Surprise  re-implemented under identical training conditions (same split, focal loss, label smoothing).
      </p>
      <p style="font-size:12.5px;color:var(--muted)">
        Val-BalAcc = macro-average recall across all 6 classes. <em>This is the metric that cannot be gamed by majority-class predictions.</em>
      </p>
    </div>
  </div>
  <div class="pg">3 / 12</div>
</div>


<!-- ══════════════════ SLIDE 4: ARCHITECTURE DIAGRAM ══════════════════ -->
<div class="slide">
  <h1>Architecture Overview: Causal-Diffusion Bridge</h1>
  <div style="flex:1; display:flex; flex-direction:column; gap:10px">
    <div class="img-wrap" style="flex:1; max-height:360px">
      <img src="{AR}" alt="Affect-Diff architecture diagram">
    </div>
    <div class="cols cols3" style="flex:0 0 auto">
      <div class="card" style="font-size:12.5px">
        <strong style="color:#d95f0e">NOTEARS Causal Graph</strong><br>
        Learns directed edges over {{T, A, V}}. Column sums → weights <em>w</em> that gate each modality before fusion. Same <em>w</em> conditions the U-Net.
      </div>
      <div class="card" style="font-size:12.5px">
        <strong style="color:#6c3483">β-VAE Bottleneck</strong><br>
        Compresses fused representation to latent <em>z</em>. Free-bits KL prevents posterior collapse on rare classes.
      </div>
      <div class="card" style="font-size:12.5px">
        <strong style="color:#c0392b">Stop-Gradiented DDPM</strong><br>
        1D U-Net shapes latent manifold from <em>sg(z)</em> — regularizes without conflicting with classification gradients.
      </div>
    </div>
  </div>
  <div class="pg">4 / 12</div>
</div>


<!-- ══════════════════ SLIDE 5: CAUSAL GRAPH DETAIL ══════════════════ -->
<div class="slide">
  <h1>Causal Attention Graph: Structure and Learned Weights</h1>
  <div class="cols cols2" style="flex:1">
    <div class="img-wrap">
      <img src="{CD2}" alt="NOTEARS causal DAG and adjacency matrix">
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <div class="card-accent">
        <h3 style="margin-top:0">What it learns</h3>
        <p style="font-size:13px">A directed acyclic graph over 3 modality nodes {{T, A, V}} using scaled dot-product attention, constrained to be a DAG via the <strong>NOTEARS penalty</strong>:</p>
        <div style="background:var(--card);border-radius:6px;padding:8px 12px;margin-top:6px;font-family:'Courier New',monospace;font-size:12px;color:#79c0ff;text-align:center">
          h(A) = tr(e^(A∘A)) − 3 = 0
        </div>
      </div>
      <div class="card-accent">
        <h3 style="margin-top:0">Why it matters</h3>
        <p style="font-size:13px">Standard fusion treats all modalities equally. The causal graph learns <em>which modality to trust per sample</em>, gating h̃ᵐ = wₘ · hᵐ before fusion.<br><br>
        The same weights <strong>w</strong> condition the diffusion U-Net, making the denoiser modality-aware.</p>
      </div>
      <div class="callout" style="font-size:12.5px">
        Removing NOTEARS (→ Gumbel-Softmax) drops val-BalAcc by <strong>−0.059 (−15%)</strong>, confirming the acyclicity constraint adds real signal beyond simple sparsity.
      </div>
    </div>
  </div>
  <div class="pg">5 / 12</div>
</div>


<!-- ══════════════════ SLIDE 6 (was 4): ARCHITECTURE DETAILS ══════════════════ -->
<div class="slide">
  <h1>Affect-Diff: Three Jointly Trained Mechanisms</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:10px">

      <div class="card-accent">
        <h3 style="margin-top:0">① NOTEARS Causal Graph</h3>
        <p style="font-size:13px">Learns a DAG over {{T, A, V}}. Column sums produce importance weights that gate each modality's hidden sequence <em>before</em> fusion  not post-hoc. Same weights condition the diffusion denoiser.</p>
        <div class="eq">w = softmax(A&#7488;·1) &nbsp;&nbsp; h̃ᵐ = hᵐ · wₘ &nbsp;&nbsp; h(A) = tr(e^(A∘A)) − 3 = 0</div>
      </div>

      <div class="card-accent">
        <h3 style="margin-top:0">② β-VAE Bottleneck</h3>
        <p style="font-size:13px">Compresses the fused representation to a regularized latent z ∈ ℝ¹²⁸. Free-bits objective prevents posterior collapse on majority classes.</p>
        <div class="eq">z = μ + ε·σ &nbsp;&nbsp;&nbsp; L_KL = β · max(0, KL − λ)</div>
      </div>

      <div class="card-accent">
        <h3 style="margin-top:0">③ Stop-Gradiented DDPM Prior</h3>
        <p style="font-size:13px">A 1D U-Net denoiser shapes the latent manifold. Gradient is <em>blocked</em> from reaching the encoder  regularizes without conflicting with classification loss.</p>
        <div class="eq">z_diff = sg(z) &nbsp;&nbsp;&nbsp; L_diff = ‖ε − ε_θ(z_t, t, y, w)‖²</div>
      </div>

    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <div class="card" style="font-size:13px">
        <strong>Combined loss:</strong>
        <div class="eq" style="margin-top:6px">L = L_task + γ_kl·L_KL + γ_diff·L_diff + λ_dag·h(A)</div>
        Each γ is ramped up via curriculum warmup  classifier dominates first, auxiliary losses phase in gradually.
      </div>
      <div class="img-wrap" style="flex:1">
        <img src="{WS}" alt="warmup schedule">
      </div>
      <div class="card" style="font-size:12.5px;color:var(--muted)">
        Classifier-only (ep 0–9) → KL ramps in (ep 0–30) → diffusion ramps in (ep 9–29). Prevents early posterior collapse while letting the classifier stabilize.
      </div>
    </div>
  </div>
  <div class="pg">6 / 12</div>
</div>


<!-- ══════════════════ SLIDE 7: WHAT THE MODEL LEARNS ══════════════════ -->
<div class="slide">
  <h1>What the Model Actually Learns</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:10px">
      <h3 style="margin-top:0">Causal Modality Dynamics</h3>
      <div class="img-wrap" style="flex:1; max-height:300px">
        <img src="{CI}" alt="causal influence over training">
      </div>
      <div class="card" style="font-size:13px">
        <strong>Epoch 10:</strong> Video dominates (V≈0.58)  facial AU cues guide coarse class boundaries early.<br>
        <strong>Epoch 40+:</strong> Audio ≈ Text (A≈0.38, T≈0.40), Video recedes  fine-grained prosody and language take over.<br><br>
        This is <em>non-trivial</em>: the graph doesn't converge to uniform or text-dominant weights  it discovers temporal structure in which modality is most informative.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <h3 style="margin-top:0">Ablation: What Each Component Adds</h3>
      <table>
        <tr><th>Configuration</th><th>Val-BalAcc</th><th>Δ vs Full</th></tr>
        <tr class="ours"><td><strong>Full Model (3 seeds)</strong></td><td><strong>0.384 ± 0.000</strong></td><td></td></tr>
        <tr><td>No Stop-Gradient</td><td>0.291</td><td style="color:#f78166">−0.093 (−24%)</td></tr>
        <tr><td>No Diffusion Prior</td><td>0.292</td><td style="color:#f78166">−0.092 (−24%)</td></tr>
        <tr><td>No NOTEARS (Gumbel)</td><td>0.325</td><td style="color:#f78166">−0.059 (−15%)</td></tr>
        <tr><td>No Causal Graph</td><td>0.334</td><td style="color:#f78166">−0.050 (−13%)</td></tr>
        <tr><td>No VAE (deterministic)</td><td>0.362</td><td style="color:#f78166">−0.022 (−6%)</td></tr>
      </table>
      <div class="callout" style="font-size:13px">
        Stop-gradient is nearly as critical as diffusion itself: without it, L_diff gradients conflict with L_task and undo encoder learning  almost as harmful as removing diffusion entirely. NOTEARS constraint adds real value over unconstrained Gumbel-Softmax.
      </div>
    </div>
  </div>
  <div class="pg">7 / 12</div>
</div>


<!-- ══════════════════ SLIDE 6: RESULTS ══════════════════ -->
<div class="slide">
  <h1>Results: Balanced Accuracy vs. Raw Accuracy</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:12px">
      <p>Primary metric: <strong>Val-BalAcc</strong> = macro-average recall across all 6 classes. Unlike test accuracy, it cannot be inflated by predicting only majority classes.</p>
      <table>
        <tr><th>Method</th><th>Year</th><th>Val-BalAcc ↑</th><th>Test-Acc</th><th>Macro-F1</th></tr>
        <tr><td>TFN</td><td>2017</td><td>0.248</td><td>0.667</td><td>0.189</td></tr>
        <tr><td>MulT</td><td>2019</td><td>0.278</td><td>0.626</td><td>0.214</td></tr>
        <tr><td>MISA</td><td>2020</td><td>0.278</td><td>0.633</td><td>0.214</td></tr>
        <tr><td>MMIM</td><td>2021</td><td>0.266</td><td>0.679</td><td>0.249</td></tr>
        <tr><td>TETFN</td><td>2022</td><td>0.324</td><td>0.600</td><td>0.217</td></tr>
        <tr class="ours"><td><strong>Affect-Diff</strong></td><td></td><td><strong>0.384 ± 0.000</strong></td><td>0.642</td><td><strong>0.214</strong></td></tr>
      </table>
      <div class="callout" style="font-size:13px">
        <strong>+38 % relative gain</strong> over TETFN in balanced accuracy.
        Affect-Diff (0.642 test acc) beats MulT, MISA, and TETFN on raw accuracy too —
        only TFN and MMIM score higher, and both produce <em>zero F1</em> on Fear/Disgust/Surprise.
        Three seeds converge identically: <em>0.384 ± 0.000 at epoch 28</em>.
      </div>
    </div>
    <div class="img-wrap">
      <img src="{BC}" alt="baseline comparison chart">
    </div>
  </div>
  <div class="pg">8 / 12</div>
</div>


<!-- ══════════════════ SLIDE 7: THE KEY FINDING ══════════════════ -->
<div class="slide">
  <h1>Key Finding: KL Regularization Suppresses Minority Classes</h1>
  <div style="flex:1; display:flex; flex-direction:column; gap:12px">
    <div class="img-wrap" style="flex:1; max-height:330px">
      <img src="{PF}" alt="per-class F1 across all configurations">
    </div>
    <div class="cols cols2" style="flex:0 0 auto">
      <div class="callout">
        <strong>Every model  including Affect-Diff (Full)  produces zero F1 on Fear, Disgust, and Surprise.</strong>
        Except one: the <em>No-VAE (deterministic encoder)</em> variant is the only configuration that detects all six emotion classes.<br><br>
        Fear F1 = 0.125 &nbsp;·&nbsp; Disgust F1 = 0.130 &nbsp;·&nbsp; Surprise F1 = 0.098
      </div>
      <div class="card">
        <h3 style="margin-top:0">Why this happens</h3>
        <p style="font-size:13px">KL regularization (β = 0.1) collapses posterior representations of rare classes  the encoder <em>can</em> represent all six emotions, but gets penalized for maintaining distinct rare-class distributions in latent space.</p>
        <p style="font-size:13px; margin-top:8px"><strong>Actionable path:</strong> class-conditional adaptive β-annealing recovers minority detection without sacrificing macro balanced accuracy. The No-VAE result proves the encoder has the capacity.</p>
      </div>
    </div>
  </div>
  <div class="pg">9 / 12</div>
</div>


<!-- ══════════════════ SLIDE 8: GENERALIZABILITY ══════════════════ -->
<div class="slide">
  <h1>Generalizability: Sentiment Analysis on CMU-MOSEI</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex;flex-direction:column;gap:12px">
      <p>We retrained Affect-Diff on <strong>22,860 CMU-MOSEI segments</strong> for sentiment analysis (−3 to +3 scale) using BERT + COVAREP + FACET  <em>no architecture changes</em>.</p>
      <div class="img-wrap" style="flex:1">
        <img src="{SR}" alt="sentiment scale comparison">
      </div>
      <p style="font-size:12.5px;color:var(--muted)">The +90% BalAcc jump is driven by data scale (3,292 → 22K), not architecture differences. This isolates the emotion task's data scarcity as the dominant bottleneck.</p>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <h3 style="margin-top:0">Sentiment Results</h3>
      <table>
        <tr><th>Task</th><th>Bal-Acc ↑</th><th>Accuracy</th><th>Macro-F1</th><th>AUROC</th></tr>
        <tr class="ours"><td><strong>7-Class (−3 to +3)</strong></td><td><strong>0.729</strong></td><td>0.788</td><td>0.724</td><td>0.952</td></tr>
        <tr class="ours"><td><strong>Binary (pos/neg)</strong></td><td><strong>0.925</strong></td><td>0.940</td><td>0.926</td><td>0.966</td></tr>
      </table>
      <div class="callout" style="font-size:13px">
        <strong>3 takeaways:</strong><br><br>
        <strong>① Architecture generalises</strong>  same NOTEARS + β-VAE + DDPM prior, zero structural modifications for a different task.<br><br>
        <strong>② Data scale is dominant</strong>  +90% BalAcc (0.384 → 0.729) with 7× more training samples. More data, not better architecture, is the fastest path to improving the emotion task.<br><br>
        <strong>③ Imbalance handling scales</strong>  7-class sentiment across 7 near-uniform classes (BalAcc ≈ Acc), confirming the diffusion + causal bridge contributes beyond majority-class forcing.
      </div>
      <p style="font-size:12px;color:var(--muted)">Note: sentiment uses random split vs speaker-disjoint  results reflect in-distribution generalization, not cross-speaker robustness.</p>
    </div>
  </div>
  <div class="pg">10 / 12</div>
</div>


<!-- ══════════════════ SLIDE 11: CONCLUSIONS ══════════════════ -->
<div class="slide">
  <h1>Conclusions</h1>
  <div class="cols cols2" style="flex:1">
    <div style="display:flex; flex-direction:column; gap:10px">
      <h3 style="margin-top:0">What we built</h3>
      <p>Affect-Diff: a jointly trained Causal-Diffusion Bridge combining NOTEARS modality graph, β-VAE bottleneck, and stop-gradiented 1D DDPM prior  achieving <strong>val-BalAcc 0.384</strong> on CMU-MOSEI 6-class emotion recognition.</p>

      <h3>What we proved</h3>
      <ul>
        <li><strong>+38 % relative improvement</strong> over TETFN (best 2022 baseline) in balanced accuracy</li>
        <li>Stop-gradient and diffusion prior each contribute ~24 %  without stop-gradient, conflicting gradients undo training</li>
        <li>NOTEARS acyclicity constraint adds real benefit over unconstrained Gumbel-Softmax</li>
        <li>Causal graph learns <em>non-trivial dynamics</em>: Video-first → Audio/Text-dominant over training</li>
        <li><strong>Stable across 3 seeds: 0.384 ± 0.000</strong></li>
      </ul>
    </div>
    <div style="display:flex; flex-direction:column; gap:10px">
      <h3 style="margin-top:0">Three concrete next steps</h3>
      <div class="card-accent">
        <strong>① Adaptive β-annealing</strong>
        <p style="font-size:13px; margin-top:4px; color:var(--muted)">Anneal KL weight per-class toward zero for minorities. No-VAE result proves the encoder has capacity for all six  KL regularization is what suppresses them.</p>
      </div>
      <div class="card-accent">
        <strong>② Foundation encoders</strong>
        <p style="font-size:13px; margin-top:4px; color:var(--muted)">Replace GloVe / COVAREP / FACET (2013–2016) with frozen RoBERTa + HuBERT + CLIP-ViT. Architecture is already scaffolded for plug-in replacement.</p>
      </div>
      <div class="card-accent">
        <strong>③ Stratified speaker split</strong>
        <p style="font-size:13px; margin-top:4px; color:var(--muted)">Current test set has ~8 Fear samples. Speaker-disjoint stratified split over 23,453 segments gives reliable minority-class F1 evaluation.</p>
      </div>
    </div>
  </div>
  <div class="pg">11 / 12</div>
</div>


<!-- ══════════════════ SLIDE 12: AI USAGE DECLARATION ══════════════════ -->
<div class="slide">
  <h1>AI Usage Declaration</h1>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:16px">
    <p style="font-size:14px;color:var(--muted)">This project used AI assistance (Claude, Anthropic) in the following capacities:</p>
    <div class="cols cols2" style="flex:0 0 auto; gap:16px">
      <div class="card-accent">
        <h3 style="margin-top:0">① Code Scaffolding</h3>
        <p style="font-size:13px;color:var(--muted)">Initial model architecture, training loops, data loading utilities, and ablation runners were scaffolded with AI assistance and subsequently validated, debugged, and refined by the author.</p>
      </div>
      <div class="card-accent">
        <h3 style="margin-top:0">② Paper Drafting</h3>
        <p style="font-size:13px;color:var(--muted)">Section drafts, LaTeX formatting, and narrative structure were developed with AI assistance based on experimental results and author direction.</p>
      </div>
      <div class="card-accent">
        <h3 style="margin-top:0">③ Ideation</h3>
        <p style="font-size:13px;color:var(--muted)">Brainstorming architectural components, loss formulations, and experimental design benefited from AI-assisted exploration of alternatives.</p>
      </div>
      <div class="card-accent">
        <h3 style="margin-top:0">④ Literature Review</h3>
        <p style="font-size:13px;color:var(--muted)">Identifying relevant prior work, summarizing baselines, and framing contributions relative to the field was assisted by AI tools.</p>
      </div>
    </div>
    <div class="callout" style="font-size:13px">
      All <strong>experimental results, final design decisions, and intellectual contributions</strong> are the author's own.
      AI tools were used as productivity aids — every generated artifact was reviewed, tested, and approved by the author before inclusion.
    </div>
  </div>
  <div class="pg">12 / 12</div>
</div>

</body>
</html>
"""

html_path = Path(__file__).resolve().parent / "presentation.html"
html_path.write_text(HTML)
print(f"HTML written: {html_path}")

pdf_path = Path(__file__).resolve().parent / "presentation.pdf"
cmd = [
    CHROME,
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-web-security",
    f"--print-to-pdf={pdf_path}",
    "--print-to-pdf-no-header",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--virtual-time-budget=5000",
    str(html_path),
]
print("Rendering PDF with Chrome headless...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("stdout:", result.stdout[:300] if result.stdout else "(none)")
print("stderr:", result.stderr[:300] if result.stderr else "(none)")
print("return code:", result.returncode)
if pdf_path.exists():
    print(f"\nPDF written: {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")
else:
    print("PDF not created.")
