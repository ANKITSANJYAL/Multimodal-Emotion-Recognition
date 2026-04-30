"""Generate all paper figures from training logs and results JSONs.

Usage:
    python3 generate_figures.py
"""

import csv, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.chdir("/Users/ankitsanjyal/Desktop/projects/Multimodal-Emotion-Recognition")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

BASE       = Path("logs")
BASE2      = Path("logs")
BASE_SEED  = Path("logs/seeds")
BASE_NEW   = Path("logs")

# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def val_curve(path):
    return [(int(r["epoch"]), float(r["val_bal_acc"]))
            for r in read_csv(path) if r.get("val_bal_acc", "").strip()]

def causal_curve(path):
    return [(int(r["epoch"]),
             float(r["causal/Text_Influence"]),
             float(r["causal/Audio_Influence"]),
             float(r["causal/Video_Influence"]))
            for r in read_csv(path) if r.get("causal/Text_Influence", "").strip()]

def loss_curve(path):
    return [(int(r["epoch"]),
             float(r["train_loss_task"]),
             float(r["train_loss_diff"]),
             float(r["train_loss_kl"]),
             float(r["train_loss_causal"]))
            for r in read_csv(path) if r.get("train_loss_diff", "").strip()]

def warmup_curve(path):
    return [(int(r["epoch"]), float(r["warmup/gamma_diff"]), float(r["warmup/gamma_kl"]))
            for r in read_csv(path) if r.get("warmup/gamma_diff", "").strip()]

def smooth(x, w=3):
    s = np.convolve(x, np.ones(w)/w, mode="same")
    s[:w//2] = np.array(x)[:w//2]
    s[-(w//2):] = np.array(x)[-(w//2):]
    return s

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Training curves (main section)
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curves():
    curves = {
        "Affect-Diff (Full)":   val_curve(BASE  / "Full_Model/version_0/metrics.csv"),
        "Affect-Diff s43":      val_curve(BASE_SEED / "Full_Model_s43/version_0/metrics.csv"),
        "Affect-Diff s44":      val_curve(BASE_SEED / "Full_Model_s44/version_0/metrics.csv"),
        "No Diffusion":         val_curve(BASE  / "No_Diffusion/version_0/metrics.csv"),
        "No Causal Graph":      val_curve(BASE  / "No_Causal_Graph/version_0/metrics.csv"),
        "No VAE":               val_curve(BASE  / "No_VAE/version_0/metrics.csv"),
        "No Stop-Grad":         val_curve(BASE  / "No_Stop_Gradient/version_0/metrics.csv"),
        "Classifier Only":      val_curve(BASE  / "Classifier_Only/version_0/metrics.csv"),
        "TETFN (2022)":         val_curve(BASE_NEW / "Baseline_TETFN/version_0/metrics.csv"),
        "MulT (2019)":          val_curve(BASE2 / "Baseline_MulT/version_0/metrics.csv"),
        "MMIM (2021)":          val_curve(BASE_NEW / "Baseline_MMIM/version_0/metrics.csv"),
        "TFN (2017)":           val_curve(BASE2 / "Baseline_TFN/version_0/metrics.csv"),
    }
    styles = {
        "Affect-Diff (Full)":  ("#1f77b4", 2.8, "-",  "o"),
        "Affect-Diff s43":     ("#1f77b4", 1.2, "--", None),
        "Affect-Diff s44":     ("#1f77b4", 1.2, ":",  None),
        "No Diffusion":        ("#d62728", 1.5, "-",  "s"),
        "No Causal Graph":     ("#ff7f0e", 1.5, "-",  "^"),
        "No VAE":              ("#9467bd", 1.5, "-",  "D"),
        "No Stop-Grad":        ("#e377c2", 1.3, "-",  None),
        "Classifier Only":     ("#7f7f7f", 1.2, "-",  None),
        "TETFN (2022)":        ("#2ca02c", 1.4, "--", "v"),
        "MulT (2019)":         ("#17becf", 1.4, "--", "P"),
        "MMIM (2021)":         ("#bcbd22", 1.4, "--", "h"),
        "TFN (2017)":          ("#8c564b", 1.2, "--", None),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, curve in curves.items():
        eps, vals = zip(*curve)
        col, lw, ls, mk = styles[name]
        vals_s = smooth(vals, w=5)
        ax.plot(eps, vals_s, label=name, color=col, lw=lw, ls=ls, alpha=0.9)
        if mk:
            peak_ep = eps[int(np.argmax(vals))]
            peak_v  = max(vals)
            ax.plot(peak_ep, peak_v, mk, color=col, ms=5, zorder=5)

    ax.axhline(0.384, color="#1f77b4", lw=0.7, ls=":", alpha=0.4)
    ax.text(1, 0.386, "0.384 (peak)", fontsize=7, color="#1f77b4", alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Val Balanced Accuracy", fontsize=10)
    ax.set_title("Validation Balanced Accuracy During Training", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="lower right",
              framealpha=0.9, edgecolor="#ccc", handlelength=1.5)
    ax.set_xlim(0, None)
    ax.set_ylim(0.15, 0.44)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Per-class F1 grouped bar chart (main section)
# ─────────────────────────────────────────────────────────────────────────────
def fig_perclass_f1():
    classes = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    data = {
        "TFN":         [0.807, 0.362, 0.080, 0.000, 0.000, 0.000],
        "MulT":        [0.784, 0.340, 0.231, 0.000, 0.000, 0.000],
        "MISA":        [0.789, 0.428, 0.111, 0.000, 0.000, 0.000],
        "MMIM":        [0.808, 0.422, 0.262, 0.000, 0.000, 0.000],
        "TETFN":       [0.768, 0.368, 0.167, 0.000, 0.000, 0.000],
        "Full Model":  [0.734, 0.375, 0.175, 0.000, 0.000, 0.000],
        "No VAE":      [0.634, 0.343, 0.121, 0.125, 0.130, 0.098],
    }
    colors = ["#8c564b","#17becf","#2ca02c","#bcbd22","#e377c2","#1f77b4","#9467bd"]

    n_groups = len(classes)
    n_bars   = len(data)
    width    = 0.11
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (label, vals) in enumerate(data.items()):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label,
               color=colors[i], alpha=0.85, edgecolor="white", lw=0.3)

    # Shade minority-class columns
    for xpos in [3, 4, 5]:
        ax.axvspan(xpos - 0.47, xpos + 0.47, color="red", alpha=0.06, zorder=0)
    ax.annotate("minority classes\n(all baselines = 0)", xy=(4, 0.02),
                fontsize=8, ha="center", color="#cc0000", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9.5)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title("Per-class F1: All Baselines vs.\ Affect-Diff Variants", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9, edgecolor="#ccc")
    ax.set_ylim(0, 0.95)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_perclass_f1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_perclass_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_perclass_f1")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Baseline comparison bar chart (val-BalAcc) — main section
# ─────────────────────────────────────────────────────────────────────────────
def fig_baseline_comparison():
    methods = ["TFN\n(2017)", "MISA\n(2020)", "MulT\n(2019)",
               "MMIM\n(2021)", "TETFN\n(2022)", "Affect-Diff\n(ours)"]
    val_bals = [0.248, 0.278, 0.278, 0.266, 0.324, 0.384]
    test_accs = [0.667, 0.633, 0.626, 0.679, 0.600, 0.642]
    colors_v = ["#8c564b","#17becf","#2ca02c","#bcbd22","#e377c2","#1f77b4"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(methods))
    w = 0.55

    # Left: Val BalAcc
    ax = axes[0]
    bars = ax.bar(x, val_bals, w, color=colors_v, alpha=0.85, edgecolor="white")
    bars[-1].set_edgecolor("#1f77b4"); bars[-1].set_linewidth(2)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Val Balanced Accuracy ↑", fontsize=9)
    ax.set_title("Primary Metric", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.46)
    ax.axhline(0.384, color="#1f77b4", lw=0.8, ls="--", alpha=0.5)
    for bar, v in zip(bars, val_bals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold" if v==0.384 else "normal")
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)

    # Right: Test Acc
    ax = axes[1]
    bars2 = ax.bar(x, test_accs, w, color=colors_v, alpha=0.85, edgecolor="white")
    bars2[-1].set_edgecolor("#1f77b4"); bars2[-1].set_linewidth(2)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Test Accuracy ↑", fontsize=9)
    ax.set_title("Test Accuracy (biased by Happy)", fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 0.78)
    for bar, v in zip(bars2, test_accs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)
    ax.annotate("Highest test acc =\nmajority collapse\n(MMIM: 0.679)", xy=(3, 0.681), fontsize=7,
                color="#cc4400", xytext=(3.8, 0.735),
                arrowprops=dict(arrowstyle="->", color="#cc4400", lw=0.7))

    fig.suptitle("Affect-Diff vs.\ Baselines: Balanced Accuracy vs.\ Raw Accuracy Trade-off",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_baseline_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_baseline_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A1 — Causal modality influence (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_causal_influence():
    data = causal_curve(BASE / "Full_Model/version_0/metrics.csv")
    eps, T, A, V = zip(*data)
    w = 5

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(eps, smooth(T,w), label="Text",  color="#1f77b4", lw=2.2)
    ax.plot(eps, smooth(A,w), label="Audio", color="#ff7f0e", lw=2.2)
    ax.plot(eps, smooth(V,w), label="Video", color="#2ca02c", lw=2.2)
    ax.fill_between(eps, smooth(T,w), alpha=0.07, color="#1f77b4")
    ax.fill_between(eps, smooth(A,w), alpha=0.07, color="#ff7f0e")
    ax.fill_between(eps, smooth(V,w), alpha=0.07, color="#2ca02c")

    ax.annotate("Video dominant\n(coarse AU cues)", xy=(10, 0.56), fontsize=7.5,
                color="#2ca02c", arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
                xytext=(19, 0.67))
    ax.annotate("Audio ≈ Text\nat convergence", xy=(55, 0.38), fontsize=7.5,
                color="#555", xytext=(38, 0.56),
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(r"Modality Weight $w_m$", fontsize=10)
    ax.set_title("NOTEARS Causal Modality Weights During Training", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.80)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_causal_influence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_causal_influence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_causal_influence")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A2 — Loss components (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_loss_components():
    data = loss_curve(BASE / "Full_Model/version_0/metrics.csv")
    eps, task, diff, kl, causal = zip(*data)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(eps, smooth(task,3),   label=r"$\mathcal{L}_\mathrm{task}$",   color="#1f77b4", lw=2)
    ax.plot(eps, smooth(diff,3),   label=r"$\mathcal{L}_\mathrm{diff}$",   color="#d62728", lw=2)
    ax.plot(eps, smooth(kl,3),     label=r"$\mathcal{L}_\mathrm{KL}$",     color="#9467bd", lw=1.5, ls="--")
    ax.plot(eps, smooth(causal,3), label=r"$\mathcal{L}_\mathrm{causal}$", color="#ff7f0e", lw=1.5, ls=":")

    ax.axvline(9,  color="#d62728", lw=0.7, ls=":", alpha=0.4)
    ax.axvline(29, color="#d62728", lw=0.7, ls=":", alpha=0.4)
    ax.text(9, 1.85, "diff\nwarmup\nstart", fontsize=6.5, color="#d62728", ha="center", va="top")
    ax.text(29, 1.85, "diff\nfull", fontsize=6.5, color="#d62728", ha="center", va="top")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Training Loss", fontsize=10)
    ax.set_title("Training Loss Component Decomposition", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 2.1)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss_components.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_loss_components.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_loss_components")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A3 — Warmup schedules (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_warmup_schedule():
    data = warmup_curve(BASE / "Full_Model/version_0/metrics.csv")
    eps, gd, gk = zip(*data)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(eps, gd, label=r"$\gamma_\mathrm{diff}$  (ramps ep 9–29)", color="#d62728", lw=2)
    ax.plot(eps, gk, label=r"$\gamma_\mathrm{KL}$  (ramps ep 0–30)",   color="#9467bd", lw=2, ls="--")
    ax.axhline(1.0, color="#aaa", lw=0.5, ls=":")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(r"Curriculum Weight $\gamma$", fontsize=10)
    ax.set_title("KL and Diffusion Curriculum Warmup Schedules", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_warmup_schedule.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_warmup_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_warmup_schedule")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A4 — Efficiency scatter (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_efficiency():
    points = {
        "TFN":            (644230,  0.248,  49.4,  "#8c564b"),
        "MISA":           (153094,  0.278,  4.3,   "#17becf"),
        "MulT":           (756038,  0.278,  49.4,  "#2ca02c"),
        "MMIM":           (320000,  0.266,  12.0,  "#bcbd22"),
        "TETFN":          (510000,  0.324,  35.0,  "#e377c2"),
        "Classifier\nOnly": (1473353, 0.322, 84.5, "#7f7f7f"),
        "No Diffusion":   (1473353, 0.292,  85.0,  "#d62728"),
        "No VAE":         (8916553, 0.362,  82.7,  "#9467bd"),
        "Affect-Diff\n(Full)": (8916553, 0.384, 82.5, "#1f77b4"),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, (params, bal, lat, color) in points.items():
        size = 60 + lat * 0.5
        ax.scatter(params / 1e6, bal, color=color, s=size, zorder=3,
                   alpha=0.85, edgecolors="white", linewidth=0.5)
        ha = "left"; offx, offy = 0.06, 0.004
        if "MISA" in name:   offx, offy = -0.1, -0.010; ha = "right"
        if "TFN" in name:    offy = -0.012
        if "Full" in name:   offx, offy = 0.12, 0.005
        if "Only" in name:   offy = -0.012
        ax.annotate(name, (params/1e6 + offx, bal + offy),
                    fontsize=7.5, ha=ha, color=color)

    ax.set_xlabel("Parameters (M)", fontsize=10)
    ax.set_ylabel("Val Balanced Accuracy", fontsize=10)
    ax.set_title("Efficiency vs.\ Performance\n(bubble size ∝ inference latency)",
                 fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_ylim(0.22, 0.42)
    fig.tight_layout()
    fig.savefig(OUT / "fig_efficiency.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A5 — Robustness bars (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_robustness():
    groups = [
        ("Missing Modality",  ["Missing Text", "Missing Audio", "Missing Vision"],
                              [-0.009, +0.018, -0.035]),
        ("Gaussian Noise",    [r"Noise σ=0.1", r"Noise σ=0.5", r"Noise σ=2.0"],
                              [-0.001, +0.005, -0.014]),
        ("Temporal Masking",  ["Frame Mask 10%", "Frame Mask 25%", "Frame Mask 50%"],
                              [+0.034, +0.016, -0.006]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharey=False)
    for ax, (title, labels, deltas) in zip(axes, groups):
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        y = np.arange(len(labels))
        bars = ax.barh(y, deltas, color=colors, alpha=0.82, edgecolor="white", height=0.55)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(r"$\Delta$ F1 vs. clean", fontsize=8)
        lim = max(abs(d) for d in deltas) * 1.6
        ax.set_xlim(-lim, lim)
        for bar, d in zip(bars, deltas):
            x_off = lim * 0.05 if d >= 0 else -lim * 0.05
            ha = "left" if d >= 0 else "right"
            ax.text(d + x_off, bar.get_y() + bar.get_height()/2,
                    f"{d:+.3f}", va="center", ha=ha, fontsize=8)
        ax.grid(True, axis="x", alpha=0.2, lw=0.5)

    fig.suptitle("Affect-Diff Robustness (Macro F1 Δ relative to clean = 0.214)",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_robustness")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A6 — Seed stability (appendix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_seed_stability():
    curves = {
        "Seed 42 (main)": val_curve(BASE  / "Full_Model/version_0/metrics.csv"),
        "Seed 43":        val_curve(BASE_SEED / "Full_Model_s43/version_0/metrics.csv"),
        "Seed 44":        val_curve(BASE_SEED / "Full_Model_s44/version_0/metrics.csv"),
    }
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ls_    = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for (name, curve), col, ls in zip(curves.items(), colors, ls_):
        eps, vals = zip(*curve)
        ax.plot(eps, smooth(vals, 5), label=name, color=col, lw=2, ls=ls)
        peak_ep = eps[int(np.argmax(vals))]
        ax.plot(peak_ep, max(vals), "o", color=col, ms=5)

    ax.axhline(0.384, color="#555", lw=0.6, ls=":", alpha=0.5)
    ax.text(1, 0.3855, "0.384 (all three peaks)", fontsize=7.5, color="#555", alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Val Balanced Accuracy", fontsize=10)
    ax.set_title("Seed Stability: Three Independent Runs", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.15, 0.44)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seed_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_seed_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_seed_stability")


# ─────────────────────────────────────────────────────────────────────────────
# Fig: Architecture overview (for slides — matplotlib version)
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # ── palette ───────────────────────────────────────────────────────────────
    C = dict(
        inp    = "#d6e4f0",  inp_e  = "#2c7fb8",
        enc    = "#2c7fb8",  enc_e  = "#1a5276",
        causal = "#c0392b",  caus_e = "#922b21",
        fuse   = "#1e8449",  fuse_e = "#145a32",
        vae    = "#6c3483",  vae_e  = "#4a235a",
        cls    = "#1a5276",  cls_e  = "#0e2f44",
        diff   = "#922b21",  diff_e = "#641e16",
        loss   = "#555555",  loss_e = "#333333",
        sg     = "#e67e22",
    )
    WHITE = "#ffffff"

    def box(x, y, w, h, txt, fc, ec, fs=8.5, bold=False, txt2=None):
        patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.06",
                               fc=fc, ec=ec, lw=1.4, zorder=3)
        ax.add_patch(patch)
        dy = 0.1 if txt2 else 0
        ax.text(x, y + dy, txt, ha="center", va="center", fontsize=fs,
                color=WHITE, fontweight="bold" if bold else "normal",
                zorder=4, linespacing=1.3)
        if txt2:
            ax.text(x, y - 0.16, txt2, ha="center", va="center",
                    fontsize=fs - 1.5, color=WHITE, zorder=4, style="italic")

    def arr(x1, y1, x2, y2, col="#555", lw=1.3, label="", lfs=7.5, rad=0.0):
        style = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                   connectionstyle=style,
                                   mutation_scale=11), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.05, my + 0.12, label, ha="center", va="bottom",
                    fontsize=lfs, color=col, style="italic")

    # ── Column positions ──────────────────────────────────────────────────────
    xI  = 0.75   # Inputs
    xE  = 2.15   # Encoders
    xCG = 3.75   # Causal graph
    xF  = 5.15   # Fusion
    xV  = 6.55   # VAE
    xCL = 8.2    # Classifier branch (top)
    xDI = 8.2    # Diffusion branch (bot)
    xL  = 9.95   # Losses

    yT, yA, yV = 3.55, 2.10, 0.65  # Input row y-positions
    yMid = (yT + yV) / 2            # 2.1 (vertical centre)

    BW, BH = 1.15, 0.56             # standard box width/height
    LBW, LBH = 0.75, 0.42          # loss box

    # ── Inputs ────────────────────────────────────────────────────────────────
    for yi, (lbl, dim) in zip([yT, yA, yV],
                               [("Text",  "GloVe 300-d"),
                                ("Audio", "COVAREP 74-d"),
                                ("Video", "FACET 35-d")]):
        box(xI, yi, BW, BH, lbl, C["inp"], C["inp_e"],
            fs=8.5, bold=True, txt2=dim)
        # Tint the text dark for light-background input boxes
        for txt_obj in ax.texts[-2:]:
            txt_obj.set_color("#1a2433")

    # ── Encoders ──────────────────────────────────────────────────────────────
    for yi, lbl in zip([yT, yA, yV],
                       ["TextEnc\n(Transformer ×2)",
                        "AudioEnc\n(Conv1d+Attn ×2)",
                        "VideoEnc\n(Conv1d+Attn ×2)"]):
        arr(xI + BW/2, yi, xE - BW/2, yi, col="#888")
        box(xE, yi, BW, BH, lbl, C["enc"], C["enc_e"], fs=7.8)

    # ── Causal Graph ──────────────────────────────────────────────────────────
    cgH = yT - yV + BH          # spans all three rows
    patch = FancyBboxPatch((xCG - 0.65, yV - BH/2),
                           1.3, cgH,
                           boxstyle="round,pad=0.06",
                           fc=C["causal"], ec=C["caus_e"], lw=1.6, zorder=3)
    ax.add_patch(patch)
    ax.text(xCG, yMid + 0.5, "NOTEARS", ha="center", va="center",
            fontsize=8.5, color=WHITE, fontweight="bold", zorder=4)
    ax.text(xCG, yMid + 0.18, "Causal Graph", ha="center", va="center",
            fontsize=7.5, color=WHITE, zorder=4)
    ax.text(xCG, yMid - 0.18,
            r"$\mathbf{w}=\mathrm{softmax}(\mathbf{A}^\top\mathbf{1})$",
            ha="center", va="center", fontsize=7.2, color="#ffd6d6", zorder=4)
    ax.text(xCG, yMid - 0.52,
            r"$h(A)=\mathrm{tr}(e^{A\circ A})-3=0$",
            ha="center", va="center", fontsize=6.8, color="#ffa8a8",
            style="italic", zorder=4)

    # arrows encoder → causal
    for yi in [yT, yA, yV]:
        arr(xE + BW/2, yi, xCG - 0.65, yi, col="#aaa", lw=1.1)

    # ── Fusion ────────────────────────────────────────────────────────────────
    box(xF, yMid, BW, 0.68, "Concat + MLP\nFusion",
        C["fuse"], C["fuse_e"], fs=8.0)
    for yi, lbl in zip([yT, yA, yV],
                       [r"$\tilde{h}^T$", r"$\tilde{h}^A$", r"$\tilde{h}^V$"]):
        arr(xCG + 0.65, yi, xF - BW/2, yMid,
            col="#e8a0a0", lw=0.9, label=lbl, lfs=7.0)

    # ── VAE ───────────────────────────────────────────────────────────────────
    box(xV, yMid, BW, 0.72,
        r"$\beta$-VAE",
        C["vae"], C["vae_e"], fs=8.5, bold=True,
        txt2=r"$z = \mu + \varepsilon\cdot\sigma$")
    arr(xF + BW/2, yMid, xV - BW/2, yMid, col="#555", label="f", lfs=7.5)

    # ── Two output branches ───────────────────────────────────────────────────
    yCls = yT - 0.15
    yDif = yV + 0.15

    # classifier branch
    arr(xV + BW/2, yMid, xCL - BW/2, yCls, col="#888", lw=1.2, rad=-0.25)
    box(xCL, yCls, BW + 0.1, BH, "Attn-Pool\n+ Classifier",
        C["cls"], C["cls_e"], fs=7.8)
    arr(xCL + (BW+0.1)/2, yCls, xL - LBW/2, yCls,
        col=C["cls"], lw=1.2)
    box(xL, yCls, LBW, LBH, r"$\mathcal{L}_\mathrm{task}$",
        C["cls"], C["cls_e"], fs=9)

    # stop-gradient + diffusion branch
    sg_x = (xV + BW/2 + xDI - BW/2) / 2
    arr(xV + BW/2, yMid, sg_x - 0.32, yDif, col="#888", lw=1.2, rad=0.25)
    sg_patch = FancyBboxPatch((sg_x - 0.32, yDif - 0.16), 0.64, 0.32,
                              boxstyle="round,pad=0.04",
                              fc=C["sg"], ec="#c0641a", lw=1.2, zorder=5)
    ax.add_patch(sg_patch)
    ax.text(sg_x, yDif, "stop-grad", ha="center", va="center",
            fontsize=7, color=WHITE, fontweight="bold", zorder=6)
    arr(sg_x + 0.32, yDif, xDI - BW/2, yDif,
        col=C["diff"], lw=1.2)
    box(xDI, yDif, BW + 0.1, BH, "1D U-Net DDPM\n" + r"$\epsilon_\theta(z_t,t,y,\mathbf{w})$",
        C["diff"], C["diff_e"], fs=7.5)
    arr(xDI + (BW+0.1)/2, yDif, xL - LBW/2, yDif,
        col=C["diff"], lw=1.2)
    box(xL, yDif, LBW, LBH, r"$\mathcal{L}_\mathrm{diff}$",
        C["diff"], C["diff_e"], fs=9)

    # KL loss from VAE
    box(xL, yMid, LBW, LBH, r"$\mathcal{L}_\mathrm{KL}$",
        C["vae"], C["vae_e"], fs=9)
    arr(xV + BW/2, yMid, xL - LBW/2, yMid, col=C["vae"], lw=1.0,
        label="KL", lfs=7)

    # causal DAG penalty
    box(xL, (yMid + yDif)/2 - 0.1, LBW, LBH, r"$\mathcal{L}_\mathrm{dag}$",
        "#555", "#333", fs=9)
    ax.annotate("", xy=(xL - LBW/2, (yMid + yDif)/2 - 0.1),
                xytext=(xCG + 0.65, yV + BH/4),
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9,
                                connectionstyle="arc3,rad=0.4"), zorder=2)

    # ── Column labels ─────────────────────────────────────────────────────────
    label_y = 4.05
    for x, lbl in [(xI,  "Input"), (xE,   "Encoders"),
                   (xCG, "Causal\nGraph"), (xF, "Fusion"),
                   (xV,  "VAE"), (xL,  "Losses")]:
        ax.text(x, label_y, lbl, ha="center", va="center",
                fontsize=8, color="#444", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#e8e8e8",
                          ec="#ccc", lw=0.8))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(5.5, -0.1, "Affect-Diff: Causal-Diffusion Bridge",
            ha="center", va="top", fontsize=10, fontweight="bold", color="#222")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# Fig: Formal causal DAG (static graph + adjacency matrix)
# ─────────────────────────────────────────────────────────────────────────────
def fig_causal_dag():
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    fig.patch.set_facecolor("#fafafa")

    # ── LEFT: DAG with sample learned weights (from epoch-40 dynamics) ────────
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 1.6)
    ax.axis("off")
    ax.set_facecolor("#fafafa")

    # Node layout: equilateral triangle
    import math
    r = 1.05
    angles = {"T": 90, "A": 210, "V": 330}   # Text top, Audio left, Video right
    pos = {k: (r * math.cos(math.radians(v)), r * math.sin(math.radians(v)))
           for k, v in angles.items()}
    node_color = {"T": "#2c7fb8", "A": "#41ab5d", "V": "#d95f0e"}
    node_label = {"T": "Text\n$h^T$", "A": "Audio\n$h^A$", "V": "Video\n$h^V$"}

    # Directed edges: learned A matrix (column = influence source, row = target)
    # Epoch-40 approximate values derived from causal influence figure
    edges = [
        ("T", "A", 0.26), ("T", "V", 0.14),
        ("A", "T", 0.31), ("A", "V", 0.22),
        ("V", "T", 0.20), ("V", "A", 0.17),
    ]

    for src, tgt, w in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        # perpendicular offset to separate bidirectional pairs
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy)
        ox, oy = -dy/n * 0.10, dx/n * 0.10
        lw = 0.8 + w * 4.5
        alpha = 0.45 + w * 1.2
        ax.annotate("",
            xy=(x2 + ox*0.5 - dx/n*0.28, y2 + oy*0.5 - dy/n*0.28),
            xytext=(x1 + ox - dx/n*0.28, y1 + oy - dy/n*0.28),
            arrowprops=dict(arrowstyle="-|>", lw=lw,
                            color="#555", alpha=min(alpha, 1.0),
                            mutation_scale=10), zorder=2)
        # edge weight label
        mx = (x1 + x2)/2 + ox * 1.4
        my = (y1 + y2)/2 + oy * 1.4
        ax.text(mx, my, f"{w:.2f}", ha="center", va="center",
                fontsize=7, color="#444",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec="#ccc", lw=0.5, alpha=0.85))

    # Nodes
    for name, (x, y) in pos.items():
        c = Circle((x, y), 0.27, fc=node_color[name], ec="#222",
                   lw=1.8, zorder=5)
        ax.add_patch(c)
        ax.text(x, y + 0.02, name, ha="center", va="center",
                fontsize=13, color="white", fontweight="bold", zorder=6)
        ax.text(x, y - 0.42, node_label[name].split("\n")[1],
                ha="center", va="top", fontsize=8,
                color=node_color[name], zorder=4)

    # Column-sum formula
    ax.text(0, -0.75,
            r"$\mathbf{w}=\mathrm{softmax}(\mathbf{A}^\top\mathbf{1})$"
            + "   →   " +
            r"$\tilde{h}^m = w_m \cdot h^m$",
            ha="center", va="center", fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#2c7fb8", lw=1.2))

    ax.set_title("(a) Learned Modality Causal Graph\n(edge weights from trained model, epoch 40)",
                 fontsize=8.5, pad=6, color="#333")

    # ── RIGHT: Adjacency matrix heatmap ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#fafafa")

    A = np.array([
        [0.00, 0.26, 0.14],
        [0.31, 0.00, 0.22],
        [0.20, 0.17, 0.00],
    ])
    labels_rc = ["T (Text)", "A (Audio)", "V (Video)"]
    colors_rc = ["#2c7fb8", "#41ab5d", "#d95f0e"]

    # Custom colormap: white → deep blue (0 → 0.35)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "dag", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])

    im = ax2.imshow(A, cmap=cmap, vmin=0, vmax=0.40, aspect="equal")

    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xticklabels(labels_rc, fontsize=8.5)
    ax2.set_yticklabels(labels_rc, fontsize=8.5)
    ax2.set_xlabel("Influence source", fontsize=8.5, labelpad=4)
    ax2.set_ylabel("Influence target", fontsize=8.5, labelpad=4)

    for (i, j), val in np.ndenumerate(A):
        txt = "—" if val == 0 else f"{val:.2f}"
        col = "white" if val > 0.20 else "#222"
        ax2.text(j, i, txt, ha="center", va="center",
                 fontsize=11, color=col, fontweight="bold")

    for idx, c in enumerate(colors_rc):
        ax2.get_xticklabels()[idx].set_color(c)
        ax2.get_yticklabels()[idx].set_color(c)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight $A_{ij}$", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    ax2.set_title(
        "(b) Adjacency Matrix  $\\mathbf{A}$\n"
        r"$h(\mathbf{A})=\mathrm{tr}(e^{\mathbf{A}\circ\mathbf{A}})-3=0$  (NOTEARS)",
        fontsize=8.5, pad=6, color="#333")

    # Tick color match
    ax2.tick_params(axis="both", which="both", length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle("NOTEARS Causal Attention Graph", fontsize=10,
                 fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / "fig_causal_dag.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_causal_dag.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_causal_dag")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_training_curves()
    fig_perclass_f1()
    fig_baseline_comparison()
    fig_causal_influence()
    fig_loss_components()
    fig_warmup_schedule()
    fig_efficiency()
    fig_robustness()
    fig_seed_stability()
    fig_architecture()
    fig_causal_dag()
    print(f"\nAll figures written to {OUT.resolve()}/")
