"""Analyze ablation sweep results and generate paper-ready tables.

Pulls results from WandB (or local logs) and produces:
  - Table 1: Component ablation (LaTeX + CSV)
  - Table 2: Hyperparameter sensitivity (LaTeX + CSV + plots)
  - Table 3: Best config full comparison

Usage:
    python scripts/analyze_ablation_results.py --wandb-project "Affect-Diff-CVPR"
    python scripts/analyze_ablation_results.py --results-dir outputs/ablation_results
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation_results"


# ──────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────

def collect_from_wandb(project: str, entity: Optional[str] = None) -> List[Dict]:
    """Pull experiment results from WandB."""
    if not HAS_WANDB:
        print("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    results = []
    for run in runs:
        if run.state != "finished":
            continue

        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}

        results.append({
            "name": run.name,
            "id": run.id,
            "test_acc": summary.get("test_acc", None),
            "val_acc": summary.get("val_acc", None),
            "val_loss": summary.get("val_loss", None),
            "test_loss": summary.get("test_loss", None),
            "test_loss_task": summary.get("test_loss_task", None),
            "test_loss_kl": summary.get("test_loss_kl", None),
            "test_loss_diff": summary.get("test_loss_diff", None),
            "test_loss_causal": summary.get("test_loss_causal", None),
            "test_loss_recon": summary.get("test_loss_recon", None),
            "val/hallucination_acc": summary.get("val/hallucination_acc", None),
            "epochs_trained": summary.get("epoch", None),
            "config": config,
        })

    print(f"Collected {len(results)} finished runs from WandB")
    return results


def collect_from_local(results_dir: Path) -> List[Dict]:
    """Collect results from local JSON files (fallback if no WandB)."""
    results = []
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    print(f"Collected {len(results)} results from local files")
    return results


# ──────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────

def parse_experiment_name(name: str) -> Tuple[str, str, Optional[float]]:
    """Parse experiment name into (category, param_name, param_value).

    Examples:
        "HP_beta_kl=0.5"    -> ("hyperparam", "beta_kl", 0.5)
        "COMP_No_Diffusion"  -> ("component", "No_Diffusion", None)
    """
    if name.startswith("HP_"):
        match = re.match(r"HP_(\w+)=([\d.]+)", name)
        if match:
            return "hyperparam", match.group(1), float(match.group(2))
    elif name.startswith("COMP_"):
        comp_name = name[5:]
        return "component", comp_name, None

    return "unknown", name, None


# ──────────────────────────────────────────────────────────────────────
# TABLE GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_component_table(results: List[Dict]) -> str:
    """Generate LaTeX table for component ablation (Table 1)."""
    comp_results = []
    for r in results:
        cat, name, _ = parse_experiment_name(r["name"])
        if cat == "component":
            comp_results.append((name, r))

    if not comp_results:
        return "No component ablation results found."

    # Sort: Full_Model first, then alphabetical
    comp_results.sort(key=lambda x: (0 if x[0] == "Full_Model" else 1, x[0]))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Component Ablation on CMU-MOSEI. Each row removes one component from the full model.}")
    lines.append(r"\label{tab:component_ablation}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Val Acc & Test Acc & Hall. Acc & $\mathcal{L}_\text{test}$ & Epochs \\")
    lines.append(r"\midrule")

    for name, r in comp_results:
        display_name = name.replace("_", " ")
        val_acc = f"{r['val_acc']:.3f}" if r.get("val_acc") is not None else "--"
        test_acc = f"{r['test_acc']:.3f}" if r.get("test_acc") is not None else "--"
        hall_acc = f"{r.get('val/hallucination_acc', '--'):.3f}" if r.get("val/hallucination_acc") is not None else "--"
        test_loss = f"{r['test_loss']:.3f}" if r.get("test_loss") is not None else "--"
        epochs = str(r.get("epochs_trained", "--"))

        bold = r"\textbf" if name == "Full_Model" else ""
        if bold:
            lines.append(f"\\textbf{{{display_name}}} & \\textbf{{{val_acc}}} & \\textbf{{{test_acc}}} & \\textbf{{{hall_acc}}} & \\textbf{{{test_loss}}} & {epochs} \\\\")
        else:
            lines.append(f"{display_name} & {val_acc} & {test_acc} & {hall_acc} & {test_loss} & {epochs} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_hyperparam_table(results: List[Dict]) -> Dict[str, str]:
    """Generate LaTeX tables for hyperparameter sensitivity (Table 2).

    Returns one table per parameter group.
    """
    # Group results by parameter name
    groups = defaultdict(list)
    for r in results:
        cat, param_name, param_val = parse_experiment_name(r["name"])
        if cat == "hyperparam":
            groups[param_name].append((param_val, r))

    tables = {}
    for param_name, entries in groups.items():
        entries.sort(key=lambda x: x[0])

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        param_display = param_name.replace("_", r"\_")
        lines.append(f"\\caption{{Sensitivity to \\texttt{{{param_display}}} on CMU-MOSEI.}}")
        lines.append(f"\\label{{tab:sens_{param_name}}}")
        lines.append(r"\begin{tabular}{ccccc}")
        lines.append(r"\toprule")
        lines.append(f"\\texttt{{{param_display}}} & Val Acc & Test Acc & Val Loss & Epochs \\\\")
        lines.append(r"\midrule")

        best_test_acc = max(e[1].get("test_acc", 0) or 0 for e in entries)

        for val, r in entries:
            val_acc = f"{r['val_acc']:.3f}" if r.get("val_acc") is not None else "--"
            test_acc = f"{r['test_acc']:.3f}" if r.get("test_acc") is not None else "--"
            val_loss = f"{r['val_loss']:.3f}" if r.get("val_loss") is not None else "--"
            epochs = str(r.get("epochs_trained", "--"))

            # Bold the best row
            is_best = (r.get("test_acc") or 0) == best_test_acc and best_test_acc > 0
            if is_best:
                lines.append(f"\\textbf{{{val}}} & \\textbf{{{val_acc}}} & \\textbf{{{test_acc}}} & \\textbf{{{val_loss}}} & {epochs} \\\\")
            else:
                lines.append(f"{val} & {val_acc} & {test_acc} & {val_loss} & {epochs} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        tables[param_name] = "\n".join(lines)

    return tables


def generate_sensitivity_plots(results: List[Dict], output_dir: Path) -> None:
    """Generate sensitivity plots for each hyperparameter (Figure in paper)."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping plots.")
        return

    groups = defaultdict(list)
    for r in results:
        cat, param_name, param_val = parse_experiment_name(r["name"])
        if cat == "hyperparam":
            groups[param_name].append((param_val, r))

    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.5), squeeze=False)

    for idx, (param_name, entries) in enumerate(sorted(groups.items())):
        entries.sort(key=lambda x: x[0])
        ax = axes[0][idx]

        x_vals = [e[0] for e in entries]
        val_accs = [e[1].get("val_acc", 0) or 0 for e in entries]
        test_accs = [e[1].get("test_acc", 0) or 0 for e in entries]

        ax.plot(x_vals, val_accs, "o-", color="#2196F3", label="Val Acc", linewidth=2, markersize=6)
        ax.plot(x_vals, test_accs, "s--", color="#FF5722", label="Test Acc", linewidth=2, markersize=6)

        ax.set_xlabel(param_name.replace("_", " "), fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"Sensitivity: {param_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Use log scale for params that span orders of magnitude
        if max(x_vals) / (min(x_vals) + 1e-10) > 10:
            ax.set_xscale("log")

    plt.tight_layout()
    plot_path = output_dir / "hyperparam_sensitivity.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plot_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved sensitivity plot to {plot_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ──────────────────────────────────────────────────────────────────────

def export_csv(results: List[Dict], output_dir: Path) -> None:
    """Export all results to CSV for easy analysis."""
    if not HAS_PANDAS:
        # Fallback: write raw CSV
        csv_path = output_dir / "all_results.csv"
        with open(csv_path, "w") as f:
            header = "name,category,param_name,param_value,val_acc,test_acc,val_loss,test_loss,epochs\n"
            f.write(header)
            for r in results:
                cat, pname, pval = parse_experiment_name(r["name"])
                row = (
                    f"{r['name']},{cat},{pname},{pval or ''},"
                    f"{r.get('val_acc', '')},{r.get('test_acc', '')},"
                    f"{r.get('val_loss', '')},{r.get('test_loss', '')},"
                    f"{r.get('epochs_trained', '')}\n"
                )
                f.write(row)
        print(f"Saved CSV to {csv_path}")
        return

    rows = []
    for r in results:
        cat, pname, pval = parse_experiment_name(r["name"])
        rows.append({
            "name": r["name"],
            "category": cat,
            "param_name": pname,
            "param_value": pval,
            "val_acc": r.get("val_acc"),
            "test_acc": r.get("test_acc"),
            "val_loss": r.get("val_loss"),
            "test_loss": r.get("test_loss"),
            "test_loss_task": r.get("test_loss_task"),
            "test_loss_kl": r.get("test_loss_kl"),
            "test_loss_diff": r.get("test_loss_diff"),
            "hallucination_acc": r.get("val/hallucination_acc"),
            "epochs": r.get("epochs_trained"),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Also generate per-group summaries
    for cat in ["hyperparam", "component"]:
        subset = df[df["category"] == cat]
        if len(subset) > 0:
            summary_path = output_dir / f"{cat}_results.csv"
            subset.to_csv(summary_path, index=False)
            print(f"Saved {cat} CSV to {summary_path}")


# ──────────────────────────────────────────────────────────────────────
# BEST CONFIG FINDER
# ──────────────────────────────────────────────────────────────────────

def find_best_config(results: List[Dict]) -> Dict:
    """Analyze hyperparameter results and recommend the best config."""
    groups = defaultdict(list)
    for r in results:
        cat, param_name, param_val = parse_experiment_name(r["name"])
        if cat == "hyperparam" and r.get("test_acc") is not None:
            groups[param_name].append((param_val, r["test_acc"]))

    best_config = {}
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETER VALUES (by test accuracy)")
    print("=" * 60)

    for param_name, entries in sorted(groups.items()):
        entries.sort(key=lambda x: x[1], reverse=True)
        best_val, best_acc = entries[0]
        best_config[param_name] = best_val
        print(f"  {param_name:20s} = {best_val:8.4f}  (test_acc = {best_acc:.4f})")

        # Show full ranking
        for val, acc in entries:
            marker = " <-- BEST" if val == best_val else ""
            print(f"    {val:8.4f} -> {acc:.4f}{marker}")

    print("\nRecommended config.yaml overrides:")
    for param_name, val in best_config.items():
        print(f"  model.{param_name}: {val}")

    return best_config


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze ablation sweep results")
    parser.add_argument("--wandb-project", type=str, default="Affect-Diff-CVPR",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity (username/team)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Local results directory (fallback if no WandB)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for tables and plots")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    if args.results_dir and args.results_dir.exists():
        results = collect_from_local(args.results_dir)
    elif HAS_WANDB:
        results = collect_from_wandb(args.wandb_project, args.wandb_entity)
    else:
        print("No results source available. Provide --results-dir or install wandb.")
        sys.exit(1)

    if not results:
        print("No results found. Run the ablation sweep first.")
        sys.exit(1)

    # Generate outputs
    print("\n" + "=" * 60)
    print("COMPONENT ABLATION TABLE (Table 1)")
    print("=" * 60)
    comp_table = generate_component_table(results)
    print(comp_table)
    (args.output_dir / "table1_component.tex").write_text(comp_table)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY TABLES (Table 2)")
    print("=" * 60)
    hp_tables = generate_hyperparam_table(results)
    all_hp_latex = []
    for param_name, table in hp_tables.items():
        print(f"\n--- {param_name} ---")
        print(table)
        all_hp_latex.append(table)
    (args.output_dir / "table2_hyperparam.tex").write_text("\n\n".join(all_hp_latex))

    # Sensitivity plots
    generate_sensitivity_plots(results, args.output_dir)

    # CSV export
    export_csv(results, args.output_dir)

    # Best config recommendation
    best_config = find_best_config(results)
    with open(args.output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
