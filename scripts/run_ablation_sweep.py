"""Ablation sweep runner for Affect-Diff (CVPR 2026).

Generates and executes ablation experiments sequentially on a single
GPU node (2× V100). Designed to be run inside a tmux session.

Features:
  - Crash recovery: tracks completed experiments in a JSON manifest.
    Re-running the same command skips already-finished experiments.
  - Deduplication: HP default runs that duplicate the component baseline
    are automatically removed.
  - Per-experiment results saved after each run for analysis.
  - Estimated time printed before starting.

Usage (inside a tmux session):
    # Dry run — see all commands
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch dry

    # Run all ablations sequentially
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch run

    # Run only hyperparameter sensitivity
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode hyperparam --launch run

    # Run only component ablation
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode component --launch run

    # Resume after a crash (automatically skips completed experiments)
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "configs" / "sweeps"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "ablation_results"
LOGS_DIR = PROJECT_ROOT / "logs" / "ablation"
MANIFEST_PATH = RESULTS_DIR / "completed.json"

# Training settings shared across all ablation runs
# 100 epochs + patience 30 with val_acc/max monitoring gives adequate convergence.
# Estimated ~15 min per run on 2× V100 (early stopping fires around epoch 40-60).
SHARED_OVERRIDES = {
    "trainer.epochs": 100,
    "trainer.patience": 30,
    "trainer.devices": 2,
    "data.batch_size": 64,
    "data.num_workers": 8,
}

ESTIMATED_MINUTES_PER_RUN = 15  # ~4-5 sec/epoch × ~50 epochs with early stopping


# ──────────────────────────────────────────────────────────────────────
# MANIFEST (crash recovery)
# ──────────────────────────────────────────────────────────────────────

def load_completed() -> Dict:
    """Load manifest of completed experiments."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def mark_completed(name: str, result: Dict) -> None:
    """Mark an experiment as completed in the manifest."""
    completed = load_completed()
    completed[name] = {
        "finished_at": datetime.now().isoformat(),
        "exit_code": result.get("exit_code"),
        "test_acc": result.get("test_acc"),
        "val_acc": result.get("val_acc"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(completed, f, indent=2)


def get_completed_names() -> Set[str]:
    """Get set of experiment names that already finished successfully."""
    completed = load_completed()
    return {name for name, info in completed.items() if info.get("exit_code") == 0}


# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT GENERATION
# ──────────────────────────────────────────────────────────────────────

def load_hyperparam_sweep() -> Dict:
    with open(SWEEP_DIR / "hyperparameter_ablation.yaml") as f:
        return yaml.safe_load(f)


def load_component_sweep() -> List[Dict]:
    with open(SWEEP_DIR / "component_ablation.yaml") as f:
        data = yaml.safe_load(f)
    return data["experiments"]


def generate_hyperparam_experiments() -> List[Tuple[str, Dict]]:
    """Generate OFAT hyperparameter experiments.

    Skips the "default" value run for each group since those are
    identical to the component ablation "Full_Model" baseline.
    """
    config = load_hyperparam_sweep()
    experiments = []

    defaults = {
        group_name: group_cfg["default"]
        for group_name, group_cfg in config.items()
    }

    for group_name, group_cfg in config.items():
        param_key = group_cfg["parameter"]
        values = group_cfg["values"]
        default_val = group_cfg["default"]

        for val in values:
            # Skip the default value — it's the same as COMP_Full_Model
            if val == default_val:
                continue

            overrides = dict(SHARED_OVERRIDES)
            for other_name, other_cfg in config.items():
                if other_name != group_name:
                    overrides[other_cfg["parameter"]] = defaults[other_name]
            overrides[param_key] = val

            exp_name = f"HP_{group_name}_{val}"
            experiments.append((exp_name, overrides))

    return experiments


def generate_component_experiments() -> List[Tuple[str, Dict]]:
    """Generate component ablation experiments."""
    component_cfgs = load_component_sweep()
    experiments = []

    for exp_cfg in component_cfgs:
        overrides = dict(SHARED_OVERRIDES)
        overrides.update(exp_cfg.get("overrides", {}))
        exp_name = f"COMP_{exp_cfg['name']}"
        experiments.append((exp_name, overrides))

    return experiments


# ──────────────────────────────────────────────────────────────────────
# COMMAND BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_train_command(exp_name: str, overrides: Dict) -> str:
    """Build a Hydra-compatible training command."""
    cmd_parts = [
        "python train.py",
        f'experiment_name="{exp_name}"',
        # Keep Hydra from changing cwd so checkpoints/logs land in project root
        'hydra.run.dir="."',
        'hydra.output_subdir=null',
    ]
    for key, val in overrides.items():
        if isinstance(val, bool):
            cmd_parts.append(f"{key}={'true' if val else 'false'}")
        elif isinstance(val, str):
            cmd_parts.append(f'{key}="{val}"')
        else:
            cmd_parts.append(f"{key}={val}")

    return " ".join(cmd_parts)


# ──────────────────────────────────────────────────────────────────────
# METRIC EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def _extract_metric_from_log(log_file: Path, metric_name: str) -> Optional[float]:
    """Best-effort extraction of a metric from the training log."""
    try:
        content = log_file.read_text()

        # For val_acc, extract from checkpoint filename which has the BEST value
        # e.g., "affect-diff-epoch=17-val_acc=0.718.ckpt"
        if metric_name == "val_acc":
            ckpt_matches = re.findall(r"val_acc=([0-9]+\.[0-9]+)\.ckpt", content)
            if ckpt_matches:
                return float(ckpt_matches[-1])

        # For test metrics: PyTorch Lightning prints "test_acc     0.7080"
        pattern = rf"{metric_name}\s+([0-9]+\.[0-9]+)"
        matches = re.findall(pattern, content)
        if matches:
            return float(matches[-1])  # take last occurrence
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# LAUNCHERS
# ──────────────────────────────────────────────────────────────────────

def launch_dry(experiments: List[Tuple[str, Dict]]) -> None:
    """Print commands without executing — for review."""
    completed = get_completed_names()
    remaining = [e for e in experiments if e[0] not in completed]
    total_mins = len(remaining) * ESTIMATED_MINUTES_PER_RUN

    print(f"\n{'='*70}")
    print(f"DRY RUN: {len(experiments)} experiments")
    print(f"Already completed: {len(experiments) - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Estimated time: ~{total_mins // 60}h {total_mins % 60}m")
    print(f"{'='*70}\n")

    for i, (name, overrides) in enumerate(experiments, 1):
        status = "✅ DONE" if name in completed else "⏳ TODO"
        cmd = build_train_command(name, overrides)
        print(f"[{i:3d}/{len(experiments)}] {status} {name}")
        print(f"    {cmd}\n")


def launch_run(experiments: List[Tuple[str, Dict]]) -> None:
    """Run experiments sequentially with crash recovery."""
    completed = get_completed_names()
    remaining = [(n, o) for n, o in experiments if n not in completed]

    total = len(experiments)
    done = total - len(remaining)
    todo = len(remaining)
    est_mins = todo * ESTIMATED_MINUTES_PER_RUN

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ABLATION SWEEP — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total: {total} | Done: {done} | Remaining: {todo}")
    print(f"  Estimated time: ~{est_mins // 60}h {est_mins % 60}m")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"{'='*70}\n")

    if todo == 0:
        print("All experiments already completed! Run analyze to see results.")
        print(f"  python scripts/analyze_ablation_results.py")
        return

    for i, (name, overrides) in enumerate(remaining, 1):
        cmd = build_train_command(name, overrides)
        log_file = LOGS_DIR / f"{name}.log"

        print(f"\n{'─'*70}")
        print(f"[{done + i}/{total}] {name}")
        print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Log:      {log_file}")
        print(f"  Remaining after this: {todo - i}")
        est_left = (todo - i) * ESTIMATED_MINUTES_PER_RUN
        print(f"  Est. time left after this: ~{est_left // 60}h {est_left % 60}m")
        print(f"{'─'*70}")

        # Run with stdout/stderr going to both console and log file
        tee_cmd = f"({cmd}) 2>&1 | tee {log_file}"
        result = subprocess.run(tee_cmd, shell=True, cwd=PROJECT_ROOT)

        # Extract test accuracy from log (best effort)
        test_acc = _extract_metric_from_log(log_file, "test_acc")
        val_acc = _extract_metric_from_log(log_file, "val_acc")

        mark_completed(name, {
            "exit_code": result.returncode,
            "test_acc": test_acc,
            "val_acc": val_acc,
        })

        if result.returncode == 0:
            print(f"\n  ✅ {name} finished | test_acc={test_acc or '?'} | val_acc={val_acc or '?'}")
        else:
            print(f"\n  ❌ {name} FAILED (exit code {result.returncode})")
            print(f"     Check log: {log_file}")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    all_completed = load_completed()
    successes = sum(1 for v in all_completed.values() if v.get("exit_code") == 0)
    failures = sum(1 for v in all_completed.values() if v.get("exit_code") != 0)
    print(f"  Succeeded: {successes} | Failed: {failures}")
    print(f"  Results: {MANIFEST_PATH}")
    print(f"\n  Next step:")
    print(f"    python scripts/analyze_ablation_results.py")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Affect-Diff Ablation Sweep (sequential, tmux-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview all experiments
  python scripts/run_ablation_sweep.py --mode all --launch dry

  # Run everything (resume-safe)
  python scripts/run_ablation_sweep.py --mode all --launch run

  # Run only component ablation
  python scripts/run_ablation_sweep.py --mode component --launch run

  # Reset and start fresh
  python scripts/run_ablation_sweep.py --reset --mode all --launch run
""",
    )
    parser.add_argument(
        "--mode",
        choices=["hyperparam", "component", "all"],
        default="all",
        help="Which ablation set to run",
    )
    parser.add_argument(
        "--launch",
        choices=["dry", "run"],
        default="dry",
        help="'dry' to preview, 'run' to execute sequentially",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the completed experiments manifest and start fresh",
    )
    args = parser.parse_args()

    if args.reset:
        # Clear manifest
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
            print("Cleared completed experiments manifest.")
        # Clean old ablation checkpoints
        ckpt_root = PROJECT_ROOT / "checkpoints"
        cleaned_ckpt = 0
        for exp_dir in ckpt_root.iterdir():
            if exp_dir.is_dir() and (exp_dir.name.startswith("COMP_") or exp_dir.name.startswith("HP_")):
                shutil.rmtree(exp_dir)
                cleaned_ckpt += 1
        if cleaned_ckpt:
            print(f"Removed {cleaned_ckpt} old ablation checkpoint directories.")
        # Clean old ablation logs
        cleaned_logs = 0
        if LOGS_DIR.exists():
            for log_file in LOGS_DIR.glob("*.log"):
                log_file.unlink()
                cleaned_logs += 1
        if cleaned_logs:
            print(f"Removed {cleaned_logs} old ablation log files.")
        print()

    # Component ablation goes first (Full_Model baseline is the reference)
    experiments = []
    if args.mode in ("component", "all"):
        comp_exps = generate_component_experiments()
        print(f"Component ablation:        {len(comp_exps)} experiments")
        experiments.extend(comp_exps)

    if args.mode in ("hyperparam", "all"):
        hp_exps = generate_hyperparam_experiments()
        print(f"Hyperparameter sensitivity: {len(hp_exps)} experiments (defaults deduplicated)")
        experiments.extend(hp_exps)

    print(f"Total: {len(experiments)} experiments")

    if args.launch == "dry":
        launch_dry(experiments)
    elif args.launch == "run":
        launch_run(experiments)


if __name__ == "__main__":
    main()
