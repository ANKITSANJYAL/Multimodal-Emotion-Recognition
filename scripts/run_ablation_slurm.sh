#!/bin/bash
# ============================================================
# Affect-Diff: Master SLURM Launcher for Ablation Studies
# CVPR 2026
#
# Usage:
#   # Dry run (see all commands)
#   bash scripts/run_ablation_slurm.sh dry
#
#   # Submit all ablation jobs
#   bash scripts/run_ablation_slurm.sh submit
#
#   # Check job status
#   bash scripts/run_ablation_slurm.sh status
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry}"
LOGS_DIR="logs/ablation"
mkdir -p "$LOGS_DIR"

echo "============================================"
echo "  Affect-Diff Ablation Sweep"
echo "  Date: $(date)"
echo "  Mode: $MODE"
echo "============================================"

if [ "$MODE" = "dry" ]; then
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch dry
elif [ "$MODE" = "submit" ]; then
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch slurm
elif [ "$MODE" = "sequential" ]; then
    conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch sequential
elif [ "$MODE" = "status" ]; then
    echo ""
    echo "Running/Pending ablation jobs:"
    squeue -u "$USER" --format="%.10i %.25j %.8T %.10M %.6D %R" | grep -E "HP_|COMP_" || echo "  (none found)"
    echo ""
    echo "Completed ablation logs:"
    ls -lt "$LOGS_DIR"/*.out 2>/dev/null | head -20 || echo "  (no logs yet)"
elif [ "$MODE" = "analyze" ]; then
    conda run -n emotion_rec python scripts/analyze_ablation_results.py --wandb-project "Affect-Diff-CVPR"
else
    echo "Usage: $0 {dry|submit|sequential|status|analyze}"
    exit 1
fi
