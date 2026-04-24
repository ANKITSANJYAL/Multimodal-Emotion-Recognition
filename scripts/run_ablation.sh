#!/bin/bash
# ============================================================
# Affect-Diff: Ablation Study Launcher
# CVPR 2026
#
# Run inside a tmux session on a GPU node:
#   tmux new -s ablation
#   bash scripts/run_ablation.sh run
#
# Detach: Ctrl+b d
# Reattach: tmux attach -t ablation
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry}"

echo "============================================"
echo "  Affect-Diff Ablation Study"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  Mode: $MODE"
echo "============================================"

case "$MODE" in
    dry)
        conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch dry
        ;;
    run)
        conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode all --launch run
        ;;
    component)
        conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode component --launch run
        ;;
    hyperparam)
        conda run -n emotion_rec python scripts/run_ablation_sweep.py --mode hyperparam --launch run
        ;;
    reset)
        conda run -n emotion_rec python scripts/run_ablation_sweep.py --reset --mode all --launch dry
        ;;
    analyze)
        conda run -n emotion_rec python scripts/analyze_ablation_results.py
        ;;
    status)
        echo ""
        if [ -f "outputs/ablation_results/completed.json" ]; then
            echo "Completed experiments:"
            conda run -n emotion_rec python -c '
import json
data = json.load(open("outputs/ablation_results/completed.json"))
ok = sum(1 for v in data.values() if v.get("exit_code") == 0)
fail = sum(1 for v in data.values() if v.get("exit_code") != 0)
print("  Succeeded: %d | Failed: %d" % (ok, fail))
print()
for name, info in sorted(data.items()):
    code = info.get("exit_code", -1)
    acc = info.get("test_acc", "?")
    mark = "OK" if code == 0 else "FAIL"
    print("  [%-4s] %-40s  test_acc=%s" % (mark, name, acc))
'
        else
            echo "  No experiments completed yet."
        fi
        ;;
    *)
        echo "Usage: $0 {dry|run|component|hyperparam|reset|analyze|status}"
        echo ""
        echo "  dry        Preview all experiment commands"
        echo "  run        Run all experiments sequentially (resume-safe)"
        echo "  component  Run only component ablation (Table 1)"
        echo "  hyperparam Run only hyperparameter sensitivity (Table 2)"
        echo "  reset      Clear progress and preview fresh"
        echo "  analyze    Generate paper tables and plots from results"
        echo "  status     Show which experiments are done"
        exit 1
        ;;
esac
