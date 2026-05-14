#!/usr/bin/env bash
# One-shot orchestrator: runs the full post-training validation pipeline.
#
# RECOMMENDED: run the stages step-by-step (see RUNBOOK.md). Each stage takes
# hours and a mid-pipeline failure wastes upstream compute. This script exists
# for unattended overnight runs and as authoritative documentation of the
# argument flow between stages.
#
# Usage:
#   bash scripts/run_all_validation.sh \
#       --checkpoint experiments/regolith_recovery/seed_0/.../agent_200000.pt \
#       --seeds "0 1 2 3 4" --num_trials 100
#   bash scripts/run_all_validation.sh ... --dry-run    # print, do not execute
#   bash scripts/run_all_validation.sh ... --skip-crm   # skip CRM tier (no FSI build)
set -euo pipefail

CKPT=""
SEEDS="0 1 2 3 4"
NTRIALS=100
DRY=0
SKIP_CRM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --seeds)      SEEDS="$2"; shift 2;;
    --num_trials) NTRIALS="$2"; shift 2;;
    --dry-run)    DRY=1; shift;;
    --skip-crm)   SKIP_CRM=1; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "ERROR: --checkpoint required"; exit 2
fi

run() {
  echo "+ $*"
  if [[ $DRY -eq 0 ]]; then eval "$@"; fi
}

cd "$(dirname "$0")/.."
ONNX="sim2real/onnx_export/output/recovery_policy.onnx"

echo "=== [1/6] ONNX export ==="
run "./launch.sh sim2real/onnx_export/export_model.py --policy_ckpt '$CKPT' --out_dir '$(dirname $ONNX)'"

echo "=== [2/6] Full validation grid (5 seeds × ${NTRIALS} × 3 conds) ==="
run "./launch.sh scripts/run_full_validation.py --checkpoint '$CKPT' --num_trials $NTRIALS --seeds '$SEEDS'"

echo "=== [3/6] OOD sinkage × friction sweep ==="
run "./launch.sh scripts/run_ood_sweep.py --checkpoint '$CKPT'"

echo "=== [4/6] Cross-engine: Chrono ChSystemNSC + Bekker-Wong terrain ==="
# Uses conda env chrono_viz (pychrono 10.0.0 core+robot; Bekker-Wong via custom ChLoad)
run "conda run -n chrono_viz python cross_engine/chrono_validation.py --onnx '$ONNX' --num_trials 50 --seeds 0 1 2"

echo "=== [5/6] (CRM/SPH tier requires pychrono built with FSI — skipped) ==="

echo "=== [6/6] Aggregate figures ==="
run "python3 scripts/make_rliable_figure.py"
run "python3 scripts/classify_failures.py"

echo
echo "DONE. Artifacts:"
echo "  experiments/full_validation/seed_*.json"
echo "  cross_engine/results/chrono_{scm,crm}_summary.json"
echo "  paper/figures/rliable_iqm.pdf"
echo "  paper/figures/failure_modes.pdf"
