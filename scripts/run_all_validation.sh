#!/usr/bin/env bash
# Cross-engine validation pipeline: ONNX export → Chrono granular terrain.
#
# Stages:
#   1. ONNX export (training checkpoint → onnx)
#   2. Chrono validation — granular (Bekker-Wong) terrain
#   3. Chrono validation — rigid baseline (NSC low-friction)
#
# Usage:
#   bash scripts/run_all_validation.sh \
#       --checkpoint experiments/regolith_recovery/seed_0/.../agent_200000.pt \
#       --seeds "0 1 2" --num_trials 50
#   bash scripts/run_all_validation.sh ... --dry-run
set -euo pipefail

CKPT=""
SEEDS="0 1 2"
NTRIALS=50
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --seeds)      SEEDS="$2"; shift 2;;
    --num_trials) NTRIALS="$2"; shift 2;;
    --dry-run)    DRY=1; shift;;
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

echo "=== [1/3] ONNX export ==="
run "./launch.sh sim2real/onnx_export/export_model.py --policy_ckpt '$CKPT' --out_dir '$(dirname $ONNX)'"

echo "=== [2/3] Chrono — granular (Bekker-Wong) terrain ==="
run "conda run -n chrono_viz python cross_engine_validation/chrono_validation.py \
  --onnx '$ONNX' \
  --terrain granular \
  --num_trials $NTRIALS --seeds $SEEDS"

echo "=== [3/3] Chrono — rigid baseline (NSC low-friction) ==="
run "conda run -n chrono_viz python cross_engine_validation/chrono_validation.py \
  --onnx '$ONNX' \
  --terrain rigid \
  --num_trials $NTRIALS --seeds $SEEDS"

echo "=== [4/4] Chrono — constant drive baseline (no policy) ==="
run "conda run -n chrono_viz python cross_engine_validation/chrono_validation.py \
  --control constant_drive \
  --terrain granular \
  --num_trials $NTRIALS --seeds $SEEDS"

echo
echo "DONE. Artifacts:"
echo "  cross_engine_validation/results/chrono_scm_policy_*          (granular, learned policy)"
echo "  cross_engine_validation/results/chrono_scm_constant_drive_*  (granular, open-loop naive)"
echo "  cross_engine_validation/results/chrono_nsc_policy_*          (rigid baseline)"
