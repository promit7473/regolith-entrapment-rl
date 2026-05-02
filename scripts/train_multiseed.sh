#!/usr/bin/env bash
# Multi-seed training driver. Runs scripts/train.py for each seed serially
# (Newton MPM saturates a single GPU at num_envs=64, so parallelism between
# seeds is not feasible on a 5070Ti — run them back-to-back instead).
#
# Each seed gets its own experiment subdirectory under
# experiments/regolith_recovery/seed_{N}/ so the rliable bootstrap pipeline
# can stratify across seeds without colliding W&B run names.
#
# Usage:
#   bash scripts/train_multiseed.sh                 # 5 seeds × 200k steps
#   bash scripts/train_multiseed.sh --timesteps 4000000 --num_envs 256
#   bash scripts/train_multiseed.sh --seeds "0 1 2"
#
# Estimated wall-clock on RTX 5070Ti at num_envs=64, timesteps=200000:
#   ~4.5 h per seed × 5 seeds = ~23 h.
# On RTX 4090 at num_envs=256, timesteps=4_000_000: ~8 h per seed.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SEEDS="0 1 2 3 4"
TIMESTEPS=200000
NUM_ENVS=64

while [[ $# -gt 0 ]]; do
  case $1 in
    --seeds)     SEEDS="$2"; shift 2;;
    --timesteps) TIMESTEPS="$2"; shift 2;;
    --num_envs)  NUM_ENVS="$2"; shift 2;;
    --help)
      grep '^# ' "$0" | sed 's/^# \?//'
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

LOG_DIR="experiments/multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[multiseed] Seeds       : $SEEDS"
echo "[multiseed] Timesteps   : $TIMESTEPS"
echo "[multiseed] Num envs    : $NUM_ENVS"
echo "[multiseed] Log dir     : $LOG_DIR"

for SEED in $SEEDS; do
  echo "════════════════════════════════════════════════════════════"
  echo "[multiseed] Starting seed=$SEED at $(date -Is)"
  echo "════════════════════════════════════════════════════════════"
  ./launch.sh scripts/train.py \
      --seed "$SEED" \
      --num_envs "$NUM_ENVS" \
      --timesteps "$TIMESTEPS" \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
  echo "[multiseed] Finished seed=$SEED at $(date -Is)"
done

echo "[multiseed] All seeds complete."
echo "[multiseed] Per-seed checkpoints under experiments/regolith_recovery/"
echo "[multiseed] To stratify rliable plots, run:"
echo "    ./launch.sh scripts/make_rliable_figure.py --multiseed_dir experiments/regolith_recovery/"
