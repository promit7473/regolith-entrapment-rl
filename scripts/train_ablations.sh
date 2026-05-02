#!/usr/bin/env bash
# Ablation training driver. Each ablation flag is read by scripts/train.py /
# envs/entrapment_env.py via env-vars; one seed per ablation is enough for the
# paper's ablation table since the multi-seed bar comes from the main runs.
#
# Ablations:
#   no_priv_critic — symmetric critic (29D, drops oracle features)
#   no_dr          — fixed sand friction & no obs noise
#   no_pen_grind   — drops slip-conditional grinding penalty
#   no_pen_hop     — drops vertical-bounce penalty
#
# Usage: bash scripts/train_ablations.sh [--timesteps N] [--num_envs M]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TIMESTEPS=200000
NUM_ENVS=64
SEED=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --timesteps) TIMESTEPS="$2"; shift 2;;
    --num_envs)  NUM_ENVS="$2";  shift 2;;
    --seed)      SEED="$2";      shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

ABLATIONS=(no_priv_critic no_dr no_pen_grind no_pen_hop)
LOG_DIR="experiments/ablations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

for AB in "${ABLATIONS[@]}"; do
  echo "════════════════════════════════════════════════════════════"
  echo "[ablation] $AB  (seed=$SEED, timesteps=$TIMESTEPS)"
  echo "════════════════════════════════════════════════════════════"
  ABLATION="$AB" ./launch.sh scripts/train.py \
      --seed "$SEED" --num_envs "$NUM_ENVS" --timesteps "$TIMESTEPS" \
    2>&1 | tee "$LOG_DIR/${AB}.log"
done
echo "[ablation] All ablations complete. Logs → $LOG_DIR"
