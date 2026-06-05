#!/usr/bin/env bash
# Scaled in-engine (Newton) escape comparison: all controllers, in-trained-range
# sinkage levels, 30 trials/level. Sequential (single GPU). Writes per-controller
# JSON/CSV under experiments/escape_eval/sweep/.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=experiments/escape_eval/sweep
mkdir -p "$OUT"
LEVELS="0.15,0.20,0.25,0.28"
NENV=16
EPS=30
S1=experiments/regolith_recovery/seed_1/seed_1/checkpoints/best_agent.pt
S3=experiments/regolith_recovery/seed_3/seed_3/checkpoints/best_agent.pt

run() {  # $1=tag  rest=args
  local tag="$1"; shift
  echo "=========== $(date +%H:%M:%S) START $tag ==========="
  PYTHONUNBUFFERED=1 ./launch.sh scripts/escape_eval.py --headless \
     --num_envs $NENV --episodes_per_level $EPS --sinkage_levels $LEVELS --seed 0 \
     --out_json "$OUT/${tag}.json" --out_csv "$OUT/${tag}.csv" "$@" \
     > "/tmp/sweep_${tag}.log" 2>&1
  echo "=========== $(date +%H:%M:%S) END $tag (exit $?) ==========="
}

run constant_drive --control constant_drive
run rocking        --control rocking --half_period 2.0
run policy_seed1   --control policy --checkpoint "$S1"
run policy_seed3   --control policy --checkpoint "$S3"
echo "ALL_SWEEP_RUNS_DONE"
