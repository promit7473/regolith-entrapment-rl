#!/usr/bin/env bash
# Tuned-scripted-baseline search: give the non-learning side its best shot.
#
# Sweeps the compatible-actuation scripted family (rocking, steer_paddle,
# rock_paddle) over cycle period and throttle, at the trapped depths. The best
# cell becomes the "tuned scripted" row of the paper's comparison table — the
# fairest non-learning bar a drive+steer-only platform admits. Also answers
# the reviewer question "RL, or just parameter search?" with data.
#
# Usage (gate-level, ~3-4 h):
#   bash scripts/tune_scripted_baseline.sh
# Paper-level: raise EPS to 50 after the winner is identified.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EPS="${EPS:-4}"
ENVS="${ENVS:-4}"
LEVELS="${LEVELS:-0.15,0.20}"
MU="${MU:-0.75}"
OUT="${OUT:-experiments/scripted_tuning}"
mkdir -p "$OUT"

echo "control,half_period,drive_mag,overall_escape_rate" | tee "$OUT/summary.csv"
for ctrl in rocking steer_paddle rock_paddle; do
  for hp in 1.0 2.0 3.0; do
    for dm in 0.5 1.0; do
      tag="${ctrl}_hp${hp}_dm${dm}"
      PYTHONUNBUFFERED=1 timeout 2400 ./launch.sh scripts/escape_eval.py --headless \
        --control "$ctrl" --num_envs "$ENVS" --episodes_per_level "$EPS" \
        --sinkage_levels "$LEVELS" --friction_override "$MU" \
        --half_period "$hp" --drive_mag "$dm" \
        --out_json "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
      rate=$(python3 -c "import json;print(json.load(open('$OUT/$tag.json'))['overall_escape_rate'])" 2>/dev/null || echo "FAIL")
      echo "$ctrl,$hp,$dm,$rate" | tee -a "$OUT/summary.csv"
    done
  done
done
echo "Done. Best cell:"
sort -t, -k4 -rn "$OUT/summary.csv" | head -2
