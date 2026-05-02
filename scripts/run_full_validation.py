#!/usr/bin/env python
"""Full sim2sim validation protocol: N trials × 5 seeds × 3 conditions, with
Mann-Whitney U + Bonferroni pairwise tests across {recovery_gps, recovery_random,
no_recovery}. Wraps sim2sim_validation/run_validation.py and aggregates the
per-seed JSON summaries.

Usage:
  ./launch.sh scripts/run_full_validation.py \
      --checkpoint experiments/regolith_recovery/seed_0/.../agent_200000.pt \
      --num_trials 100 --seeds "0 1 2 3 4"
"""
import argparse
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
MODES = ["recovery_gps", "recovery_random", "no_recovery"]


def run_one(checkpoint: str, seed: int, num_trials: int, num_envs: int,
            goal_x: float, goal_y: float, out_dir: Path) -> dict:
    out_json = out_dir / f"seed_{seed}.json"
    if out_json.exists():
        print(f"[full_val] seed={seed} cached → {out_json}")
        return json.loads(out_json.read_text())

    cmd = [
        "./launch.sh", "sim2sim_validation/run_validation.py",
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--num_trials", str(num_trials),
        "--goal_x", str(goal_x),
        "--goal_y", str(goal_y),
        "--seed", str(seed),
        "--experiments", ",".join(MODES),
    ]
    print(f"[full_val] launching seed={seed}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, check=True)

    sim2sim_dir = REPO / "experiments" / "sim2sim"
    latest = max(sim2sim_dir.glob("summary_*.json"), key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text())
    out_json.write_text(json.dumps(data, indent=2))
    return data


def collect_metric(per_seed: list[dict], mode: str, key: str) -> np.ndarray:
    """Concatenate per-trial values across seeds for (mode, metric)."""
    vals = []
    for run in per_seed:
        trials = run["experiments"][mode].get("trials", [])
        for t in trials:
            v = t.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def pairwise_mwu(per_seed: list[dict], key: str) -> list[dict]:
    rows = []
    pairs = list(combinations(MODES, 2))
    bonf = len(pairs)
    for a, b in pairs:
        xa = collect_metric(per_seed, a, key)
        xb = collect_metric(per_seed, b, key)
        if len(xa) == 0 or len(xb) == 0:
            rows.append({"metric": key, "a": a, "b": b, "p": None,
                         "p_bonf": None, "n_a": int(len(xa)), "n_b": int(len(xb))})
            continue
        try:
            stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        rows.append({
            "metric": key, "a": a, "b": b,
            "U": float(stat), "p": float(p), "p_bonf": float(min(1.0, p * bonf)),
            "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
            "n_a": int(len(xa)), "n_b": int(len(xb)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="0 1 2 3 4")
    ap.add_argument("--num_trials", type=int, default=100)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--goal_x", type=float, default=3.0)
    ap.add_argument("--goal_y", type=float, default=0.0)
    ap.add_argument("--out_dir", default=str(REPO / "experiments" / "full_validation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split()]
    per_seed = [run_one(args.checkpoint, s, args.num_trials, args.num_envs,
                        args.goal_x, args.goal_y, out_dir) for s in seeds]

    metrics_keys = ["time_to_escape", "time_to_goal", "path_efficiency",
                    "escape_heading_error"]
    stats = {k: pairwise_mwu(per_seed, k) for k in metrics_keys}

    success_rate = {}
    for m in MODES:
        flags = []
        for run in per_seed:
            for t in run["experiments"][m].get("trials", []):
                flags.append(int(bool(t.get("reached_goal", False))))
        success_rate[m] = float(np.mean(flags)) if flags else 0.0

    report = {
        "args": vars(args), "seeds": seeds, "modes": MODES,
        "n_total_trials_per_mode": args.num_trials * len(seeds),
        "success_rate": success_rate,
        "mannwhitney_bonferroni": stats,
    }
    report_path = out_dir / "aggregate_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Aggregate ===")
    for m, sr in success_rate.items():
        print(f"  {m:18s}: success={sr:.3f}")
    print("\nPairwise (Bonferroni-corrected, α=0.05):")
    for k, rows in stats.items():
        print(f"  [{k}]")
        for r in rows:
            if r.get("p") is None:
                continue
            star = "*" if (r["p_bonf"] is not None and r["p_bonf"] < 0.05) else " "
            print(f"   {star} {r['a']:18s} vs {r['b']:18s}  "
                  f"p={r['p']:.4g}  p_bonf={r['p_bonf']:.4g}  "
                  f"med {r['median_a']:.2f} / {r['median_b']:.2f}")
    print(f"\n[full_val] report → {report_path}")


if __name__ == "__main__":
    main()
