#!/usr/bin/env python
"""Classify per-trial failure modes from full_validation seed JSONs and render
a stacked-bar figure for the paper.

Taxonomy:
  goal_reached       — reached_goal=True (success)
  high_centered      — entrapped, no escape, |Δpos|<0.3m for trial duration
  circular_spin      — escape triggered but heading_error>120° at end
  inverted           — terminated early via env termination (chassis flip / OOB)
  stall_in_bed       — escaped but never reached goal
  timeout_no_progress — neither escaped nor terminated early; hit max_steps with low Δpos
  timeout_other       — fallthrough

Inputs : experiments/full_validation/seed_*.json
Outputs: paper/figures/failure_modes.pdf
         paper/figures/failure_modes.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
MODES = ["recovery_gps", "recovery_random", "no_recovery"]
LABELS = ["goal_reached", "high_centered", "circular_spin", "inverted",
          "stall_in_bed", "timeout_no_progress", "timeout_other"]
COLORS = ["#1a9850", "#fdae61", "#f46d43", "#d73027",
          "#fee090", "#abd9e9", "#bababa"]


def classify(trial: dict) -> str:
    if trial.get("reached_goal"):
        return "goal_reached"
    if trial.get("terminated_early"):
        return "inverted"
    escaped = trial.get("escaped", False)
    hdg_err = trial.get("escape_heading_error")
    delta_pos = trial.get("delta_pos")
    if not escaped:
        if delta_pos is not None and delta_pos < 0.3:
            return "high_centered"
        return "timeout_no_progress"
    if hdg_err is not None and abs(hdg_err) > 120.0:
        return "circular_spin"
    return "stall_in_bed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed_dir", default=str(REPO / "experiments" / "full_validation"))
    ap.add_argument("--out_dir", default=str(REPO / "paper" / "figures"))
    args = ap.parse_args()

    runs = [json.loads(p.read_text())
            for p in sorted(Path(args.multiseed_dir).glob("seed_*.json"))]
    if not runs:
        print(f"[failures] no seed_*.json under {args.multiseed_dir}")
        return

    counts = {m: {l: 0 for l in LABELS} for m in MODES}
    for run in runs:
        for mode in MODES:
            for t in run["experiments"][mode].get("trials", []):
                counts[mode][classify(t)] += 1

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["mode," + ",".join(LABELS)]
    for m in MODES:
        rows.append(m + "," + ",".join(str(counts[m][l]) for l in LABELS))
    (out_dir / "failure_modes.csv").write_text("\n".join(rows))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(len(MODES))
    bottoms = np.zeros(len(MODES))
    totals = np.array([sum(counts[m][l] for l in LABELS) for m in MODES], dtype=float)
    totals[totals == 0] = 1.0
    for label, color in zip(LABELS, COLORS):
        vals = np.array([counts[m][label] for m in MODES]) / totals
        ax.bar(xs, vals, bottom=bottoms, label=label, color=color)
        bottoms += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Trial fraction")
    ax.set_title("Failure-mode distribution by condition")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "failure_modes.pdf")
    print(f"[failures] {out_dir / 'failure_modes.pdf'}")
    print(f"[failures] {out_dir / 'failure_modes.csv'}")


if __name__ == "__main__":
    main()
