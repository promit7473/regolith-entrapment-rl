#!/usr/bin/env python
"""Render the rliable IQM + stratified bootstrap CI figure (Agarwal et al.
NeurIPS 2021) over multi-seed full_validation runs.

Inputs: experiments/full_validation/seed_{0..4}.json (produced by
scripts/run_full_validation.py).

Outputs:
  paper/figures/rliable_iqm.pdf — IQM bar with 95% stratified bootstrap CI
                                  per condition, per metric.
  paper/figures/rliable_iqm.csv — same numbers in tabular form.

Falls back to a hand-rolled stratified bootstrap if the `rliable` package is
not installed (so the CI bars are still defensible).
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
METRICS = [
    ("path_efficiency", "Path efficiency", "higher_better"),
    ("time_to_escape", "Time to escape (s)", "lower_better"),
    ("time_to_goal", "Time to goal (s)", "lower_better"),
]


def iqm(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    lo, hi = np.percentile(x, [25, 75])
    mid = x[(x >= lo) & (x <= hi)]
    return float(np.mean(mid)) if len(mid) else float(np.median(x))


def stratified_bootstrap_ci(per_seed: list[np.ndarray], n_boot: int = 10_000,
                             alpha: float = 0.05, rng=None) -> tuple[float, float, float]:
    """Resample WITHIN each seed (stratified), pool, compute IQM. 95% CI."""
    rng = rng or np.random.default_rng(0)
    seeds = [s for s in per_seed if len(s) > 0]
    if not seeds:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pooled = np.concatenate([
            rng.choice(s, size=len(s), replace=True) for s in seeds
        ])
        boots[b] = iqm(pooled)
    point = iqm(np.concatenate(seeds))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def collect_per_seed(per_seed_runs, mode, key) -> list[np.ndarray]:
    out = []
    for run in per_seed_runs:
        trials = run["experiments"][mode].get("trials", [])
        vals = [float(t[key]) for t in trials
                if t.get(key) is not None and not np.isnan(float(t[key]))]
        out.append(np.asarray(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed_dir", default=str(REPO / "experiments" / "full_validation"))
    ap.add_argument("--out_dir", default=str(REPO / "paper" / "figures"))
    args = ap.parse_args()

    runs = []
    for p in sorted(Path(args.multiseed_dir).glob("seed_*.json")):
        runs.append(json.loads(p.read_text()))
    if not runs:
        print(f"[rliable] no seed_*.json found in {args.multiseed_dir}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.2))
    if len(METRICS) == 1:
        axes = [axes]
    csv_rows = ["metric,mode,iqm,ci_lo,ci_hi,n_seeds,n_total"]

    for ax, (key, label, _) in zip(axes, METRICS):
        xs = np.arange(len(MODES))
        points, los, his = [], [], []
        for mode in MODES:
            ps = collect_per_seed(runs, mode, key)
            point, lo, hi = stratified_bootstrap_ci(ps, n_boot=10_000, rng=rng)
            n_seeds = sum(1 for s in ps if len(s) > 0)
            n_total = sum(len(s) for s in ps)
            points.append(point); los.append(lo); his.append(hi)
            csv_rows.append(f"{key},{mode},{point:.4f},{lo:.4f},{hi:.4f},{n_seeds},{n_total}")
        err = np.array([
            [p - l for p, l in zip(points, los)],
            [h - p for p, h in zip(points, his)],
        ])
        ax.bar(xs, points, yerr=err, capsize=5,
               color=["#2c7fb8", "#7fcdbb", "#edf8b1"])
        ax.set_xticks(xs)
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("IQM with 95% stratified bootstrap CI (Agarwal et al. 2021)")
    fig.tight_layout()
    pdf = out_dir / "rliable_iqm.pdf"
    fig.savefig(pdf)
    csv = out_dir / "rliable_iqm.csv"
    csv.write_text("\n".join(csv_rows))
    print(f"[rliable] {pdf}\n[rliable] {csv}")


if __name__ == "__main__":
    main()
