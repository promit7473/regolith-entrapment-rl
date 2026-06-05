"""Generate the in-engine escape-comparison figures for the paper from
escape_eval.py outputs (policy vs naive baselines, Newton/MPM).

Reads per-controller JSON (overall + per_level escape + action_trace) and writes:
  figures/escape_comparison.png   — overall escape per controller, bootstrap CIs
  figures/escape_vs_sinkage.png   — escape rate vs sinkage depth, per controller
  figures/policy_action_behavior.png — drive/steer saturation + rocking (policies)

Usage:
  python3 scripts/plot_escape_comparison.py --in experiments/escape_eval/sweep \
      --out paper/figures
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Controller display order + styling (naive baselines → learned policies)
CONTROLLERS = [
    ("rocking",        "Rocking\n(scripted)",      "#b0b0b0"),
    ("constant_drive", "Constant drive\n(floor)",  "#e08214"),
    ("policy_seed1",   "Policy seed 1",            "#2166ac"),
    ("policy_seed3",   "Policy seed 3",            "#4393c3"),
]


def _load(in_dir, tag):
    p = os.path.join(in_dir, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig_overall(data, out):
    labels, rates, los, his, colors = [], [], [], [], []
    for tag, lab, col in CONTROLLERS:
        d = data.get(tag)
        if d is None:
            continue
        labels.append(lab); colors.append(col)
        rates.append(100 * d["overall_escape_rate"])
        lo, hi = d["overall_ci"]
        los.append(100 * (d["overall_escape_rate"] - lo))
        his.append(100 * (hi - d["overall_escape_rate"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.bar(x, rates, yerr=[los, his], capsize=5, color=colors,
                  edgecolor="black", linewidth=0.6)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 2.5, f"{r:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Escape rate (%)")
    ax.set_ylim(0, 109)
    ax.set_title("In-engine (Newton/MPM) escape rate: policy vs naive baselines")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_vs_sinkage(data, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for tag, lab, col in CONTROLLERS:
        d = data.get(tag)
        if d is None:
            continue
        levels = sorted(d["per_level"].keys(), key=float)
        xs = [float(k) * 100 for k in levels]
        ys = [100 * d["per_level"][k]["escape_rate"] for k in levels]
        lo = [100 * (d["per_level"][k]["escape_rate"] - d["per_level"][k]["ci"][0]) for k in levels]
        hi = [100 * (d["per_level"][k]["ci"][1] - d["per_level"][k]["escape_rate"]) for k in levels]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                    label=lab.replace("\n", " "), color=col, linewidth=1.8)
    ax.set_xlabel("Initial sinkage depth (cm)")
    ax.set_ylabel("Escape rate (%)")
    ax.set_ylim(-3, 109)
    ax.set_title("Escape rate vs entrapment severity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_behavior(data, out):
    """Throttle saturation, steer saturation, and rocking index per policy."""
    rows = []
    for tag in ("policy_seed1", "policy_seed3"):
        d = data.get(tag)
        if d is None or "action_trace" not in d:
            continue
        A = np.array([t["action"] for t in d["action_trace"]])
        drive = np.clip(A[:, :6], -1, 1); steer = np.clip(A[:, 6:10], -1, 1)
        md = drive.mean(1)
        rock = int((np.diff(np.sign(md)) != 0).sum()) / max(1, len(md))
        rows.append((tag.replace("policy_", "Policy "),
                     (drive > 0.95).mean(), (drive < -0.05).mean(),
                     (np.abs(steer) > 0.95).mean(), rock))
    if not rows:
        print("no policy action traces; skipping behavior fig"); return
    metrics = ["Drive\nfull-fwd", "Drive\nreverse", "Steer\nfull-lock", "Rocking\n(sign-flip)"]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (lab, *vals) in enumerate(rows):
        ax.bar(x + (i - 0.5) * w, [100 * v for v in vals], w, label=lab,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Fraction of steps (%)")
    ax.set_title("Learned recovery is steering-dominated, not rocking")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = {tag: _load(args.in_dir, tag) for tag, _, _ in CONTROLLERS}
    present = [k for k, v in data.items() if v]
    print("loaded:", present)
    fig_overall(data, os.path.join(args.out_dir, "escape_comparison.png"))
    fig_vs_sinkage(data, os.path.join(args.out_dir, "escape_vs_sinkage.png"))
    fig_behavior(data, os.path.join(args.out_dir, "policy_action_behavior.png"))


if __name__ == "__main__":
    main()
