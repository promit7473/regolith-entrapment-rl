"""Publication-quality failure-mode analysis plots.

Reads experiments/regolith_recovery/failure_modes.csv (written by entrapment_env.py)
and generates four figures:

  1. failure_modes_escape_vs_sinkage.png   — bar chart: escape rate by sinkage bin
  2. failure_modes_escape_vs_curriculum.png — scatter + trend: escape rate vs curriculum level
  3. failure_modes_ep_length_dist.png      — histogram: episode length for escaped vs failed
  4. failure_modes_heatmap.png             — 2-D heatmap: escape rate over (sinkage × curriculum)

Run:
    python scripts/plot_failure_modes.py
    python scripts/plot_failure_modes.py --csv path/to/failure_modes.csv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Style (matches scripts/plot_training.py) ──────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#444444",
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.color":           "#DDDDDD",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.6,
    "grid.alpha":           1.0,
    "font.family":          "DejaVu Sans",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "axes.labelcolor":      "#222222",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.color":          "#444444",
    "ytick.color":          "#444444",
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "#BBBBBB",
    "lines.linewidth":      2.0,
    "savefig.dpi":          200,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
})

COLOR_ESC  = "#2196F3"   # blue — escaped
COLOR_FAIL = "#F44336"   # red  — failed/timed out
COLOR_HEAT = "RdYlGn"    # diverging green=good, red=bad


def _remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: escape rate vs sinkage (bar chart) ─────────────────────────────────

def plot_escape_vs_sinkage(df: pd.DataFrame, out_dir: str):
    bins   = np.arange(0.10, 0.35, 0.05)   # [0.10, 0.15, 0.20, 0.25, 0.30]
    labels = [f"{b:.2f}–{b+0.05:.2f}" for b in bins]
    df2    = df.copy()
    df2["sinkage_bin"] = pd.cut(df2["sinkage"], bins=bins, labels=labels, right=False)

    grouped   = df2.groupby("sinkage_bin", observed=False)["escaped"]
    rates     = grouped.mean()
    counts    = grouped.count()
    valid     = counts > 0

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(valid.sum())
    bars = ax.bar(xs, rates[valid].values, color=COLOR_ESC, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(rates[valid].index.tolist(), rotation=30, ha="right")
    ax.set_xlabel("Sinkage Depth (m)")
    ax.set_ylabel("Escape Rate")
    ax.set_title("Escape Rate vs Sinkage Depth")
    ax.set_ylim(0.0, 1.05)
    # Annotate bars with count
    for bar, n in zip(bars, counts[valid].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="#444444")
    _remove_spines(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_modes_escape_vs_sinkage.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: escape rate vs curriculum level (scatter + trend) ──────────────────

def plot_escape_vs_curriculum(df: pd.DataFrame, out_dir: str):
    bins   = np.linspace(0.0, 1.0, 11)   # 10 curriculum bins
    df2    = df.copy()
    df2["curr_bin"] = pd.cut(df2["curriculum_level"], bins=bins, right=False)
    grouped = df2.groupby("curr_bin", observed=False)["escaped"]
    rates   = grouped.mean().dropna()
    mids    = [iv.mid for iv in rates.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mids, rates.values, color=COLOR_ESC, s=60, zorder=3, label="Bin mean")
    # Trend line (linear fit)
    if len(mids) >= 2:
        z  = np.polyfit(mids, rates.values, 1)
        px = np.linspace(min(mids), max(mids), 100)
        ax.plot(px, np.polyval(z, px), color="#FF9800", linewidth=1.5,
                linestyle="--", label=f"Trend (slope={z[0]:+.3f})")
    ax.set_xlabel("Curriculum Level")
    ax.set_ylabel("Escape Rate")
    ax.set_title("Escape Rate vs Curriculum Progress")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    _remove_spines(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_modes_escape_vs_curriculum.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: episode length distribution (histogram) ────────────────────────────

def plot_ep_length_dist(df: pd.DataFrame, out_dir: str):
    escaped = df[df["escaped"] == 1]["episode_steps"]
    failed  = df[df["escaped"] == 0]["episode_steps"]
    max_steps = int(df["episode_steps"].max()) + 1
    bins = np.linspace(0, max_steps, min(50, max_steps // 10 + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(failed.values,  bins=bins, alpha=0.7, color=COLOR_FAIL, label="Failed/Timeout", density=True)
    ax.hist(escaped.values, bins=bins, alpha=0.7, color=COLOR_ESC,  label="Escaped",        density=True)
    ax.set_xlabel("Episode Steps")
    ax.set_ylabel("Density")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    _remove_spines(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_modes_ep_length_dist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: 2-D heatmap (sinkage × curriculum_level) ──────────────────────────

def plot_heatmap(df: pd.DataFrame, out_dir: str):
    sinkage_bins = np.arange(0.10, 0.35, 0.05)
    curr_bins    = np.linspace(0.0, 1.0, 6)   # 5 curriculum bins
    s_labels     = [f"{b:.2f}" for b in sinkage_bins]
    c_labels     = [f"{b:.1f}" for b in curr_bins[:-1]]

    df2 = df.copy()
    df2["s_bin"] = pd.cut(df2["sinkage"], bins=sinkage_bins,
                          labels=s_labels, right=False)
    df2["c_bin"] = pd.cut(df2["curriculum_level"], bins=curr_bins,
                          labels=c_labels, right=False)

    pivot = df2.pivot_table(
        values="escaped", index="c_bin", columns="s_bin",
        aggfunc="mean", observed=False,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        pivot.values.astype(float),
        aspect="auto", origin="lower",
        cmap=COLOR_HEAT, vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Escape Rate")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Sinkage Depth (m)")
    ax.set_ylabel("Curriculum Level")
    ax.set_title("Escape Rate: Sinkage × Curriculum Level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotate cells
    for i in range(pivot.values.shape[0]):
        for j in range(pivot.values.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.8 else "white")
    fig.tight_layout()
    path = os.path.join(out_dir, "failure_modes_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "experiments", "regolith_recovery", "failure_modes.csv"),
        help="Path to failure_modes.csv",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(REPO_ROOT, "experiments", "regolith_recovery", "plots"),
        help="Output directory for PNG files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[plot_failure_modes] ERROR: CSV not found: {args.csv}")
        print("  Run training first to generate failure_modes.csv")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"[plot_failure_modes] Loaded {len(df)} episodes from {args.csv}")

    if len(df) == 0:
        print("[plot_failure_modes] No data — CSV is empty.")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)

    print("[plot_failure_modes] Generating plots...")
    plot_escape_vs_sinkage(df, args.out_dir)
    plot_escape_vs_curriculum(df, args.out_dir)
    plot_ep_length_dist(df, args.out_dir)
    plot_heatmap(df, args.out_dir)

    print(f"\n[plot_failure_modes] All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
