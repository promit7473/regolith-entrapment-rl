"""
Publication-quality training convergence plots.
Style inspired by Bi & Ding (2026), Computers & Electronics in Agriculture.

Usage (no Isaac Sim needed — pure Python):
    python3 scripts/plot_training.py
    python3 scripts/plot_training.py --exp ppo_regolith
    python3 scripts/plot_training.py --exp ppo_regolith run_b --compare
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# ── Publication style ─────────────────────────────────────────────────────────
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

# Method color palette (matches paper style — distinct, print-safe)
PALETTE = {
    "ppo_regolith":  "#2166AC",   # deep blue
    "default_0":     "#2166AC",
    "default_1":     "#D6604D",
    "default_2":     "#4DAC26",
    "default_3":     "#8073AC",
    "default_4":     "#E08214",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_BASE  = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
PLOTS_DIR = os.path.join(EXP_BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_color(name, idx):
    return PALETTE.get(name, PALETTE.get(f"default_{idx % 5}", "#2166AC"))


def smooth(y, w=20):
    if len(y) < w:
        return y, y
    k   = np.ones(w) / w
    s   = np.convolve(y, k, mode="valid")
    return s, y[len(y) - len(s):]   # smoothed, aligned raw


def load_scalars(path):
    ea = event_accumulator.EventAccumulator(
        path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev    = ea.Scalars(tag)
        steps = np.array([e.step  for e in ev], dtype=np.float32)
        vals  = np.array([e.value for e in ev], dtype=np.float32)
        # Sort by step and deduplicate (smoke-test reruns can append lower steps)
        order = np.argsort(steps, kind="stable")
        steps, vals = steps[order], vals[order]
        # Keep only the last occurrence of each step value
        _, unique_idx = np.unique(steps[::-1], return_index=True)
        keep = len(steps) - 1 - unique_idx
        keep.sort()
        out[tag] = {"step": steps[keep], "value": vals[keep]}
    return out


def shade_band(ax, steps, values_raw, color, alpha=0.12):
    """Light confidence band using rolling std of raw data."""
    w = max(5, len(values_raw) // 30)
    if len(values_raw) < w * 2:
        return
    k   = np.ones(w) / w
    mu  = np.convolve(values_raw, k, mode="valid")
    sq  = np.convolve(values_raw ** 2, k, mode="valid")
    sig = np.sqrt(np.maximum(sq - mu ** 2, 0))
    s   = steps[len(steps) - len(mu):]
    ax.fill_between(s, mu - sig, mu + sig, color=color, alpha=alpha, linewidth=0)


# ── Figure 1 — Reward Convergence (paper Fig. 20 style) ───────────────────────

def plot_reward_convergence(exp_names, all_data, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.set_title("Reward Convergence", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Episode Reward")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))

    tag = "Reward / Total reward (mean)"
    plotted = 0
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        if tag not in data or len(data[tag]["step"]) == 0:
            continue
        color  = get_color(name, i)
        steps  = data[tag]["step"]
        values = data[tag]["value"]

        shade_band(ax, steps, values, color, alpha=0.15)
        sm, sm_steps = smooth(values, w=15)
        ax.plot(sm_steps, sm, color=color, linewidth=2.2,
                label=name, zorder=3)
        # Thin raw line
        ax.plot(steps, values, color=color, linewidth=0.6, alpha=0.3, zorder=2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure 2 — Loss Curves ─────────────────────────────────────────────────────

def plot_losses(exp_names, all_data, out_path):
    tags   = ["Loss / Policy loss", "Loss / Value loss", "Loss / Entropy loss"]
    labels = ["Policy Loss", "Value Loss", "Entropy Loss"]
    colors = ["#2166AC", "#D6604D", "#4DAC26"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Training Losses", fontweight="bold", fontsize=12, y=1.01)

    for ax, tag, label, color in zip(axes, tags, labels, colors):
        ax.set_title(label, pad=6)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
        ax.spines[["top", "right"]].set_visible(False)

        for i, (name, data) in enumerate(zip(exp_names, all_data)):
            if tag not in data or len(data[tag]["step"]) == 0:
                continue
            c      = get_color(name, i) if len(exp_names) > 1 else color
            steps  = data[tag]["step"]
            values = data[tag]["value"]
            sm, sm_steps = smooth(values, w=10)
            ax.plot(steps, values, color=c, linewidth=0.5, alpha=0.3)
            ax.plot(sm_steps, sm, color=c, linewidth=2.0,
                    label=name)

        if len(exp_names) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure 3 — Policy Metrics ──────────────────────────────────────────────────

def plot_policy_metrics(exp_names, all_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Policy Metrics", fontweight="bold", fontsize=12, y=1.01)

    panels = [
        ("Policy / Standard deviation", "Policy Std Dev", "Std Dev", "#2166AC"),
        ("Episode / Total timesteps (mean)", "Mean Episode Length", "Steps", "#D6604D"),
    ]

    for ax, (tag, title, ylabel, color) in zip(axes, panels):
        ax.set_title(title, pad=6)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
        ax.spines[["top", "right"]].set_visible(False)

        for i, (name, data) in enumerate(zip(exp_names, all_data)):
            if tag not in data or len(data[tag]["step"]) == 0:
                continue
            c      = get_color(name, i)
            steps  = data[tag]["step"]
            values = data[tag]["value"]
            sm, sm_steps = smooth(values, w=10)
            ax.plot(steps, values, color=c, linewidth=0.5, alpha=0.3)
            ax.plot(sm_steps, sm, color=c, linewidth=2.0,
                    label=name)

        if len(exp_names) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure 4 — Instantaneous Reward (dual-axis style, like paper Fig. 12) ──────

def plot_instant_reward(exp_names, all_data, out_path):
    """Reward mean vs std dev on dual y-axis, paper Fig.12 style."""
    tag_r = "Reward / Instantaneous reward (mean)"
    tag_s = "Policy / Standard deviation"

    # Only plot for the best / latest run
    target = exp_names[-1]
    data   = all_data[-1]
    if tag_r not in data:
        return

    fig, ax1 = plt.subplots(figsize=(9, 4))
    color1 = "#2166AC"
    color2 = "#D6604D"

    ax1.set_title("Instantaneous Reward & Policy Std Dev", pad=8)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Instantaneous Reward", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax1.spines["top"].set_visible(False)

    steps  = data[tag_r]["step"]
    values = data[tag_r]["value"]
    sm, sm_steps = smooth(values, w=15)
    ax1.plot(steps, values, color=color1, linewidth=0.5, alpha=0.3)
    ax1.plot(sm_steps, sm, color=color1, linewidth=2.2,
             label="Instant Reward (mean)")

    if tag_s in data:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Policy Std Dev", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(color2)

        st = data[tag_s]["step"]
        vl = data[tag_s]["value"]
        sm2, sm2_steps = smooth(vl, w=15)
        ax2.plot(st, vl, color=color2, linewidth=0.5, alpha=0.3)
        ax2.plot(sm2_steps, sm2, color=color2, linewidth=2.2,
                 label="Policy Std Dev", linestyle="--")
        ax2.legend(loc="upper right", fontsize=8)

    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure 5 — Escape rate convergence (paper Fig. 21 style) ─────────────────

def plot_escape_rate(exp_names, all_data, out_path):
    """
    Escape success rate over training steps.
    Reads the 'escape_rate' metric logged by the env via extras['log'].
    skrl wraps these as 'Episode / escape_rate (mean)' in TensorBoard.
    """
    # Accept several possible tag names from different skrl/Isaac Lab versions
    _CANDIDATE_TAGS = [
        "Info / escape_rate",          # logged by patched post_interaction (train.py)
        "Episode / escape_rate (mean)",
        "Episode/escape_rate",
        "Metrics/escape_rate",
        "escape_rate",
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title("Escape Rate Convergence", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Escape Rate  (fraction of envs)")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    plotted = 0
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        tag = next((t for t in _CANDIDATE_TAGS if t in data and
                    len(data[t]["step"]) > 0), None)
        if tag is None:
            continue
        color  = get_color(name, i)
        steps  = data[tag]["step"]
        values = data[tag]["value"]

        shade_band(ax, steps, values, color, alpha=0.15)
        sm, sm_steps = smooth(values, w=15)
        ax.plot(sm_steps, sm, color=color, linewidth=2.2,
                label=name, zorder=3)
        ax.plot(steps, values, color=color, linewidth=0.6, alpha=0.3, zorder=2)
        plotted += 1

    if plotted == 0:
        # No escape_rate tag found — print available tags for debugging
        for name, data in zip(exp_names, all_data):
            episode_tags = [t for t in data if "escape" in t.lower() or "Episode" in t]
            if episode_tags:
                print(f"  [escape_rate] Available episode tags for {name}: {episode_tags[:8]}")
        plt.close(fig)
        return

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure 6 — Summary stats table image ──────────────────────────────────────

def plot_summary_table(exp_names, all_data, out_path):
    rows = []
    col_labels = ["Run", "Steps", "Final Reward (mean)", "Min Reward", "Policy Std Dev"]
    for name, data in zip(exp_names, all_data):
        tag_r = "Reward / Total reward (mean)"
        tag_m = "Reward / Total reward (min)"
        tag_s = "Policy / Standard deviation"
        steps   = int(data[tag_r]["step"][-1])  if tag_r in data and len(data[tag_r]["step"]) else 0
        r_mean  = f"{data[tag_r]['value'][-1]:.3f}" if tag_r in data and len(data[tag_r]['value']) else "—"
        r_min   = f"{data[tag_m]['value'][-1]:.3f}" if tag_m in data and len(data[tag_m]['value']) else "—"
        std     = f"{data[tag_s]['value'][-1]:.3f}" if tag_s in data and len(data[tag_s]['value']) else "—"
        rows.append([name, f"{steps:,}", r_mean, r_min, std])

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2166AC")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF3FB")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle("Training Summary", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.exp:
        exp_names = args.exp
    else:
        exp_names = sorted([
            d for d in os.listdir(EXP_BASE)
            if os.path.isdir(os.path.join(EXP_BASE, d)) and d != "plots"
        ])

    if not exp_names:
        print("No experiments found under", EXP_BASE)
        sys.exit(1)

    print(f"\nGenerating publication-quality plots for: {exp_names}\n")

    all_data = []
    for name in exp_names:
        path = os.path.join(EXP_BASE, name)
        if not os.path.isdir(path):
            print(f"  [skip] {name}")
            all_data.append({})
            continue
        print(f"  Loading: {name}")
        data = load_scalars(path)
        all_data.append(data)
        if data:
            max_step = max(data[t]["step"].max() for t in data if len(data[t]["step"]) > 0)
            print(f"    tags={len(data)}  max_step={int(max_step):,}")

    # Per-run plots
    for name, data in zip(exp_names, all_data):
        if not data:
            continue
        pfx = os.path.join(PLOTS_DIR, name)
        plot_instant_reward([name], [data], f"{pfx}_reward_dual_axis.png")

    # Combined / comparison plots
    plot_reward_convergence(exp_names, all_data,
                            os.path.join(PLOTS_DIR, "reward_convergence.png"))
    plot_escape_rate(exp_names, all_data,
                     os.path.join(PLOTS_DIR, "escape_rate.png"))
    plot_losses(exp_names, all_data,
                os.path.join(PLOTS_DIR, "losses.png"))
    plot_policy_metrics(exp_names, all_data,
                        os.path.join(PLOTS_DIR, "policy_metrics.png"))
    plot_summary_table(exp_names, all_data,
                       os.path.join(PLOTS_DIR, "summary_table.png"))

    print(f"\nAll plots saved to: {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()
