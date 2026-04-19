"""
Extra training metric plots not covered by plot_training.py.

Generates:
  1. Curriculum & Milestones   — curriculum_progress + milestone hit rates
  2. Environment Metrics       — mean_dist, mean_vx, mean_abs_slip
  3. Detection Flags           — entrap_flag_rate, torque_anomaly_rate
  4. Reward Breakdown          — instantaneous reward mean/min/max bands
  5. Training Overview         — 2×3 combined figure for paper/report

Usage (no Isaac Sim needed):
    python3 scripts/plot_extra_metrics.py
    python3 scripts/plot_extra_metrics.py --exp ppo_gru_regolith
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_BASE  = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
PLOTS_DIR = os.path.join(EXP_BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def smooth(y, w=20, x=None):
    if x is None:
        x = np.arange(len(y), dtype=np.float32)
    if len(y) < w:
        return y, x
    k = np.ones(w) / w
    s = np.convolve(y, k, mode="valid")
    return s, x[len(x) - len(s):]


def shade_band(ax, steps, values_raw, color, alpha=0.12):
    w = max(5, len(values_raw) // 30)
    if len(values_raw) < w * 2:
        return
    k  = np.ones(w) / w
    mu = np.convolve(values_raw, k, mode="valid")
    sq = np.convolve(values_raw ** 2, k, mode="valid")
    sig = np.sqrt(np.maximum(sq - mu ** 2, 0))
    s = steps[len(steps) - len(mu):]
    ax.fill_between(s, mu - sig, mu + sig, color=color, alpha=alpha, linewidth=0)


def _find_latest_event_file(dirpath):
    """Return the path to the newest tfevents file in dirpath (by mtime)."""
    import glob
    candidates = glob.glob(os.path.join(dirpath, "events.out.tfevents.*"))
    if not candidates:
        return dirpath
    return max(candidates, key=os.path.getmtime)


def load_scalars(path):
    if os.path.isdir(path):
        path = _find_latest_event_file(path)
    ea = event_accumulator.EventAccumulator(
        path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        steps = np.array([e.step for e in ev], dtype=np.float32)
        vals  = np.array([e.value for e in ev], dtype=np.float32)
        order = np.argsort(steps, kind="stable")
        steps, vals = steps[order], vals[order]
        _, unique_idx = np.unique(steps[::-1], return_index=True)
        keep = len(steps) - 1 - unique_idx
        keep.sort()
        out[tag] = {"step": steps[keep], "value": vals[keep]}
    return out


def fmt_step(x, _):
    return f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"


def _plot_tag(ax, data, tag, color, label=None, w=40):
    if tag not in data or len(data[tag]["step"]) == 0:
        return False
    steps = data[tag]["step"]
    vals  = data[tag]["value"]
    shade_band(ax, steps, vals, color, alpha=0.15)
    sm, sm_steps = smooth(vals, w=w, x=steps)
    shade_band(ax, steps, vals, color, alpha=0.10)
    ax.plot(sm_steps, sm, color=color, linewidth=2.5, label=label or tag, zorder=3)
    return True


# ── Figure: Curriculum & Milestones ──────────────────────────────────────────

def plot_curriculum_milestones(data, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Curriculum Progress & Escape Milestones", fontweight="bold", fontsize=12, y=1.01)

    # Curriculum progress
    ax1.set_title("Curriculum Progress", pad=6)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Curriculum Level")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax1.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax1, data, "Info / curriculum_progress", "#2166AC", "Curriculum", w=20)
    ax1.set_ylim(-0.05, 1.1)

    # Milestones + distance
    ax2.set_title("Milestone Hit Rate & Mean Distance", pad=6)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Hit Rate")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax2.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax2, data, "Info / milestone_0_5m", "#4DAC26", "0.5 m")
    _plot_tag(ax2, data, "Info / milestone_1_0m", "#E08214", "1.0 m")
    ax2.legend(loc="upper left")
    ax2.set_ylim(-0.05, 1.1)
    # Secondary axis: mean distance from origin
    if "Info / mean_dist" in data and len(data["Info / mean_dist"]["step"]) > 0:
        ax2r = ax2.twinx()
        ax2r.set_ylabel("Mean Dist / m", color="#2166AC")
        ax2r.tick_params(axis="y", labelcolor="#2166AC")
        ax2r.spines["top"].set_visible(False)
        _plot_tag(ax2r, data, "Info / mean_dist", "#2166AC", "Mean dist (m)")
        ax2r.axhline(1.5, color="#D6604D", linewidth=1.0, linestyle=":", alpha=0.7, label="Escape (1.5 m)")
        ax2r.set_ylim(bottom=0)
        ax2r.legend(loc="center right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure: Environment Metrics ──────────────────────────────────────────────

def plot_env_metrics(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Environment Metrics Over Training", fontweight="bold", fontsize=12, y=1.01)

    panels = [
        ("Info / mean_dist", "Mean Distance from Origin", "Distance / m", "#2166AC"),
        ("Info / mean_vx", "Mean Forward Velocity", "v_x / m·s⁻¹", "#4DAC26"),
        ("Info / mean_abs_slip", "Mean |Slip|", "Slip ratio", "#D6604D"),
    ]

    for ax, (tag, title, ylabel, color) in zip(axes, panels):
        ax.set_title(title, pad=6)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
        ax.spines[["top", "right"]].set_visible(False)
        _plot_tag(ax, data, tag, color, title, w=20)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure: Detection Flags ─────────────────────────────────────────────────

def plot_detection_flags(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("Entrapment & Torque Anomaly Detection Rates", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate (fraction of envs)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)

    _plot_tag(ax, data, "Info / entrap_flag_rate", "#D6604D", "Entrapment flag", w=25)
    _plot_tag(ax, data, "Info / torque_anomaly_rate", "#E08214", "Torque anomaly", w=25)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure: Reward Breakdown (mean/min/max band) ────────────────────────────

def plot_reward_breakdown(data, out_path):
    tag_mean = "Reward / Total reward (mean)"
    tag_min  = "Reward / Total reward (min)"
    tag_max  = "Reward / Total reward (max)"

    if tag_mean not in data:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("Total Episode Reward (mean ± min/max)", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Episode Reward")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)

    steps = data[tag_mean]["step"]
    mean_v = data[tag_mean]["value"]

    if tag_min in data and tag_max in data:
        min_v = data[tag_min]["value"]
        max_v = data[tag_max]["value"]
        # Align lengths
        n = min(len(steps), len(min_v), len(max_v))
        ax.fill_between(steps[:n], min_v[:n], max_v[:n],
                        color="#2166AC", alpha=0.15, label="Min–Max band")

    shade_band(ax, steps, mean_v, "#2166AC", alpha=0.10)
    sm, sm_steps = smooth(mean_v, w=30, x=steps)
    ax.plot(sm_steps, sm, color="#2166AC", linewidth=2.5, label="Mean reward")
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Figure: Training Overview (2×3 combined) ────────────────────────────────

def plot_training_overview(data, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Training Overview — PPO GRU Regolith Escape (200k steps)",
                 fontweight="bold", fontsize=14, y=1.01)

    # Row 1: Reward, Losses (policy+value), Entropy
    ax = axes[0, 0]
    ax.set_title("Episode Reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Reward / Total reward (mean)", "#2166AC", "Mean")

    ax = axes[0, 1]
    ax.set_title("Policy & Value Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Value Loss", color="#D6604D")
    ax.tick_params(axis="y", labelcolor="#D6604D")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Loss / Value loss", "#D6604D", "Value loss")
    ax2b = ax.twinx()
    ax2b.set_ylabel("Policy Loss", color="#2166AC")
    ax2b.tick_params(axis="y", labelcolor="#2166AC")
    ax2b.spines["top"].set_visible(False)
    _plot_tag(ax2b, data, "Loss / Policy loss", "#2166AC", "Policy loss")
    ax.legend(loc="upper right"); ax2b.legend(loc="center right")

    ax = axes[0, 2]
    ax.set_title("Entropy Loss & Policy Std Dev")
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy Loss")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Loss / Entropy loss", "#4DAC26", "Entropy")
    ax2 = ax.twinx()
    ax2.set_ylabel("Std Dev", color="#8073AC")
    ax2.tick_params(axis="y", labelcolor="#8073AC")
    ax2.spines["top"].set_visible(False)
    if "Policy / Standard deviation" in data:
        s = data["Policy / Standard deviation"]["step"]
        v = data["Policy / Standard deviation"]["value"]
        sm, sm_s = smooth(v, w=40, x=s)
        ax2.plot(sm_s, sm, color="#8073AC", linewidth=2.2, linestyle="--", label="Std Dev")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # Row 2: Distance, Milestones, Detection flags
    ax = axes[1, 0]
    ax.set_title("Mean Distance from Origin")
    ax.set_xlabel("Step"); ax.set_ylabel("Distance / m")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / mean_dist", "#2166AC", "Mean dist")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.set_title("Milestone Hit Rates")
    ax.set_xlabel("Step"); ax.set_ylabel("Rate")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / milestone_0_5m", "#4DAC26", "0.5 m")
    _plot_tag(ax, data, "Info / milestone_1_0m", "#E08214", "1.0 m")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.05, 1.1)

    ax = axes[1, 2]
    ax.set_title("Detection Flag Rates")
    ax.set_xlabel("Step"); ax.set_ylabel("Rate")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / entrap_flag_rate", "#D6604D", "Entrapment", w=25)
    _plot_tag(ax, data, "Info / torque_anomaly_rate", "#E08214", "Torque anomaly", w=25)
    _plot_tag(ax, data, "Info / mean_abs_slip", "#2166AC", "Mean |slip|", w=25)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None)
    args = parser.parse_args()

    if args.exp:
        exp_name = args.exp
    else:
        # Auto-detect latest
        dirs = sorted([d for d in os.listdir(EXP_BASE)
                       if os.path.isdir(os.path.join(EXP_BASE, d)) and d != "plots"
                       and d != "episode_data"])
        if not dirs:
            print("No experiments found under", EXP_BASE)
            sys.exit(1)
        exp_name = dirs[-1]

    path = os.path.join(EXP_BASE, exp_name)
    print(f"\nLoading: {exp_name}")
    data = load_scalars(path)
    print(f"  tags={len(data)}  max_step={max(data[t]['step'].max() for t in data if len(data[t]['step'])):.0f}")

    print(f"\nGenerating extra metric plots...\n")

    plot_curriculum_milestones(data, os.path.join(PLOTS_DIR, "curriculum_milestones.png"))
    plot_env_metrics(data, os.path.join(PLOTS_DIR, "env_metrics.png"))
    plot_detection_flags(data, os.path.join(PLOTS_DIR, "detection_flags.png"))
    plot_reward_breakdown(data, os.path.join(PLOTS_DIR, "reward_breakdown.png"))
    plot_training_overview(data, os.path.join(PLOTS_DIR, "training_overview.png"))

    print(f"\nAll extra plots saved to: {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()
