import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "#FAFAFA",
    "axes.edgecolor":       "#444444",
    "axes.linewidth":       0.9,
    "axes.grid":            True,
    "grid.color":           "#E0E0E0",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.8,
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset":     "stix",
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
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#CCCCCC",
    "lines.linewidth":      2.0,
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "savefig.facecolor":    "white",
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
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


def _find_event_dir(path):
    for root, _dirs, files in os.walk(path):
        if any(f.startswith("events.out.tfevents") for f in files):
            return root
    return path


def load_scalars(path):
    path = _find_event_dir(path)
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


def plot_curriculum_milestones(data, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Curriculum Progress & Escape Milestones", fontweight="bold", fontsize=12, y=1.01)


    ax1.set_title("Curriculum Progress", pad=6)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Curriculum Level")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax1.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax1, data, "Info / curriculum_progress", "#2166AC", "Curriculum", w=20)
    ax1.set_ylim(-0.05, 1.1)


    ax2.set_title("Escape Progress Along Heading & Mean Distance", pad=6)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Fraction of Envs Past Threshold")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax2.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax2, data, "Info / progress_0_5m", "#4DAC26", "0.5 m")
    _plot_tag(ax2, data, "Info / progress_1_0m", "#E08214", "1.0 m")
    _plot_tag(ax2, data, "Info / progress_2_0m", "#B2182B", "2.0 m")
    ax2.legend(loc="upper left")
    ax2.set_ylim(-0.05, 1.1)
    if "Info / mean_dist" in data and len(data["Info / mean_dist"]["step"]) > 0:
        ax2r = ax2.twinx()
        ax2r.set_ylabel("Mean Projected Dist / m", color="#2166AC")
        ax2r.tick_params(axis="y", labelcolor="#2166AC")
        ax2r.spines["top"].set_visible(False)
        _plot_tag(ax2r, data, "Info / mean_dist", "#2166AC", "Mean dist (m)")
        ax2r.axhline(3.0, color="#D6604D", linewidth=1.0, linestyle=":", alpha=0.7, label="Escape (3.0 m)")
        ax2r.set_ylim(bottom=0)
        ax2r.legend(loc="center right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


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


def plot_detection_flags(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("Entrapment & Torque Anomaly Detection Rates", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate (fraction of envs)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)

    _plot_tag(ax, data, "Info / entrap_flag_rate", "#D6604D", "Entrapment flag", w=25)
    _plot_tag(ax, data, "Info / slip_anomaly_rate", "#E08214", "Slip anomaly", w=25)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_reward_breakdown(data, out_path):
    components = [
        ("Info / rew_progress", "r_progress",  "#1B7837"),
        ("Info / rew_escape",   "r_escape",    "#4DAC26"),
        ("Info / rew_rocking",  "r_rocking",   "#66BD63"),
        ("Info / pen_slip",     "-p_slip",     "#D6604D"),
        ("Info / pen_tilt",     "-p_tilt",     "#E08214"),
        ("Info / pen_smooth",   "-p_smooth",   "#FDAE61"),
        ("Info / pen_abnormal", "-p_abnormal", "#B2182B"),
    ]

    if not any(tag in data for tag, _, _ in components):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Reward Breakdown — Per-Component Contribution", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Per-Step Reward Term")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#666", linewidth=0.7, linestyle=":")

    plotted = 0
    for tag, label, color in components:
        if tag not in data or len(data[tag]["step"]) == 0:
            continue
        steps  = data[tag]["step"]
        values = data[tag]["value"]
        sm, sm_steps = smooth(values, w=30, x=steps)
        ax.plot(sm_steps, sm, color=color, linewidth=2.0, label=label, zorder=3)
        plotted += 1


    if plotted == 0:
        plt.close(fig)
        return

    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_training_overview(data, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Training Overview — PPO-GRU Regolith Escape (300 k steps)",
                 fontweight="bold", fontsize=14, y=1.01)


    ax = axes[0, 0]
    ax.set_title("Escape Rate")
    ax.set_xlabel("Step"); ax.set_ylabel("Escape Rate")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.03, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / escape_rate", "#9B2226", "Escape rate", w=80)
    if "Info / escape_rate" in data and len(data["Info / escape_rate"]["step"]) > 0:
        peak = float(data["Info / escape_rate"]["value"][-50:].mean())
        step_end = float(data["Info / escape_rate"]["step"][-1])
        ax.axhline(peak, color="#9B2226", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(step_end * 0.05, peak + 0.02,
                f"{peak*100:.1f}%", fontsize=8, color="#9B2226")

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


    ax = axes[1, 0]
    ax.set_title("Mean Distance from Origin")
    ax.set_xlabel("Step"); ax.set_ylabel("Distance / m")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / mean_dist", "#2166AC", "Mean dist")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.set_title("Escape Progress Along Heading (omnidirectional)")
    ax.set_xlabel("Step"); ax.set_ylabel("Fraction Past Threshold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / progress_0_5m",  "#4DAC26", "0.5 m")
    _plot_tag(ax, data, "Info / progress_1_0m",  "#E08214", "1.0 m")
    _plot_tag(ax, data, "Info / progress_2_0m",  "#B2182B", "2.0 m")
    _plot_tag(ax, data, "Info / progress_3_0m",  "#762A83", "3.0 m (escape — fully clear)")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.05, 1.1)

    ax = axes[1, 2]
    ax.set_title("Detection Flag Rates")
    ax.set_xlabel("Step"); ax.set_ylabel("Rate")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    _plot_tag(ax, data, "Info / entrap_flag_rate", "#D6604D", "Entrapment", w=25)
    _plot_tag(ax, data, "Info / slip_anomaly_rate", "#E08214", "Slip anomaly", w=25)
    _plot_tag(ax, data, "Info / mean_abs_slip", "#2166AC", "Mean |slip|", w=25)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None)
    args = parser.parse_args()

    if args.exp:
        exp_name = args.exp
    else:

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
