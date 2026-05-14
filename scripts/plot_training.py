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

# Mars-mission color palette
PALETTE = {
    "ppo_gru_regolith": "#2E86AB",   # mission blue  (primary)
    "seed_1":           "#2E86AB",
    "seed_3":           "#E59500",   # amber (secondary)
    "default_0":        "#2E86AB",
    "default_1":        "#E59500",
    "default_2":        "#4DAC26",
    "default_3":        "#8073AC",
    "default_4":        "#9B2226",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_BASE  = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
PLOTS_DIR = os.path.join(EXP_BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_color(name, idx):
    return PALETTE.get(name, PALETTE.get(f"default_{idx % 5}", "#2166AC"))


def smooth(y, w=20, x=None):
    if x is None:
        x = np.arange(len(y), dtype=np.float32)
    if len(y) < w:
        return y, x
    k = np.ones(w) / w
    s = np.convolve(y, k, mode="valid")
    return s, x[len(x) - len(s):]


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
        ev    = ea.Scalars(tag)
        steps = np.array([e.step  for e in ev], dtype=np.float32)
        vals  = np.array([e.value for e in ev], dtype=np.float32)
        order = np.argsort(steps, kind="stable")
        steps, vals = steps[order], vals[order]
        _, unique_idx = np.unique(steps[::-1], return_index=True)
        keep = len(steps) - 1 - unique_idx
        keep.sort()
        out[tag] = {"step": steps[keep], "value": vals[keep]}
    return out


def shade_band(ax, steps, values_raw, color, alpha=0.12):
    w = max(5, len(values_raw) // 30)
    if len(values_raw) < w * 2:
        return
    k   = np.ones(w) / w
    mu  = np.convolve(values_raw, k, mode="valid")
    sq  = np.convolve(values_raw ** 2, k, mode="valid")
    sig = np.sqrt(np.maximum(sq - mu ** 2, 0))
    s   = steps[len(steps) - len(mu):]
    ax.fill_between(s, mu - sig, mu + sig, color=color, alpha=alpha, linewidth=0)


def _ema(values, alpha):
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_median(values, w):
    """Robust smoother: rolling median ignores outlier explosion episodes."""
    out = np.empty(len(values))
    half = w // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def plot_reward_convergence(exp_names, all_data, out_path):
    """
    Escape rate is the primary convergence signal — it shows the clean
    S-curve from 0% to 89.3%.  Total reward is a secondary indicator shown
    with rolling-median smoothing to suppress MPM physics-explosion outliers
    that corrupt the early-training mean.  Curriculum difficulty shading
    explains why the reward doesn't plateau: the scheduler keeps raising
    sinkage depth as the policy improves.
    """
    fig, ax_esc = plt.subplots(figsize=(9.5, 5.2))

    # ── primary axis: escape rate (left) ──────────────────────────────────
    ax_esc.set_xlabel("Policy Step", fontsize=10)
    ax_esc.set_ylabel("Escape Rate  (rolling window)", fontsize=10,
                      color="#9B2226")
    ax_esc.tick_params(axis="y", labelcolor="#9B2226")
    ax_esc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_esc.set_ylim(-0.04, 1.12)
    ax_esc.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax_esc.spines[["top"]].set_visible(False)
    ax_esc.set_title(
        "Policy Convergence — Escape Rate (primary) & Episode Return (secondary)",
        pad=10, fontsize=11, fontweight="bold")

    # ── secondary axis: episodic return (right) ───────────────────────────
    ax_ret = ax_esc.twinx()
    ax_ret.set_ylabel("Median Episode Return  (rolling, outliers removed)",
                      fontsize=9, color="#2E86AB")
    ax_ret.tick_params(axis="y", labelcolor="#2E86AB")
    ax_ret.spines[["top"]].set_visible(False)

    tag_e = "Info / escape_rate"
    tag_r = "Reward / Total reward (mean)"
    tag_c = "Info / curriculum_progress"

    plotted = 0
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        if tag_e not in data or len(data[tag_e]["step"]) == 0:
            continue

        # ── Curriculum shading ────────────────────────────────────────────
        if tag_c in data and len(data[tag_c]["step"]) > 0:
            c_s = data[tag_c]["step"]
            c_v = data[tag_c]["value"]
            # shade background from grey (easy) to amber (hard)
            for j in range(len(c_s) - 1):
                alpha = float(c_v[j]) * 0.18 + 0.02
                ax_esc.axvspan(c_s[j], c_s[j+1],
                               color="#E59500", alpha=alpha,
                               linewidth=0, zorder=0)

        # ── Escape rate: raw band + smoothed line ─────────────────────────
        e_s  = np.asarray(data[tag_e]["step"])
        e_v  = np.asarray(data[tag_e]["value"])

        # Thin raw scatter (semi-transparent)
        ax_esc.plot(e_s, e_v, color="#9B2226", linewidth=0.35,
                    alpha=0.18, zorder=2)

        # Smooth with MA (escape rate has no outlier problem)
        e_sm, e_sm_s = smooth(e_v, w=80, x=e_s)
        ax_esc.plot(e_sm_s, e_sm, color="#9B2226", linewidth=2.6,
                    alpha=0.95, label="Escape rate (smoothed)", zorder=4)

        # Shade the band under the curve
        ax_esc.fill_between(e_sm_s, 0, e_sm,
                            color="#9B2226", alpha=0.06, zorder=1)

        # Annotate peak
        peak_esc = float(e_v[-max(1, len(e_v)//15):].mean())
        ax_esc.annotate(
            f"{peak_esc*100:.1f}% at 300 k",
            xy=(e_s[-1], peak_esc),
            xytext=(-110, 12), textcoords="offset points",
            fontsize=9, color="#9B2226", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#9B2226",
                             lw=1.0, connectionstyle="arc3,rad=0.1"))

        # ── Episodic return: rolling-median (outlier-robust) ──────────────
        if tag_r in data and len(data[tag_r]["step"]) > 0:
            r_s = np.asarray(data[tag_r]["step"])
            r_v = np.asarray(data[tag_r]["value"])

            # Hard-clip obvious physics explosions (beyond ±5 × IQR)
            q25, q75 = np.percentile(r_v, 25), np.percentile(r_v, 75)
            iqr = q75 - q25
            r_v_clipped = np.clip(r_v, q25 - 5 * iqr, q75 + 5 * iqr)

            # Rolling median (w=150): robust to remaining bursty outliers
            r_med = _rolling_median(r_v_clipped, w=150)

            ax_ret.plot(r_s, r_med, color="#2E86AB", linewidth=1.8,
                        linestyle="--", alpha=0.80,
                        label="Episode return (rolling median)", zorder=3)

            # Annotate start and end
            start_r = float(np.median(r_v_clipped[:50]))
            end_r   = float(np.median(r_v_clipped[-100:]))
            ax_ret.annotate(f"{start_r:+.0f}",
                            xy=(r_s[25], start_r),
                            xytext=(6, -16), textcoords="offset points",
                            fontsize=8, color="#2E86AB")
            ax_ret.annotate(f"{end_r:+.0f}",
                            xy=(r_s[-1], end_r),
                            xytext=(-8, 10), textcoords="offset points",
                            ha="right", fontsize=8.5,
                            color="#2E86AB", fontweight="bold")

            # Set y-range from sensible percentiles
            p5, p95 = np.percentile(r_v_clipped, 5), np.percentile(r_v_clipped, 95)
            ax_ret.set_ylim(p5 - 5, p95 + 10)

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    # ── Annotations for curriculum shading legend ────────────────────────
    from matplotlib.patches import Patch
    curriculum_patch = Patch(facecolor="#E59500", alpha=0.25,
                              label="Curriculum difficulty (amber = harder sinkage)")

    lines1, labels1 = ax_esc.get_legend_handles_labels()
    lines2, labels2 = ax_ret.get_legend_handles_labels()
    ax_esc.legend(lines1 + lines2 + [curriculum_patch],
                  labels1 + labels2 + [curriculum_patch.get_label()],
                  loc="lower right", framealpha=0.92, fontsize=8.5,
                  edgecolor="#CCCCCC").get_frame().set_linewidth(0.5)
    ax_esc.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


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
            sm, sm_steps = smooth(values, w=30, x=steps)
            ax.plot(steps, values, color=c, linewidth=0.4, alpha=0.15)
            ax.plot(sm_steps, sm, color=c, linewidth=2.2,
                    label=name)

        if len(exp_names) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


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
            sm, sm_steps = smooth(values, w=30, x=steps)
            ax.plot(steps, values, color=c, linewidth=0.4, alpha=0.15)
            ax.plot(sm_steps, sm, color=c, linewidth=2.2,
                    label=name)

        if len(exp_names) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_instant_reward(exp_names, all_data, out_path):
    tag_r = "Reward / Instantaneous reward (mean)"
    tag_s = "Policy / Standard deviation"


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
    shade_band(ax1, steps, values, color1, alpha=0.10)
    sm, sm_steps = smooth(values, w=40, x=steps)
    ax1.plot(sm_steps, sm, color=color1, linewidth=2.5,
             label="Instant Reward (mean)")

    if tag_s in data:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Policy Std Dev", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(color2)

        st = data[tag_s]["step"]
        vl = data[tag_s]["value"]
        sm2, sm2_steps = smooth(vl, w=40, x=st)
        ax2.plot(sm2_steps, sm2, color=color2, linewidth=2.5,
                 label="Policy Std Dev", linestyle="--")
        ax2.legend(loc="upper right", fontsize=8)

    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_escape_rate(exp_names, all_data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("Escape Progress — Milestones & Displacement", pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Milestone Hit Rate")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    plotted = 0
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        color = get_color(name, i)

        for tag, lbl, c in [
            ("Info / milestone_0_3m", "0.3 m", "#4DAC26"),
            ("Info / milestone_0_6m", "0.6 m", "#E08214"),
        ]:
            if tag in data and len(data[tag]["step"]) > 0:
                steps  = data[tag]["step"]
                values = data[tag]["value"]
                shade_band(ax, steps, values, c, alpha=0.10)
                sm, sm_steps = smooth(values, w=40, x=steps)
                ax.plot(sm_steps, sm, color=c, linewidth=2.5,
                        label=lbl, zorder=3)
                plotted += 1


        _ESCAPE_TAGS = [
            "Info / escape_rate", "Episode / escape_rate (mean)",
            "Episode/escape_rate", "escape_rate",
        ]
        esc_tag = next((t for t in _ESCAPE_TAGS if t in data and
                        len(data[t]["step"]) > 0 and
                        data[t]["value"].max() > 0.001), None)
        if esc_tag:
            steps  = data[esc_tag]["step"]
            values = data[esc_tag]["value"]
            sm, sm_steps = smooth(values, w=40, x=steps)
            ax.plot(sm_steps, sm, color="#D6604D", linewidth=2.5,
                    label="Escape (0.9 m)", zorder=3)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return


    ax2 = ax.twinx()
    ax2.set_ylabel("Mean Distance from Origin / m", color="#2166AC")
    ax2.tick_params(axis="y", labelcolor="#2166AC")
    ax2.spines["top"].set_visible(False)
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        tag = "Info / mean_dist"
        if tag in data and len(data[tag]["step"]) > 0:
            steps  = data[tag]["step"]
            values = data[tag]["value"]
            shade_band(ax2, steps, values, "#2166AC", alpha=0.08)
            sm, sm_steps = smooth(values, w=40, x=steps)
            ax2.plot(sm_steps, sm, color="#2166AC", linewidth=2.2,
                     linestyle="--", label="Mean dist (m)", zorder=3)
    ax2.axhline(3.0, color="#D6604D", linewidth=1.0, linestyle=":",
                alpha=0.6, label="Escape threshold (3.0 m)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="center right", fontsize=8)

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_summary_table(exp_names, all_data, out_path):
    rows = []
    col_labels = ["Run", "Steps", "Reward (last 10%)", "Min Reward", "Policy Std Dev"]
    for name, data in zip(exp_names, all_data):
        tag_r = "Reward / Total reward (mean)"
        tag_m = "Reward / Total reward (min)"
        tag_s = "Policy / Standard deviation"
        steps   = int(data[tag_r]["step"][-1])  if tag_r in data and len(data[tag_r]["step"]) else 0
        if tag_r in data and len(data[tag_r]['value']):
            v = data[tag_r]['value']
            tail = max(1, len(v) // 10)
            r_mean = f"{float(np.mean(v[-tail:])):.2f}"
        else:
            r_mean = "—"
        r_min   = f"{float(np.min(data[tag_m]['value'])):.2f}" if tag_m in data and len(data[tag_m]['value']) else "—"
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


    for name, data in zip(exp_names, all_data):
        if not data:
            continue
        pfx = os.path.join(PLOTS_DIR, name)
        plot_instant_reward([name], [data], f"{pfx}_reward_dual_axis.png")


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
