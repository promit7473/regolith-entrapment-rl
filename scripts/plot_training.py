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


PALETTE = {
    "ppo_regolith":  "#2166AC",
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


def load_scalars(path):
    # Pass the directory directly so EventAccumulator merges ALL event files.
    # (Picking only the latest file loses history when a run produces many files.)
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


def plot_reward_convergence(exp_names, all_data, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#FAFAFA")

    ax.set_title("Reward Convergence", pad=10, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.5,
            color="#CCCCCC", alpha=0.7, zorder=0)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":", zorder=1)

    tag = "Reward / Total reward (mean)"
    plotted = 0
    for i, (name, data) in enumerate(zip(exp_names, all_data)):
        if tag not in data or len(data[tag]["step"]) == 0:
            continue
        color  = get_color(name, i)
        steps  = data[tag]["step"]
        values = data[tag]["value"]


        shade_band(ax, steps, values, color, alpha=0.08)


        ax.plot(steps, values, color=color, linewidth=0.5, alpha=0.22,
                label=f"{name} (raw)", zorder=2)


        sm, sm_steps = smooth(values, w=30, x=steps)
        ax.plot(sm_steps, sm, color=color, linewidth=1.6, alpha=0.75,
                label=f"{name} (MA w=30)", zorder=3)


        from scipy.signal import savgol_filter
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            IsotonicRegression = None

        ISO_COLOR   = "#C0392B"
        ISO_SG_W    = 351
        ISO_SG_POLY = 2

        if IsotonicRegression is not None and len(values) > ISO_SG_W:
            iso = IsotonicRegression(increasing=True)
            y_arr = np.asarray(values, dtype=np.float32)
            x_arr = np.asarray(steps, dtype=np.float32)
            fit = iso.fit_transform(x_arr, y_arr)
            w = ISO_SG_W if ISO_SG_W % 2 == 1 else ISO_SG_W + 1
            w = min(w, len(fit) - (1 - len(fit) % 2))
            if w >= ISO_SG_POLY + 2:


                pad = w // 2
                padded = np.concatenate([
                    np.full(pad, fit[0]), fit, np.full(pad, fit[-1])
                ])
                padded = savgol_filter(padded, window_length=w,
                                       polyorder=ISO_SG_POLY)
                fit = padded[pad:-pad]

                fit[0]  = float(iso.fit_transform(x_arr, y_arr)[0])
                fit[-1] = float(y_arr[-max(1, len(y_arr)//20):].mean())
            steps_arr = np.asarray(steps)

            ax.fill_between(steps_arr, fit.min(), fit, color=ISO_COLOR,
                            alpha=0.08, zorder=3.5)

            ax.plot(steps_arr, fit, color=ISO_COLOR, linewidth=2.2,
                    solid_capstyle="round",
                    label=f"{name} (isotonic regression)", zorder=5)


            n_bars = 12
            idxs = np.linspace(0, len(steps_arr) - 1, n_bars, dtype=int)
            half = max(5, len(y_arr) // (n_bars * 2))
            errs = np.array([
                y_arr[max(0, j - half):min(len(y_arr), j + half)].std()
                for j in idxs
            ])
            ERR_COLOR = "#7B1E1E"
            ax.errorbar(steps_arr[idxs], fit[idxs], yerr=errs,
                        fmt="none", ecolor=ERR_COLOR, elinewidth=1.2,
                        capsize=3, capthick=1.2, alpha=0.85, zorder=5.5,
                        label=f"{name} (±1σ local)")


            start_x, start_y = steps_arr[0], float(fit[0])
            end_x,   end_y   = steps_arr[-1], float(fit[-1])
            ax.scatter([start_x, end_x], [start_y, end_y],
                       s=45, color=ISO_COLOR, zorder=6,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(f"start: {start_y:+.1f}",
                        xy=(start_x, start_y),
                        xytext=(10, -14), textcoords="offset points",
                        fontsize=9, color=ISO_COLOR)
            ax.annotate(f"plateau: {end_y:+.1f}",
                        xy=(end_x, end_y),
                        xytext=(-12, 38), textcoords="offset points",
                        ha="right", fontsize=9, color=ISO_COLOR,
                        fontweight="bold")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    leg = ax.legend(loc="lower right", framealpha=0.95,
                    fontsize=9, edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#666666")
    ax.tick_params(colors="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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
    ax2.axhline(0.9, color="#D6604D", linewidth=1.0, linestyle=":",
                alpha=0.6, label="Escape threshold (0.9 m)")
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
