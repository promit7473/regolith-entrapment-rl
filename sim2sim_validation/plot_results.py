import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#444444",
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.color":           "#DDDDDD",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.6,
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset":     "stix",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#CCCCCC",
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

PRETTY = {
    "recovery_gps":    "Ours (GPS-directed)",
    "recovery_random": "Recovery, random heading",
    "no_recovery":     "No recovery (baseline)",
}

COLORS = {
    "recovery_gps":    "#2E86AB",
    "recovery_random": "#E59500",
    "no_recovery":     "#9B2226",
}


def _draw_map_pin(ax, x, y, color, label=None,
                  head_r_data=0.25, tail_drop_data=0.55,
                  hole_frac=0.40, edge="white", lw=1.0):
    from matplotlib.patches import PathPatch, Circle
    from matplotlib.path import Path

    cx, cy_head = x, y + tail_drop_data
    r = head_r_data


    import math
    theta = math.radians(60.0)
    lx = cx - r * math.sin(theta)
    rx = cx + r * math.sin(theta)
    ly = cy_head - r * math.cos(theta)


    verts = [
        (x, y),
        (rx, y + 0.25 * tail_drop_data),
        (rx + 0.4*r, ly - 0.1*r),
        (rx, ly),

        (rx + 0.55*r, ly + 0.2*r),  (cx + 0.55*r, cy_head + r), (cx, cy_head + r),
        (cx - 0.55*r, cy_head + r), (lx - 0.55*r, ly + 0.2*r),  (lx, ly),
        (lx - 0.4*r, ly - 0.1*r),
        (lx, y + 0.25 * tail_drop_data),
        (x, y),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
    ]
    pin = PathPatch(Path(verts, codes), facecolor=color,
                    edgecolor=edge, linewidth=lw, zorder=6,
                    label=label)
    ax.add_patch(pin)

    ax.add_patch(Circle((cx, cy_head), r * hole_frac,
                        facecolor="white", edgecolor=color,
                        linewidth=0.6, zorder=7))


def find_latest_json(experiments_dir: str) -> str:
    pattern = os.path.join(experiments_dir, "summary_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No summary_*.json in {experiments_dir}")
    return matches[-1]


def load_data(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    if "experiments" not in data:
        raise ValueError(
            f"{json_path} has no 'experiments' key — re-run with the unified "
            "run_validation.py (--experiments flag)."
        )
    return data


def plot_rates(experiments: dict, out_path: str):
    modes = list(experiments.keys())
    recovery = [experiments[m]["recovery_rate"]   * 100 for m in modes]
    goal     = [experiments[m]["goal_reach_rate"] * 100 for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - width/2, recovery, width, label="Recovery rate",
           color=[COLORS.get(m, "#666") for m in modes], alpha=0.85,
           edgecolor="#222", linewidth=0.5)
    ax.bar(x + width/2, goal, width, label="Goal-reach rate",
           color=[COLORS.get(m, "#666") for m in modes], alpha=0.45,
           edgecolor="#222", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in modes], rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Recovery and Goal-Reach Rates by Experiment")
    ax.legend(loc="upper right")

    for i, (r, g) in enumerate(zip(recovery, goal)):
        ax.text(i - width/2, r + 1.5, f"{r:.0f}%", ha="center", fontsize=8)
        ax.text(i + width/2, g + 1.5, f"{g:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_path_efficiency(experiments: dict, out_path: str):
    modes = list(experiments.keys())
    data, labels, colors = [], [], []
    for m in modes:
        trials = experiments[m].get("trials", [])
        effs = [t["path_efficiency"] for t in trials if t["reached_goal"]]
        if effs:
            data.append(effs)
            labels.append(PRETTY.get(m, m))
            colors.append(COLORS.get(m, "#666"))

    if not data:
        return False

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#222", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#222")

    ax.set_ylabel("Path efficiency (straight-line / actual)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Path Efficiency by Experiment (goal-reaching trials)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_heading_error(experiments: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plotted = False
    for m, summary in experiments.items():
        trials = summary.get("trials", [])
        errs = [t["escape_heading_error"] for t in trials if t["escaped"]]
        if not errs:
            continue
        ax.hist(errs, bins=18, range=(0, 180), alpha=0.55,
                color=COLORS.get(m, "#666"), edgecolor="#222", linewidth=0.4,
                label=f"{PRETTY.get(m, m)} (n={len(errs)})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Escape heading error vs. true direction to B (deg)")
    ax.set_ylabel("Number of trials")
    ax.set_title("Escape Heading Error: GPS-Directed vs. Random Ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_time_to_escape(experiments: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plotted = False
    for m, summary in experiments.items():
        trials = summary.get("trials", [])
        times = [t["time_to_escape"] for t in trials
                 if t["escaped"] and t["time_to_escape"] > 0]
        if not times:
            continue
        ax.hist(times, bins=15, alpha=0.55,
                color=COLORS.get(m, "#666"), edgecolor="#222", linewidth=0.4,
                label=f"{PRETTY.get(m, m)} (n={len(times)})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Time to escape (policy steps; 1 step = 0.04 s)")
    ax.set_ylabel("Number of trials")
    ax.set_title("Distribution of Time-to-Escape")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_trajectory(mode_name: str, summary: dict, out_path: str,
                    style: str = "color"):
    MONO = (style == "mono")


    P_BG       = "white"           if MONO else "#FBFAF6"
    P_TEXT     = "#000000"         if MONO else "#1C1C1E"
    P_TEXT2    = "#444444"         if MONO else "#5A5C60"
    P_GRID     = "#D8D8D8"         if MONO else "#E6E4DE"
    P_SAND_F   = "#E8E8E8"         if MONO else "#E8D2A8"
    P_SAND_E   = "#888888"         if MONO else "#8C6F3F"
    P_SAND_T   = "#5A5A5A"         if MONO else "#6E4F1F"
    P_PATH_CM  = "Greys"           if MONO else "viridis"
    P_HALO     = "white"
    P_PIN_A    = "#222222"         if MONO else "#2A7F62"
    P_PIN_B    = "#000000"         if MONO else "#B5341E"
    P_PIN_SH   = "#BBBBBB"         if MONO else "#A8A39A"
    P_ENTRAP_F      = "#000000" if MONO else "#7A2E2A"
    P_ENTRAP_F_FILL = "#666666" if MONO else "#D9A05B"
    P_DRIVE_CM = ("Greys" if MONO else "cividis")
    P_OUTCOME_OK      = "#444444" if MONO else "#1F6E47"
    P_OUTCOME_PARTIAL = "#666666" if MONO else "#A37800"
    P_OUTCOME_FAIL    = "#000000" if MONO else "#8C2718"
    P_EVENT_TRIG = "#000000" if MONO else "#7A2E2A"
    P_EVENT_FREE = "#444444" if MONO else "#1F6E47"
    trace = summary.get("trace")
    if not trace or not trace.get("t"):
        return False

    t       = np.array(trace["t"])
    pos     = np.array(trace["pos_xy"])
    action  = np.array(trace["action"])
    mode    = np.array(trace["mode"])
    entrap  = np.array(trace["entrap_flag"])
    spawn   = np.array(trace["spawn_xy"])
    goal    = np.array(trace["goal_xy"])

    drive_cmd = action[:, :6]
    steer_cmd = action[:, 6:]


    final_dist = float(np.linalg.norm(pos[-1] - goal))
    arrival_r  = 0.5
    if final_dist < arrival_r:
        outcome, outcome_col = "GOAL REACHED", P_OUTCOME_OK
    elif (mode == 1).any() and (mode[-50:] == 0).all():
        outcome, outcome_col = "ESCAPED, GOAL MISSED", P_OUTCOME_PARTIAL
    else:
        outcome, outcome_col = "TIMEOUT", P_OUTCOME_FAIL
    n_escapes = int(((np.diff(mode) == 1)).sum())

    fig = plt.figure(figsize=(7.2, 8.6), facecolor="white")
    gs  = fig.add_gridspec(
        3, 1, height_ratios=[0.75, 0.75, 4.8], hspace=0.32,
        left=0.11, right=0.97, top=0.905, bottom=0.06,
    )


    fig.text(
        0.11, 0.965,
        f"Rover trajectory — {PRETTY.get(mode_name, mode_name)}",
        fontsize=14.5, fontweight="bold", color=P_TEXT,
        ha="left", va="top", family="serif",
    )
    fig.text(
        0.11, 0.939,
        f"Buried-spawn → 3 m goal in granular regolith  ·  "
        f"escape primitive fired {n_escapes}×  ·  "
        r"final $\|\mathbf{p} - \mathbf{B}\|$ = "
        f"{final_dist:.2f} m",
        fontsize=9, color=P_TEXT2, ha="left", va="top",
        style="italic",
    )
    fig.text(
        0.97, 0.962, f"  {outcome}  ",
        ha="right", va="top", fontsize=9.5, fontweight="bold",
        color="white", family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=outcome_col,
                  edgecolor="none"),
    )

    def _despine(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(direction="out", length=3, color="#666666")
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#666666")


    ax1 = fig.add_subplot(gs[0])
    wheel_labels = ["FL", "FR", "ML", "MR", "RL", "RR"]
    wheel_colors = matplotlib.colormaps[P_DRIVE_CM](np.linspace(0.15, 0.85, 6))
    for i in range(6):
        ax1.plot(t, drive_cmd[:, i], color=wheel_colors[i],
                 linewidth=1.0, alpha=0.9, label=wheel_labels[i])
    ax1.axhline(0.0, color="#BBBBBB", linewidth=0.5, linestyle=":")
    ax1.set_ylabel(r"drive cmd", fontsize=9.5, style="italic")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(ncol=6, fontsize=7, loc="upper right",
               handlelength=1.2, columnspacing=0.8, frameon=False)
    ax1.tick_params(labelbottom=False)
    _despine(ax1)


    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(t, steer_cmd.min(axis=1), steer_cmd.max(axis=1),
                     color=P_ENTRAP_F, alpha=0.18, label="Steer range")
    ax2.plot(t, steer_cmd.mean(axis=1), color=P_ENTRAP_F,
             linewidth=1.4, label="Mean steer cmd")
    ax2.fill_between(t, 0, entrap, color=P_ENTRAP_F_FILL, alpha=0.30,
                     step="pre", label="Entrap flag")
    ax2.set_ylabel(r"steer  /  flag", fontsize=9.5, style="italic")
    ax2.set_xlabel(r"$t$  /  s", fontsize=10)
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(fontsize=7, loc="upper right", frameon=False, ncol=3)
    _despine(ax2)


    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(P_BG)
    sand_half = 1.75
    bed_cx, bed_cy = spawn[0], spawn[1]


    ax3.add_patch(plt.Rectangle(
        (bed_cx - sand_half, bed_cy - sand_half),
        2 * sand_half, 2 * sand_half,
        facecolor=P_SAND_F, edgecolor="none", alpha=0.45, zorder=1,
        label="Regolith bed (3.5 m)",
    ))

    for off, lw, alpha in [(0.0, 1.2, 0.7), (-0.04, 0.5, 0.45)]:
        ax3.add_patch(plt.Rectangle(
            (bed_cx - sand_half + off, bed_cy - sand_half + off),
            2 * (sand_half - off), 2 * (sand_half - off),
            facecolor="none", edgecolor=P_SAND_E,
            linewidth=lw, alpha=alpha, zorder=1.1,
        ))
    ax3.text(bed_cx, bed_cy - sand_half + 0.16, "regolith",
             ha="center", va="bottom", fontsize=9,
             color=P_SAND_T, style="italic", alpha=0.85, zorder=1.2,
             family="serif")


    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe
    points   = pos.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    halo = LineCollection(segments, color=P_HALO, linewidth=4.5,
                          alpha=0.85, zorder=2.5)
    ax3.add_collection(halo)
    lc = LineCollection(segments, cmap=P_PATH_CM, linewidth=2.4,
                        norm=plt.Normalize(t.min(), t.max()), zorder=3)
    lc.set_array(t[:-1])
    ax3.add_collection(lc)


    mode_changes = np.where(np.diff(mode) != 0)[0]
    placed_labels = []
    for i in mode_changes:
        m_to = mode[i + 1]
        x, y = pos[i+1]
        if m_to == 1:
            ax3.plot(x, y, marker="X", color=P_EVENT_TRIG, markersize=8,
                     markeredgecolor="white", markeredgewidth=0.9, zorder=5)
            label, lc_color = "Entrap → escape", P_EVENT_TRIG
            dy = 0.55
        elif m_to == 0 and mode[i] == 1:
            ax3.plot(x, y, marker="o", color=P_EVENT_FREE, markersize=8,
                     markeredgecolor="white", markeredgewidth=0.9, zorder=5)
            label, lc_color = "Freed → nav", P_EVENT_FREE
            dy = -0.55
        else:
            continue

        if any(abs(px - x) < 0.7 and (pdy > 0) == (dy > 0)
               for (px, py, pdy) in placed_labels):
            dy = -dy
        ax3.annotate(
            label, xy=(x, y), xytext=(x + 0.25, y + dy),
            fontsize=7.5, color=lc_color, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=lc_color, linewidth=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=lc_color,
                            linewidth=0.7, alpha=0.7,
                            connectionstyle="arc3,rad=0.0"),
        )
        placed_labels.append((float(x), float(y), float(dy)))


    for sx, sy in [(spawn[0], spawn[1]), (goal[0], goal[1])]:
        _draw_map_pin(ax3, sx + 0.04, sy - 0.04, color=P_PIN_SH,
                      edge="none", lw=0.0)
    _draw_map_pin(ax3, spawn[0], spawn[1], color=P_PIN_A, label="Spawn (A)")
    _draw_map_pin(ax3, goal[0],  goal[1],  color=P_PIN_B, label="Goal (B)")


    pin_top_pad = 0.95
    all_x = np.concatenate([
        pos[:, 0], [spawn[0], goal[0]],
        [bed_cx-sand_half, bed_cx+sand_half],
    ])
    all_y = np.concatenate([
        pos[:, 1], [spawn[1], goal[1]],
        [bed_cy-sand_half, bed_cy+sand_half],
        [spawn[1] + pin_top_pad, goal[1] + pin_top_pad],
    ])
    margin = 0.6
    ax3.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax3.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"$x$  /  m", fontsize=10)
    ax3.set_ylabel(r"$y$  /  m", fontsize=10)
    ax3.grid(True, which="major", color=P_GRID, linewidth=0.5, alpha=0.7)
    ax3.grid(True, which="minor", color=P_GRID, linewidth=0.3, alpha=0.4)
    ax3.minorticks_on()
    ax3.tick_params(which="minor", length=2, color="#999999")
    ax3.set_axisbelow(True)
    _despine(ax3)


    xlim, ylim = ax3.get_xlim(), ax3.get_ylim()
    nx = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    ny = ylim[0] + 0.10 * (ylim[1] - ylim[0])
    arrow_len = 0.04 * (ylim[1] - ylim[0])
    ax3.annotate(
        "N", xy=(nx, ny + arrow_len * 1.2),
        xytext=(nx, ny - arrow_len * 0.4),
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#444444",
        arrowprops=dict(facecolor="#444444", edgecolor="#444444",
                        width=2.0, headwidth=8, headlength=8),
    )

    bar_len_m = 1.0
    bx0 = xlim[1] - 0.05 * (xlim[1] - xlim[0]) - bar_len_m
    by0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax3.plot([bx0, bx0 + bar_len_m], [by0, by0], color="#222222",
             linewidth=2.5, solid_capstyle="butt", zorder=8)
    ax3.text(bx0 + bar_len_m / 2, by0 + 0.06 * (ylim[1] - ylim[0]),
             f"{bar_len_m:.0f} m", ha="center", va="bottom",
             fontsize=8, color="#222222", fontweight="bold")

    leg = ax3.legend(loc="upper right", fontsize=8.5, frameon=True,
                     facecolor="white", edgecolor="#CCCCCC",
                     framealpha=0.95, borderpad=0.6, handletextpad=0.6)
    leg.set_zorder(10)
    leg.get_frame().set_linewidth(0.6)


    cbar = fig.colorbar(lc, ax=ax3, orientation="horizontal",
                        fraction=0.045, pad=0.11, aspect=42)
    cbar.set_label(r"episode time  /  s", fontsize=9.5, style="italic")
    cbar.ax.tick_params(labelsize=8.5, color="#666666", length=3)
    cbar.outline.set_edgecolor("#CCCCCC")
    cbar.outline.set_linewidth(0.5)

    fig.savefig(out_path)
    plt.close(fig)
    return True


def write_summary_table(experiments: dict, md_path: str, csv_path: str):
    cols = [
        ("Experiment",          lambda m, s: PRETTY.get(m, m)),
        ("Trials",              lambda m, s: s["n_trials"]),
        ("Recovery %",          lambda m, s: f"{s['recovery_rate']*100:.1f}"),
        ("Goal-reach %",        lambda m, s: f"{s['goal_reach_rate']*100:.1f}"),
        ("Time-to-escape μ±σ",  lambda m, s: f"{s['time_to_escape']['mean']:.0f} ± {s['time_to_escape']['std']:.0f}"),
        ("Time-to-goal μ±σ",    lambda m, s: f"{s['time_to_goal']['mean']:.0f} ± {s['time_to_goal']['std']:.0f}"),
        ("Path eff. μ±σ",       lambda m, s: f"{s['path_efficiency']['mean']:.3f} ± {s['path_efficiency']['std']:.3f}"),
        ("Hdg err deg μ±σ",     lambda m, s: f"{s['heading_error_deg']['mean']:.1f} ± {s['heading_error_deg']['std']:.1f}"),
    ]

    headers = [c[0] for c in cols]
    rows = [[fn(m, s) for _, fn in cols] for m, s in experiments.items()]


    with open(md_path, "w") as f:
        f.write("# Sim2Sim Validation Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")


    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot sim2sim validation results")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to summary_*.json. Default: latest in experiments/sim2sim/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: alongside the input JSON")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir   = os.path.join(repo_root, "experiments", "sim2sim")

    json_path = args.json or find_latest_json(exp_dir)
    out_dir   = args.out_dir or os.path.join(
        os.path.dirname(json_path),
        "figs_" + os.path.splitext(os.path.basename(json_path))[0]
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[plot] Reading: {json_path}")
    print(f"[plot] Writing: {out_dir}/")

    data = load_data(json_path)
    experiments = data["experiments"]

    plot_rates(experiments,           os.path.join(out_dir, "rates.png"))
    if plot_path_efficiency(experiments, os.path.join(out_dir, "path_efficiency.png")):
        print("  ✓ path_efficiency.png")
    if plot_heading_error(experiments,   os.path.join(out_dir, "heading_error.png")):
        print("  ✓ heading_error.png")
    if plot_time_to_escape(experiments,  os.path.join(out_dir, "time_to_escape.png")):
        print("  ✓ time_to_escape.png")
    print("  ✓ rates.png")

    for mode_name, summary in experiments.items():
        traj_path = os.path.join(out_dir, f"trajectory_{mode_name}.png")
        if plot_trajectory(mode_name, summary, traj_path, style="color"):
            print(f"  ✓ trajectory_{mode_name}.png")
        mono_path = os.path.join(out_dir, f"trajectory_{mode_name}_mono.png")
        if plot_trajectory(mode_name, summary, mono_path, style="mono"):
            print(f"  ✓ trajectory_{mode_name}_mono.png")

    write_summary_table(
        experiments,
        os.path.join(out_dir, "summary_table.md"),
        os.path.join(out_dir, "summary_table.csv"),
    )
    print("  ✓ summary_table.md, summary_table.csv")
    print(f"\n[plot] Done. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
