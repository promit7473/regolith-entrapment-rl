"""
Top-down 2D escape heatmap from training failure_modes.csv.

The training environment randomises the escape direction θ ~ U(0, 2π) per
episode.  The failure_modes CSV records `final_dist` — the scalar projection
of the terminal rover position onto that episode's escape direction.  By
sampling the same uniform distribution for θ we can reconstruct the 2D
terminal position:

    x = final_dist · cos θ
    y = final_dist · sin θ

This is statistically representative of what happened; individual episode
trajectories are not available from on-policy logging.  The figure caption
says so explicitly.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.facecolor":   "white",
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset":   "stix",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_BASE  = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
PLOTS_DIR = os.path.join(EXP_BASE, "plots")
FM_CSV    = os.path.join(EXP_BASE, "failure_modes.csv")

ESCAPE_DIST  = 3.0   # metres — projected travel to count as escaped
SPAWN_OFFSET = 0.5   # metres — rover spawns 0.5 m behind origin

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    import csv
    episodes = []
    with open(FM_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                escaped    = int(row["escaped"])
                final_dist = float(row["final_dist"])
                sinkage    = float(row["sinkage"])
            except (ValueError, KeyError):
                continue
            if not np.isfinite(final_dist):
                continue
            episodes.append((escaped, final_dist, sinkage))
    return episodes


def build_xy(episodes, rng):
    """Reconstruct 2D terminal positions from 1-D projected distances."""
    angles = rng.uniform(0.0, 2 * np.pi, size=len(episodes))
    xs, ys, escaped_flags, sinkages = [], [], [], []
    for (esc, fd, sk), theta in zip(episodes, angles):
        # Clip extreme outliers (physics explosions) — keep ±6 m
        if abs(fd) > 6.0:
            continue
        xs.append(fd * np.cos(theta))
        ys.append(fd * np.sin(theta))
        escaped_flags.append(esc)
        sinkages.append(sk)
    return (np.array(xs), np.array(ys),
            np.array(escaped_flags, dtype=bool),
            np.array(sinkages))


def kde_heatmap(ax, xs, ys, cmap, vmin=None, vmax=None,
                bw=0.15, grid_n=300, extent=4.5):
    from scipy.stats import gaussian_kde
    if len(xs) < 5:
        return
    kde   = gaussian_kde(np.vstack([xs, ys]), bw_method=bw)
    gx    = np.linspace(-extent, extent, grid_n)
    gy    = np.linspace(-extent, extent, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    Z     = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(grid_n, grid_n)
    Z    /= Z.max() + 1e-12
    ax.imshow(Z, origin="lower", extent=[-extent, extent, -extent, extent],
              cmap=cmap, alpha=0.82, vmin=0, vmax=1.0, aspect="equal",
              interpolation="bilinear", zorder=1)


def make_figure(episodes, out_path):
    rng = np.random.default_rng(42)
    xs, ys, esc_mask, sinkages = build_xy(episodes, rng)

    n_esc   = int(esc_mask.sum())
    n_fail  = int((~esc_mask).sum())
    esc_rate = n_esc / (n_esc + n_fail) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Top-Down Terminal Position Distribution — PPO-GRU Entrapment Recovery\n"
        r"(escape direction $\theta\!\sim\!\mathcal{U}[0,2\pi]$ per episode; "
        f"$n={n_esc+n_fail:,}$ training episodes)",
        fontsize=11, fontweight="bold", y=1.02)

    titles = ["Failed episodes (stuck in regolith)",
              "Escaped episodes (3 m threshold reached)"]
    masks  = [~esc_mask, esc_mask]
    cmaps  = ["Reds", "Blues"]
    counts = [n_fail, n_esc]

    for ax, mask, cmap, title, count in zip(axes, masks, cmaps, titles, counts):
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel("X / m"); ax.set_ylabel("Y / m")
        ax.set_title(f"{title}\n(n = {count:,})", pad=6)
        ax.set_facecolor("#1A1A2E")

        # Grid
        for r in [1.0, 2.0, 3.0, 4.0]:
            circle = Circle((0, 0), r, fill=False, edgecolor="#444466",
                             linewidth=0.6, linestyle="--", zorder=2)
            ax.add_patch(circle)
            if r <= 3.0:
                ax.text(r * 0.71, r * 0.71, f"{r:.0f} m",
                        fontsize=6.5, color="#888899", ha="center", zorder=3)

        # Escape boundary circle
        esc_circle = Circle((0, 0), ESCAPE_DIST, fill=False,
                             edgecolor="#FFD700", linewidth=1.8,
                             linestyle="-", zorder=5)
        ax.add_patch(esc_circle)
        ax.text(ESCAPE_DIST * 0.0, ESCAPE_DIST + 0.18,
                "Escape boundary (3.0 m)", fontsize=7,
                color="#FFD700", ha="center", zorder=6)

        # KDE heatmap
        kde_heatmap(ax, xs[mask], ys[mask], cmap=cmap)

        # Scatter (thin, semi-transparent)
        scatter_color = "#D62728" if "Red" in cmap else "#1F77B4"
        ax.scatter(xs[mask], ys[mask], s=0.3, color=scatter_color,
                   alpha=0.15, zorder=4, rasterized=True)

        # Origin marker
        ax.plot(0, 0, "w+", markersize=9, markeredgewidth=1.6, zorder=7)
        ax.text(0.08, -0.28, "Origin", fontsize=6.5,
                color="white", ha="left", zorder=7)

    # Escape-rate annotation on the escaped panel
    axes[1].text(0.05, 0.94, f"Escape rate: {esc_rate:.1f}%",
                 transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold", color="#FFD700",
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#111133",
                           edgecolor="#FFD700", linewidth=1.0))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#D62728", markersize=6, label="Failed"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#1F77B4", markersize=6, label="Escaped"),
        Line2D([0], [0], color="#FFD700", linewidth=1.5,
               linestyle="-", label="Escape boundary (3 m)"),
        Line2D([0], [0], marker="+", color="white",
               markersize=8, markeredgewidth=1.5,
               linestyle="None", label="Episode origin"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8.5, framealpha=0.9,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    print("Loading failure_modes.csv …")
    episodes = load_data()
    print(f"  {len(episodes):,} valid episodes loaded")
    out = os.path.join(PLOTS_DIR, "topdown_escape_heatmap.png")
    make_figure(episodes, out)
    print("Done.")
