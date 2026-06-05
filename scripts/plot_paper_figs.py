"""Modern paper figures: escape comparison, action behaviour, reward convergence.

- escape_comparison.png       : horizontal lollipop/bar, bootstrap CIs, direct labels
- policy_action_behavior.png  : clean grouped horizontal bars (steering vs rocking)
- reward_convergence.png       : seed-1 + seed-3 episode return, outlier-clipped,
                                 EMA-smoothed + isotonic monotone envelope

Reads escape JSONs from experiments/escape_eval/sweep/ and TensorBoard event files
from experiments/regolith_recovery/seed_{1,3}/. Writes to paper/figures/.

Usage:
  python3 scripts/plot_paper_figs.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "experiments/escape_eval/sweep")
OUT = os.path.join(ROOT, "paper/figures")
os.makedirs(OUT, exist_ok=True)

# ── Modern, restrained style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 200,
})
INK = "#222222"
C = {  # controller palette
    "rocking":        "#9AA0A6",
    "constant_drive": "#E8833A",
    "policy_seed1":   "#1F6FB2",
    "policy_seed3":   "#5BA4D6",
}
LBL = {
    "rocking": "Rocking (scripted)",
    "constant_drive": "Constant drive (open-loop)",
    "policy_seed1": "Learned policy — seed 1",
    "policy_seed3": "Learned policy — seed 3",
}
ORDER = ["rocking", "constant_drive", "policy_seed1", "policy_seed3"]


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


def _load(tag):
    p = os.path.join(SWEEP, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


# ── 1. Modern escape comparison (horizontal lollipop + CI) ──────────────────
def fig_escape_comparison():
    data = {t: _load(t) for t in ORDER}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ys = np.arange(len(ORDER))[::-1]
    for y, tag in zip(ys, ORDER):
        d = data[tag]
        if d is None:
            continue
        r = 100 * d["overall_escape_rate"]
        lo, hi = [100 * x for x in d["overall_ci"]]
        col = C[tag]
        ax.hlines(y, 0, r, color=col, lw=3.2, alpha=0.55, zorder=2)
        ax.plot([lo, hi], [y, y], color=col, lw=1.4, alpha=0.9, zorder=3)
        ax.plot([lo, lo], [y - 0.08, y + 0.08], color=col, lw=1.4, zorder=3)
        ax.plot([hi, hi], [y - 0.08, y + 0.08], color=col, lw=1.4, zorder=3)
        ax.scatter([r], [y], s=120, color=col, zorder=4, edgecolor="white", linewidth=1.3)
        ax.text(r + 3.5, y + 0.18, f"{r:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([LBL[t] for t in ORDER], fontsize=10.5)
    ax.set_xlim(-2, 112)
    ax.set_xlabel("Escape rate (%)  —  120 trials/controller, bootstrap 95% CI", fontsize=10)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", visible=False)
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "escape_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote escape_comparison.png")


# ── 2. Modern action behaviour (grouped horizontal bars) ────────────────────
def fig_behavior():
    metrics = [("Drive full-forward", "drive_fwd"),
               ("Steer full-lock", "steer_lock"),
               ("Drive reverse", "reverse"),
               ("Rocking (sign flips)", "rock")]
    vals = {}
    for tag in ("policy_seed1", "policy_seed3"):
        d = _load(tag)
        if not d or "action_trace" not in d:
            continue
        A = np.array([t["action"] for t in d["action_trace"]])
        drive = np.clip(A[:, :6], -1, 1)
        steer = np.clip(A[:, 6:10], -1, 1)
        md = drive.mean(1)
        vals[tag] = {
            "drive_fwd": 100 * (drive > 0.95).mean(),
            "steer_lock": 100 * (np.abs(steer) > 0.95).mean(),
            "reverse": 100 * (drive < -0.05).mean(),
            "rock": 100 * (np.abs(np.diff(np.sign(md))) > 0).mean(),
        }
    if not vals:
        print("no action traces; skip behavior"); return
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(metrics))[::-1]
    h = 0.36
    for i, tag in enumerate(("policy_seed1", "policy_seed3")):
        if tag not in vals:
            continue
        xs = [vals[tag][k] for _, k in metrics]
        off = (0.5 - i) * h
        bars = ax.barh(y + off, xs, height=h, color=C[tag],
                       label=LBL[tag].replace("Learned policy — ", "Policy "),
                       edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, xs):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                    va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in metrics], fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fraction of recovery steps (%)", fontsize=10)
    ax.grid(axis="y", visible=False)
    _despine(ax, keep=("bottom",))
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Recovery is steering-dominated, not rocking", fontsize=11.5,
                 fontweight="bold", color=INK, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "policy_action_behavior.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote policy_action_behavior.png")


# ── 2b. Modern escape-vs-sinkage depth curve (lines + shaded CI bands) ──────
def fig_escape_vs_sinkage():
    data = {t: _load(t) for t in ORDER}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for tag in ORDER:
        d = data[tag]
        if d is None:
            continue
        levels = sorted(d["per_level"].keys(), key=float)
        xs = np.array([float(k) * 100 for k in levels])
        ys = np.array([100 * d["per_level"][k]["escape_rate"] for k in levels])
        lo = np.array([100 * d["per_level"][k]["ci"][0] for k in levels])
        hi = np.array([100 * d["per_level"][k]["ci"][1] for k in levels])
        col = C[tag]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, zorder=1, linewidth=0)
        ax.plot(xs, ys, color=col, lw=2.2, zorder=3,
                marker="o", markersize=6, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.8)
        # direct end-label; nudge the two near-overlapping policy labels apart
        lab = LBL[tag].replace("Learned policy — ", "Policy ").replace(" (open-loop)", "").replace(" (scripted)", "")
        dy = {"policy_seed1": 9, "policy_seed3": -9}.get(tag, 0)
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(8, dy),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9.5, color=col, fontweight="bold")
    # band annotations
    ax.text(15.4, 100, "policy band", color=C["policy_seed1"], fontsize=9,
            va="top", ha="left", style="italic", alpha=0.8)
    ax.text(15.4, 20, "naive band", color=C["constant_drive"], fontsize=9,
            va="bottom", ha="left", style="italic", alpha=0.8)
    ax.set_xlabel("Initial sinkage depth (cm)", fontsize=10.5)
    ax.set_ylabel("Escape rate (%)", fontsize=10.5)
    ax.set_xlim(14.3, 31.5)
    ax.set_ylim(-5, 108)
    ax.set_xticks([15, 20, 25, 28])
    ax.set_yticks([0, 25, 50, 75, 100])
    _despine(ax)
    ax.set_title("Escape rate vs entrapment severity", fontsize=11.5,
                 fontweight="bold", color=INK, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "escape_vs_sinkage.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote escape_vs_sinkage.png")


# ── 3. Reward convergence (seed 1 + 3, clipped + EMA + isotonic) ────────────
def _tb_scalar(seed, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = glob.glob(os.path.join(ROOT, f"experiments/regolith_recovery/{seed}/{seed}/events.out.tfevents*"))
    ea = EventAccumulator(f[0], size_guidance={"scalars": 0}); ea.Reload()
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])


def _ema(x, alpha=0.02):
    out = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc; out[i] = acc
    return out


def fig_reward_convergence():
    from sklearn.isotonic import IsotonicRegression
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    seeds = [("seed_1", "policy_seed1", "Seed 1"), ("seed_3", "policy_seed3", "Seed 3")]
    CLIP = (-60, 60)  # remove MPM physics-explosion outliers
    for seed, ckey, name in seeds:
        step, val = _tb_scalar(seed, "Reward / Total reward (mean)")
        v = np.clip(val, *CLIP)
        ema = _ema(v, alpha=0.02)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(step, ema)
        col = C[ckey]
        ax.plot(step / 1000, v, color=col, lw=0.5, alpha=0.18, zorder=1)
        ax.plot(step / 1000, ema, color=col, lw=1.4, alpha=0.85, zorder=3,
                label=f"{name} (EMA)")
        ax.plot(step / 1000, iso, color=col, lw=2.6, alpha=1.0, zorder=4, ls=(0, (1, 0)))
        ax.annotate(f"{iso[-1]:+.1f}", (step[-1] / 1000, iso[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=10, fontweight="bold", color=col)
        ax.annotate(f"{iso[0]:+.1f}", (step[0] / 1000, iso[0]),
                    xytext=(-2, -10), textcoords="offset points", va="center",
                    ha="right", fontsize=9, color=col)
    ax.axhline(0, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Training step (×10³)", fontsize=10.5)
    ax.set_ylabel("Episode return (mean)", fontsize=10.5)
    ax.set_xlim(0, 300)
    ax.set_ylim(-40, 45)
    _despine(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Reward convergence (thick = isotonic monotone envelope)",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reward_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote reward_convergence.png")


# ── 4. All-seeds escape-rate overlay (honest training-variance figure) ───────
def fig_seed_variance():
    # style per seed: (color, linestyle, linewidth, label)
    SEEDS = {
        1: ("#1F6FB2", "-",  2.4, "Seed 1 — converged (89%, curric. 0.83)"),
        3: ("#2E8B57", "-",  2.4, "Seed 3 — converged (84%, curric. 0.94)"),
        2: ("#5BA4D6", "-",  1.8, "Seed 2 — competent, easier curric. 0.55"),
        0: ("#C44E52", "--", 1.8, "Seed 0 — collapsed (69% → 10%)"),
        4: ("#9AA0A6", ":",  1.8, "Seed 4 — never bootstrapped"),
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for s, (col, ls, lw, lab) in SEEDS.items():
        try:
            step, val = _tb_scalar(f"seed_{s}", "Info / escape_rate")
        except Exception:
            continue
        ema = _ema(100 * val, alpha=0.04)
        ax.plot(step / 1000, ema, color=col, ls=ls, lw=lw, label=lab,
                zorder=4 if ls == "-" else 3, alpha=0.95)
    # competence-gate line + failure annotations
    ax.axhline(50, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
    ax.text(2, 52, "competence gate (50%)", fontsize=8, color="#888888", va="bottom")
    ax.annotate("collapse", (160, 30), (110, 12), fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    ax.annotate("no bootstrap", (250, 3), (150, 28), fontsize=9, color="#7A7A7A",
                arrowprops=dict(arrowstyle="->", color="#9AA0A6", lw=1.2))
    ax.set_xlabel("Training step (×10³)", fontsize=10.5)
    ax.set_ylabel("Escape rate (%)", fontsize=10.5)
    ax.set_xlim(0, 300)
    ax.set_ylim(-3, 103)
    _despine(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="center right", handlelength=2.4)
    ax.set_title("Training variance across 5 seeds: 3 competent, 2 failed",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seed_variance.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote seed_variance.png")


# ── 5. REAL omnidirectional top-down (true terminal positions) ──────────────
def fig_topdown():
    import csv
    TD = os.path.join(ROOT, "experiments/escape_eval/topdown")
    pts = {"esc": [], "fail": []}
    nseed = 0
    for s in (1, 3):
        p = os.path.join(TD, f"topdown_seed{s}.csv")
        if not os.path.exists(p):
            continue
        nseed += 1
        for r in csv.DictReader(open(p)):
            try:
                x, y = float(r["rel_x"]), float(r["rel_y"])
                esc = int(r["escaped"]); blow = int(r.get("blowup", 0))
            except (KeyError, ValueError):
                continue
            if blow or not (abs(x) < 6 and abs(y) < 6):
                continue
            pts["esc" if esc else "fail"].append((x, y))
    if not pts["esc"]:
        print("topdown CSVs not ready; skipping topdown fig"); return
    esc = np.array(pts["esc"]); fail = np.array(pts["fail"]) if pts["fail"] else np.empty((0, 2))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    # sand-bed boundary (1.75 m half-side) and escape circle (3.0 m)
    ax.add_patch(plt.Rectangle((-1.75, -1.75), 3.5, 3.5, fill=False,
                               edgecolor="#C8A24B", lw=1.0, ls=":", zorder=2))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3.0 * np.cos(th), 3.0 * np.sin(th), color="#C8A24B", lw=1.6,
            ls="--", zorder=2)
    ax.text(0, 3.18, "escape threshold (3.0 m)", color="#9A7A20", fontsize=8.5,
            ha="center", va="bottom")
    if len(fail):
        ax.scatter(fail[:, 0], fail[:, 1], s=14, color="#C44E52", alpha=0.28,
                   edgecolor="none", zorder=3, label=f"failed (n={len(fail)})")
    ax.scatter(esc[:, 0], esc[:, 1], s=16, color="#1F6FB2", alpha=0.55,
               edgecolor="none", zorder=4, label=f"escaped (n={len(esc)})")
    ax.scatter([0], [0], marker="+", s=120, color="black", lw=1.6, zorder=5)
    ax.text(0.15, -0.05, "spawn", fontsize=8.5, va="top")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x from spawn (m)", fontsize=10)
    ax.set_ylabel("y from spawn (m)", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Omnidirectional escape — true terminal positions\n"
                 f"(seeds 1+3, {nseed*300} trials, headings ∼U(0,2π))",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "topdown_escape_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote topdown_escape_heatmap.png  (esc={len(esc)}, fail={len(fail)})")


if __name__ == "__main__":
    fig_escape_comparison()
    fig_escape_vs_sinkage()
    fig_behavior()
    fig_reward_convergence()
    fig_seed_variance()
    fig_topdown()
