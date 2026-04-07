"""
Episode Dashboard — publication-quality multi-panel figure.
Inspired by Bi & Ding (2026) Fig. 18/19/23/25.

Generates:
  Panel 1 : Drive velocity  — 6 wheels, L/R differentiated
  Panel 2 : Drive torque    — 6 wheels + zoom-inset on anomaly window
  Panel 3 : Slip ratio      — 6 wheels + shaded min-max band
  Panel 4 : Entrapment & anomaly flags (stacked binary)
  Panel 5 : IMU acceleration (3-axis)
  Panel 6 : Cumulative reward
  Panel 7 : Rover XY trajectory on MPM sand surface height-map
             with auto-annotated behaviour events

Usage:
    # --- Requires Isaac Sim + Newton ---

    # Random actions (sanity / pipeline check):
    ./launch.sh scripts/plot_episode.py --num_envs 1

    # Trained checkpoint:
    ./launch.sh scripts/plot_episode.py --num_envs 1 \\
        --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

    # Collect N episodes and overlay trajectories:
    ./launch.sh scripts/plot_episode.py --num_envs 1 --episodes 3 \\
        --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

    # --- Offline: no Isaac Sim needed ---
    # First save data once with eval.py:
    ./launch.sh scripts/eval.py --episodes 3 --checkpoint <ckpt> \\
        --save-data experiments/regolith_recovery/episode_data/run1.npz

    # Then plot anywhere, any time:
    python3 scripts/plot_episode.py --from-file experiments/regolith_recovery/episode_data/run1.npz
"""

import argparse
import os
import sys

# ── Pre-parse: detect offline (--from-file) mode BEFORE loading the Isaac Sim stack ──
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--from-file", type=str, default=None)
_pre_ns, _ = _pre.parse_known_args()
_OFFLINE = (_pre_ns.from_file is not None)

if not _OFFLINE:
    import math
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Episode dashboard plotter")
    parser.add_argument("--num_envs",   type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes",   type=int, default=1,
                        help="Number of episodes to collect (trajectories overlaid on map)")
    parser.add_argument("--max_steps",  type=int, default=500,
                        help="Max steps per episode (safety cap)")
    parser.add_argument("--from-file",  type=str, default=None,
                        help="Load saved .npz instead of running simulation")
    AppLauncher.add_app_launcher_args(parser)

    args_cli, hydra_args = parser.parse_known_args()
    os.environ["LAUNCH_OV_APP"] = "1"
    args_cli.headless = True
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher   = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    parser = argparse.ArgumentParser(description="Episode dashboard plotter (offline mode)")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Load saved .npz (created by eval.py --save-data)")
    args_cli, _ = parser.parse_known_args()
    simulation_app = None

# ── Imports (numpy/matplotlib always; sim stack only in online mode) ───────────
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

if not _OFFLINE:
    import torch
    import gymnasium as gym
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import envs
    from envs.entrapment_env import EntrapmentEnvCfg

PLOTS_DIR = os.path.join(REPO_ROOT, "experiments", "regolith_recovery", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#DDDDDD",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#BBBBBB",
    "lines.linewidth":   1.4,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# Wheel colours — distinct, print-safe
WHEEL_NAMES  = ["FL", "FR", "ML", "MR", "RL", "RR"]
WHEEL_COLORS = ["#2166AC", "#D6604D", "#4DAC26", "#E08214", "#8073AC", "#1A9641"]
LEFT_IDX     = [0, 2, 4]    # FL, ML, RL
RIGHT_IDX    = [1, 3, 5]    # FR, MR, RR

# Regolith geometry (must match entrapment_env.py)
SAND_HALF_X = 0.6
SAND_HALF_Y = 0.6
ESCAPE_DIST = 1.5   # m

# Mars sand colormap: light dusty beige → deep rust red
MARS_CMAP = LinearSegmentedColormap.from_list(
    "mars_sand", ["#F4DDB4", "#C9945A", "#8B3A1E"]
)


# ── MPM heightmap snapshot ─────────────────────────────────────────────────────

def _snap_heightmap(raw_env, env_idx=0, grid_res=60):
    """
    Bin MPM particle positions into a 2D surface height map (env-local coords).
    Returns (heightmap [grid_res × grid_res], extent_tuple) or (None, None).
    """
    # TEMPORARILY DISABLE HEIGHTMAP EXTRACTION TO AVOID CUDA MEMORY ISSUES
    return None, None


# ── Behaviour event detection ──────────────────────────────────────────────────

def _detect_events(t, pos_xy, entrap_flag, torque_anomaly, wheel_vel):
    """Return list of (time, label, xy) for key trajectory events."""
    events = []

    # Entrapment onset
    for i in range(1, len(entrap_flag)):
        if entrap_flag[i] > 0.5 and entrap_flag[i - 1] < 0.5:
            events.append((t[i], "Wheels\ntrapped", pos_xy[i].copy()))
            break

    # Torque anomaly onset
    for i in range(1, len(torque_anomaly)):
        if torque_anomaly[i] > 0.5 and torque_anomaly[i - 1] < 0.5:
            events.append((t[i], "Torque\nanomaly", pos_xy[i].copy()))
            break

    # First rocking reversal (sign flip while trapped)
    mean_vel = wheel_vel.mean(axis=-1)
    for i in range(1, len(mean_vel)):
        if (mean_vel[i] * mean_vel[i - 1] < -0.3
                and abs(mean_vel[i]) > 0.1
                and entrap_flag[i] > 0.5):
            events.append((t[i], "Rocking\nreversal", pos_xy[i].copy()))
            break

    # Escape
    dist = np.linalg.norm(pos_xy, axis=-1)
    for i in range(len(dist)):
        if dist[i] > ESCAPE_DIST:
            events.append((t[i], "Escaped\n(>1.5 m)", pos_xy[i].copy()))
            break

    return events


# ── Agent loader ───────────────────────────────────────────────────────────────

def load_agent(device, num_obs, num_act, num_envs, ckpt_path):
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from scripts.train import PolicyNet, ValueNet

    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0,      high=1.0,      shape=(num_act,))
    models    = {
        "policy": PolicyNet(obs_space, act_space, device),
        "value":  ValueNet(obs_space, act_space, device),
    }
    memory = RandomMemory(memory_size=1, num_envs=num_envs, device=device)
    agent  = PPO(models=models, memory=memory, cfg=PPO_DEFAULT_CONFIG.copy(),
                 observation_space=obs_space, action_space=act_space, device=device)
    agent.load(ckpt_path)
    agent.set_running_mode("eval")
    return agent


# ── Data collector ─────────────────────────────────────────────────────────────

def collect_episode(env, agent, max_steps):
    """Run one episode, collect per-step data + MPM heightmap snapshots."""
    obs, _  = env.reset()
    raw_env = env.unwrapped

    DRIVE_VEL_LIMIT     = 6.0
    TORQUE_LIMIT_APPROX = 3.0
    DT = float(raw_env.cfg.sim.dt) * float(raw_env.cfg.decimation)

    # env-local origin offset so trajectory is centred at (0, 0)
    env_origin_xy = raw_env.scene.env_origins[0].cpu().numpy()[:2]

    # Initial sand surface snapshot (before wheel interaction)
    hmap_init, hmap_extent = _snap_heightmap(raw_env)

    records = {k: [] for k in [
        "t", "wheel_vel", "drive_torque", "slip",
        "entrap_flag", "torque_anomaly", "imu_acc",
        "pos_xy", "reward", "action",
    ]}

    step, done_ep = 0, False
    while step < max_steps and not done_ep:
        if agent is not None:
            with torch.no_grad():
                action, _, _ = agent.act({"states": obs}, timestep=step, timesteps=0)
        else:
            action = torch.zeros(env.num_envs, env.unwrapped.cfg.action_space,
                                 device=env.device)
            action[:, :6] = 0.4 + 0.1 * torch.randn_like(action[:, :6])
            action = action.clamp(-1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        ob = obs[0].cpu().numpy()   # (29,)
        wheel_vel_norm = ob[0:6]
        slip           = ob[6:12]
        imu_acc        = ob[16:19]
        drive_torque_n = ob[20:26]
        entrap_flag    = ob[26]
        torque_anomaly = ob[27]
        # ob[28] = dist_norm (not needed here; pos_xy computed from sim state below)

        pos_w  = raw_env.root_pos[0].cpu().numpy()
        pos_xy = pos_w[:2] - env_origin_xy   # local coords

        records["t"].append(step * DT)
        records["wheel_vel"].append(wheel_vel_norm * DRIVE_VEL_LIMIT)
        records["drive_torque"].append(drive_torque_n * TORQUE_LIMIT_APPROX)
        records["slip"].append(slip)
        records["entrap_flag"].append(entrap_flag)
        records["torque_anomaly"].append(torque_anomaly)
        records["imu_acc"].append(imu_acc)
        records["pos_xy"].append(pos_xy)
        records["reward"].append(float(reward[0]))
        records["action"].append(action[0].cpu().numpy())

        _term  = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
        _trunc = truncated.squeeze(-1)  if truncated.dim()  > 1 else truncated
        if bool(_term[0]) or bool(_trunc[0]):
            done_ep = True
        step += 1

    out = {k: np.array(v) for k, v in records.items()}

    # Final sand surface snapshot (shows wheel tracks / deformation)
    hmap_final, _ = _snap_heightmap(raw_env)
    out["heightmap_initial"] = hmap_init
    out["heightmap_final"]   = hmap_final
    out["heightmap_extent"]  = hmap_extent
    return out


# ── Plotting helpers ───────────────────────────────────────────────────────────

def annotate_anomaly_spans(ax, t, flag, color="#FFD7D7", label="Entrap"):
    in_span, t0 = False, 0.0
    for ti, fi in zip(t, flag):
        if fi > 0.5 and not in_span:
            t0, in_span = ti, True
        elif fi <= 0.5 and in_span:
            ax.axvspan(t0, ti, color=color, alpha=0.25, linewidth=0, label=label)
            in_span, label = False, "_"
    if in_span:
        ax.axvspan(t0, t[-1], color=color, alpha=0.25, linewidth=0)


def add_torque_inset(ax, t, torque, torque_anomaly):
    """Zoom-inset on anomaly window — paper Fig.23/25 style."""
    anomaly_t = t[torque_anomaly > 0.5]
    if len(anomaly_t) < 3:
        return
    t_s = max(t[0],  anomaly_t[0]  - 1.0)
    t_e = min(t[-1], anomaly_t[-1] + 1.0)
    win = (t >= t_s) & (t <= t_e)
    if win.sum() < 3:
        return

    axins = inset_axes(ax, width="42%", height="48%", loc="lower right", borderpad=0.8)
    for i, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        axins.plot(t, torque[:, i], color=color, linewidth=0.9,
                   linestyle="-" if i in LEFT_IDX else "--")
    axins.set_xlim(t_s, t_e)
    tw = torque[win]
    margin = 0.05 * max(tw.max() - tw.min(), 1e-3)
    axins.set_ylim(tw.min() - margin, tw.max() + margin)
    axins.set_xticks([round(t_s, 1), round((t_s + t_e) / 2, 1), round(t_e, 1)])
    axins.tick_params(labelsize=6, pad=1)
    axins.set_facecolor("#FFF5EE")
    for sp in axins.spines.values():
        sp.set_edgecolor("#CC6633")
        sp.set_linewidth(1.0)
    axins.set_title("Anomaly detail", fontsize=6.5, color="#CC6633", pad=2)
    axins.axhline(0, color="#888", linewidth=0.4, linestyle=":")
    try:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
                   ec="#CC6633", lw=0.8, alpha=0.7)
    except Exception:
        pass


# ── Main figure ────────────────────────────────────────────────────────────────

def make_episode_figure(episodes_data, out_path, mode="random"):
    ep = episodes_data[0]
    t  = ep["t"]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Episode Dashboard  —  Mars Rover Regolith Escape  ({mode} policy)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 2, figure=fig,
                  hspace=0.52, wspace=0.30,
                  height_ratios=[1, 1, 1, 1.5])

    axes = [
        fig.add_subplot(gs[0, 0]),   # Drive velocity
        fig.add_subplot(gs[0, 1]),   # Drive torque + inset
        fig.add_subplot(gs[1, 0]),   # Slip ratio
        fig.add_subplot(gs[1, 1]),   # Entrap / anomaly flags
        fig.add_subplot(gs[2, 0]),   # IMU acceleration
        fig.add_subplot(gs[2, 1]),   # Cumulative reward
        fig.add_subplot(gs[3, :]),   # Trajectory map (full width)
    ]

    # ── Panel 1 : Drive Velocity ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Drive Velocity")
    ax.set_ylabel("rad / s")
    ax.set_xlabel("t / s")
    annotate_anomaly_spans(ax, t, ep["entrap_flag"])
    for i, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        ax.plot(t, ep["wheel_vel"][:, i], color=color, linewidth=1.2,
                linestyle="-" if i in LEFT_IDX else "--", label=name)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.legend(ncols=3, loc="upper right", handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2 : Drive Torque  +  zoom inset ────────────────────────────────
    ax = axes[1]
    ax.set_title("Drive Torque")
    ax.set_ylabel("Torque / N·m")
    ax.set_xlabel("t / s")
    annotate_anomaly_spans(ax, t, ep["torque_anomaly"],
                           color="#FFC8A0", label="Torque anomaly")
    for i, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        ax.plot(t, ep["drive_torque"][:, i], color=color, linewidth=1.2,
                linestyle="-" if i in LEFT_IDX else "--", label=name)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.legend(ncols=3, loc="upper right", handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    add_torque_inset(ax, t, ep["drive_torque"], ep["torque_anomaly"])

    # ── Panel 3 : Slip Ratio ──────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("Slip Ratio")
    ax.set_ylabel("Slip")
    ax.set_xlabel("t / s")
    annotate_anomaly_spans(ax, t, ep["entrap_flag"])
    slip_mean = ep["slip"].mean(axis=1)
    ax.fill_between(t, ep["slip"].min(axis=1), ep["slip"].max(axis=1),
                    color="#2166AC", alpha=0.15, label="min–max band")
    ax.plot(t, slip_mean, color="#2166AC", linewidth=1.8, label="mean slip")
    for i, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        ax.plot(t, ep["slip"][:, i], color=color, linewidth=0.6, alpha=0.5)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 4 : Entrapment & Anomaly Flags ─────────────────────────────────
    ax = axes[3]
    ax.set_title("Entrapment & Anomaly Status")
    ax.set_xlabel("t / s")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Active"])
    ax.set_ylim(-0.15, 1.3)
    ax.fill_between(t, 0, ep["entrap_flag"],
                    step="post", color="#D6604D", alpha=0.5, label="Entrap flag")
    ax.step(t, ep["entrap_flag"],  color="#D6604D", linewidth=1.5, where="post")
    shifted = ep["torque_anomaly"] * 0.95
    ax.fill_between(t, 0, shifted,
                    step="post", color="#E08214", alpha=0.4, label="Torque anomaly")
    ax.step(t, shifted, color="#E08214", linewidth=1.2, linestyle="--", where="post")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 5 : IMU Acceleration ────────────────────────────────────────────
    ax = axes[4]
    ax.set_title("IMU Acceleration (normalised by g)")
    ax.set_ylabel("acc / g")
    ax.set_xlabel("t / s")
    for lbl, col in zip(["acc_x", "acc_y", "acc_z"],
                         ["#2166AC", "#D6604D", "#4DAC26"]):
        idx = ["acc_x", "acc_y", "acc_z"].index(lbl)
        ax.plot(t, ep["imu_acc"][:, idx], color=col, linewidth=1.2, label=lbl)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 6 : Cumulative Reward ───────────────────────────────────────────
    ax = axes[5]
    ax.set_title("Cumulative Reward")
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("t / s")
    cum = np.cumsum(ep["reward"])
    ax.plot(t, cum, color="#2166AC", linewidth=1.8)
    ax.fill_between(t, 0, cum, where=(cum >= 0), color="#4DAC26", alpha=0.12)
    ax.fill_between(t, 0, cum, where=(cum < 0),  color="#D6604D", alpha=0.12)
    ax.axhline(0, color="#888", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 7 : Trajectory Map ──────────────────────────────────────────────
    ax = axes[6]
    ax.set_title("Rover Trajectory  (colour = elapsed time)", pad=8)
    ax.set_xlabel("X / m  (env-local)")
    ax.set_ylabel("Y / m  (env-local)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)

    # Background: MPM sand surface height-map (shows real physics deformation)
    hmap  = ep.get("heightmap_final")
    hext  = ep.get("heightmap_extent")
    hmap0 = ep.get("heightmap_initial")

    if hmap is not None and hext is not None:
        im = ax.imshow(
            hmap.T,                # transpose: imshow is row=y, col=x
            origin="lower",
            extent=hext,
            cmap=MARS_CMAP,
            alpha=0.65,
            zorder=0,
            interpolation="bilinear",
        )
        cbar_h = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.01,
                              location="left", fraction=0.03)
        cbar_h.set_label("Surface z / m", fontsize=7)
        cbar_h.ax.tick_params(labelsize=6)

        # Deformation contour: wheel tracks (most sunk regions)
        if hmap0 is not None:
            deform = hmap - hmap0
            xc = np.linspace(hext[0], hext[1], hmap.shape[0])
            yc = np.linspace(hext[2], hext[3], hmap.shape[1])
            try:
                ax.contour(xc, yc, deform.T,
                           levels=[-0.02, -0.01],
                           colors=["#4A90D9", "#2255AA"],
                           linewidths=[0.7, 0.5], linestyles="--",
                           alpha=0.6, zorder=1)
            except Exception:
                pass
    else:
        # Fallback: plain sand rectangle
        ax.add_patch(mpatches.Rectangle(
            (-SAND_HALF_X, -SAND_HALF_Y), 2 * SAND_HALF_X, 2 * SAND_HALF_Y,
            facecolor="#C8A882", edgecolor="#9E7B4E", linewidth=1.0,
            alpha=0.35, label="Sand bed (1.2×1.2 m)", zorder=0,
        ))

    # Sand bed boundary
    ax.add_patch(mpatches.Rectangle(
        (-SAND_HALF_X, -SAND_HALF_Y), 2 * SAND_HALF_X, 2 * SAND_HALF_Y,
        facecolor="none", edgecolor="#9E7B4E", linewidth=1.2,
        label="Sand bed (1.2×1.2 m)", zorder=2,
    ))

    # Escape distance circle
    ax.add_patch(plt.Circle(
        (0, 0), ESCAPE_DIST, fill=False,
        edgecolor="#D6604D", linewidth=1.2,
        linestyle="--", label=f"Escape threshold ({ESCAPE_DIST} m)", zorder=2,
    ))

    # Trajectories (colour = elapsed time, plasma)
    cmap_traj = matplotlib.colormaps["plasma"]
    for ei, ep_i in enumerate(episodes_data):
        xy   = ep_i["pos_xy"]
        t_i  = ep_i["t"]
        norm = Normalize(vmin=t_i[0], vmax=t_i[-1])
        pts  = xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap_traj, norm=norm,
                              linewidth=2.5, zorder=4, alpha=0.92)
        lc.set_array(t_i[:-1])
        ax.add_collection(lc)
        ax.plot(*xy[0],  "o", color="#00B4D8", markersize=9,
                markeredgecolor="white", markeredgewidth=1.0,
                zorder=5, label="Start" if ei == 0 else "_")
        ax.plot(*xy[-1], "*", color="#FFDD57", markersize=11,
                markeredgecolor="#888800", markeredgewidth=0.8,
                zorder=5, label="End" if ei == 0 else "_")
        trapped = ep_i["entrap_flag"] > 0.5
        if trapped.any():
            ax.scatter(xy[trapped, 0], xy[trapped, 1],
                       c="#D6604D", s=14, zorder=3, alpha=0.45,
                       label="Trapped" if ei == 0 else "_")

    # Time colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_traj,
                               norm=Normalize(vmin=0, vmax=ep["t"][-1]))
    sm.set_array([])
    cbar_t = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar_t.set_label("t / s", fontsize=8)
    cbar_t.ax.tick_params(labelsize=7)

    # Origin cross
    ax.axhline(0, color="#AAAAAA", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#AAAAAA", linewidth=0.5, linestyle=":", zorder=1)

    # ── Behaviour event annotations ───────────────────────────────────────────
    events = _detect_events(ep["t"], ep["pos_xy"],
                            ep["entrap_flag"], ep["torque_anomaly"],
                            ep["wheel_vel"])
    _ANNOT_COLORS = {
        "Wheels\ntrapped":   "#D6604D",
        "Torque\nanomaly":   "#E08214",
        "Rocking\nreversal": "#8073AC",
        "Escaped\n(>1.5 m)": "#1A9641",
    }
    _OFFSETS = [(0.35, 0.35), (-0.40, 0.35), (-0.40, -0.35), (0.35, -0.35)]
    for k, (te, label, xy_ev) in enumerate(events):
        dx, dy = _OFFSETS[k % len(_OFFSETS)]
        color  = _ANNOT_COLORS.get(label, "#333333")
        ax.annotate(
            label,
            xy=xy_ev, xycoords="data",
            xytext=(xy_ev[0] + dx, xy_ev[1] + dy),
            fontsize=7.5, color=color, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.88, linewidth=0.8),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.0, mutation_scale=8),
            zorder=6,
        )

    # Auto-scale with margin
    all_xy = np.concatenate([e["pos_xy"] for e in episodes_data], axis=0)
    margin = 0.35
    ax.set_xlim(min(all_xy[:, 0].min() - margin, -SAND_HALF_X - 0.1),
                max(all_xy[:, 0].max() + margin,  SAND_HALF_X + 0.1))
    ax.set_ylim(min(all_xy[:, 1].min() - margin, -SAND_HALF_Y - 0.1),
                max(all_xy[:, 1].max() + margin,  SAND_HALF_Y + 0.1))
    ax.legend(loc="upper left", fontsize=7.5, ncols=2)

    # ── Save dashboard ────────────────────────────────────────────────────────
    out_path = os.path.join(PLOTS_DIR, "episode_dashboard.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out_path}")

    # ── Bonus: standalone sand deformation figure ─────────────────────────────
    _save_deformation_map(ep, episodes_data)


def _save_deformation_map(ep, episodes_data):
    """Three-panel publication figure: initial / final / deformation."""
    hmap0 = ep.get("heightmap_initial")
    hmapf = ep.get("heightmap_final")
    hext  = ep.get("heightmap_extent")
    if hmap0 is None or hmapf is None:
        return

    deform = hmapf - hmap0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("MPM Regolith Surface — Initial / Final / Deformation",
                 fontsize=12, fontweight="bold")

    panels = [
        ("Initial surface",               MARS_CMAP, hmap0,  "z / m"),
        ("Final surface",                  MARS_CMAP, hmapf,  "z / m"),
        ("Deformation  (final − initial)", "RdBu_r",  deform, "Δz / m"),
    ]
    for ax, (title, cmap, d, clabel) in zip(axes, panels):
        im = ax.imshow(d.T, origin="lower", extent=hext,
                       cmap=cmap, interpolation="bilinear")
        ax.set_title(title, pad=6)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label=clabel)
        for ep_i in episodes_data:
            xy = ep_i["pos_xy"]
            ax.plot(xy[:, 0], xy[:, 1], color="white", linewidth=1.5,
                    alpha=0.75, zorder=3)
            ax.plot(*xy[0],  "o", color="#00B4D8", markersize=7, zorder=4)
            ax.plot(*xy[-1], "*", color="#FFDD57", markersize=8, zorder=4)
        ax.set_xlim(hext[0], hext[1])
        ax.set_ylim(hext[2], hext[3])

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "sand_deformation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out}")


# ── Offline loader ─────────────────────────────────────────────────────────────

_EP_KEYS = ["t", "wheel_vel", "drive_torque", "slip",
            "entrap_flag", "torque_anomaly", "imu_acc",
            "pos_xy", "reward", "action"]

def _load_from_npz(path):
    """Load episodes_data list from a .npz saved by eval.py --save-data."""
    data    = np.load(path, allow_pickle=True)
    n_eps   = int(data["n_episodes"][0])
    mode    = str(data["mode"][0])
    episodes = []
    for i in range(n_eps):
        ep = {}
        for k in _EP_KEYS:
            key = f"ep{i}_{k}"
            ep[k] = data[key] if key in data else np.array([])
        ep["heightmap_initial"] = None
        ep["heightmap_final"]   = None
        ep["heightmap_extent"]  = None
        episodes.append(ep)
    return episodes, mode


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Offline mode: plot from saved .npz without any simulation ────────────
    if getattr(args_cli, "from_file", None):
        episodes_data, mode = _load_from_npz(args_cli.from_file)
        print(f"\n{'='*55}")
        print(f"  Offline mode — loading: {args_cli.from_file}")
        print(f"  Episodes: {len(episodes_data)}  |  Mode: {mode}")
        for i, ep in enumerate(episodes_data):
            n_steps  = len(ep["t"])
            ep_rew   = float(ep["reward"].sum())
            dist_max = float(np.linalg.norm(ep["pos_xy"], axis=-1).max()) if len(ep["pos_xy"]) else 0.0
            escaped  = dist_max > ESCAPE_DIST
            print(f"  Episode {i+1}: steps={n_steps}  reward={ep_rew:.2f}  "
                  f"max_dist={dist_max:.2f}m  escaped={'YES' if escaped else 'no'}")
        print(f"{'='*55}\n")
        print("Generating episode dashboard ...")
        make_episode_figure(episodes_data, PLOTS_DIR, mode=mode)
        print(f"\nDone. Plots in: {PLOTS_DIR}\n")
        return
    # ─────────────────────────────────────────────────────────────────────────

    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env.device
    num_act = env_cfg.action_space
    num_obs = env_cfg.observation_space

    agent = None
    if args_cli.checkpoint:
        agent = load_agent(device, num_obs, num_act, env.num_envs, args_cli.checkpoint)
        mode  = "trained"
        print(f"[plot_episode] Loaded: {args_cli.checkpoint}")
    else:
        mode = "random"
        print("[plot_episode] No checkpoint — using random actions")

    print(f"\n{'='*55}")
    print(f"  Collecting {args_cli.episodes} episode(s) …")
    print(f"  Max steps / episode: {args_cli.max_steps}")
    print(f"{'='*55}\n")

    episodes_data = []
    for ep_i in range(args_cli.episodes):
        print(f"  Episode {ep_i + 1} / {args_cli.episodes} …", end=" ", flush=True)
        data = collect_episode(env, agent, args_cli.max_steps)
        episodes_data.append(data)
        n_steps   = len(data["t"])
        ep_reward = float(data["reward"].sum())
        trapped   = float(data["entrap_flag"].mean()) * 100
        dist_max  = float(np.linalg.norm(data["pos_xy"], axis=-1).max())
        escaped   = dist_max > ESCAPE_DIST
        hmap_ok   = data["heightmap_final"] is not None
        print(f"steps={n_steps}  reward={ep_reward:.2f}  "
              f"entrap={trapped:.0f}%  max_dist={dist_max:.2f}m  "
              f"escaped={'YES' if escaped else 'no'}  "
              f"heightmap={'OK' if hmap_ok else 'N/A (no MPM)'}")

    env.close()
    if simulation_app is not None:
        simulation_app.close()

    print("\nGenerating episode dashboard …")
    make_episode_figure(episodes_data, PLOTS_DIR, mode=mode)
    print(f"\nDone. Plots in: {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()
