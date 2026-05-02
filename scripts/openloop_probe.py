"""Open-loop physics probe — bypasses the policy.

Phase A (5 s): zero action. Watch chassis Z. Sinking past spawn => bed too soft.
Phase B (10 s): constant +0.3 drive, no steer. Hard-coded forward. If rover does
not translate > 0.3 m, fluidization is real and the issue is contact, not RL.

Run:
    ./launch.sh scripts/openloop_probe.py --num_envs 1
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--no-mpm", action="store_true")
parser.add_argument("--drive", type=float, default=0.3,
                    help="Phase B drive command (policy units, [-1,1])")
parser.add_argument("--phase_a", type=float, default=5.0)
parser.add_argument("--phase_b", type=float, default=10.0)
parser.add_argument("--sinkage", type=float, default=None,
                    help="Override sinkage depth in meters (bypasses DR sampling)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

os.environ["LAUNCH_OV_APP"] = "1"
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # noqa: F401
from envs.entrapment_env import EntrapmentEnvCfg


def main():
    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.no_mpm:
        cfg.skip_mpm = True
    if args_cli.sinkage is not None:
        cfg.dr_sinkage_range = (args_cli.sinkage, args_cli.sinkage)
        print(f"[probe] sinkage forced to {args_cli.sinkage:.3f} m")

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    num_act = cfg.action_space

    dt = float(cfg.sim.dt) * float(cfg.decimation)
    steps_a = int(args_cli.phase_a / dt)
    steps_b = int(args_cli.phase_b / dt)

    obs, _ = env.reset()
    origin = unwrapped.scene.env_origins[0].cpu().numpy()

    prev = {"p": None, "t": 0.0}
    drive_ids = getattr(unwrapped, "_drive_ids", None)
    if drive_ids is None:
        # try common alternates
        for attr in ("drive_ids", "_drive_joint_ids"):
            if hasattr(unwrapped, attr):
                drive_ids = getattr(unwrapped, attr)
                break
    print(f"[probe] drive_ids = {drive_ids}")

    def report(label, step):
        rp = unwrapped.root_pos[0].cpu().numpy() - origin
        t_now = step * dt
        if prev["p"] is None:
            vx = vz = 0.0
        else:
            vx = (rp[0] - prev["p"][0]) / max(t_now - prev["t"], 1e-6)
            vz = (rp[2] - prev["p"][2]) / max(t_now - prev["t"], 1e-6)
        prev["p"], prev["t"] = rp.copy(), t_now
        slip = unwrapped._slip[0].cpu().numpy() if hasattr(unwrapped, "_slip") else None
        slip_str = f" |slip|={float(abs(slip).mean()):.2f}" if slip is not None else ""
        wheel_str = ""
        if drive_ids is not None:
            try:
                jv_all = unwrapped.robot.data.joint_vel
                if hasattr(jv_all, "numpy") and not hasattr(jv_all, "cpu"):
                    import warp as wp
                    jv_t = wp.to_torch(jv_all)
                else:
                    jv_t = jv_all
                ids = drive_ids if isinstance(drive_ids, list) else list(drive_ids)
                jv = jv_t[0, ids].detach().cpu().numpy()
                wheel_str = (f" w=[{jv.min():+.2f}..{jv.max():+.2f}] "
                             f"mean={jv.mean():+.2f}")
            except Exception as e:
                wheel_str = f" w_err={type(e).__name__}"
        print(f"[{label} t={t_now:5.2f}s] "
              f"x={rp[0]:+.3f} y={rp[1]:+.3f} z={rp[2]:+.3f} "
              f"vx={vx:+.3f} vz={vz:+.3f}{slip_str}{wheel_str}")

    print("\n=== Phase A: zero action — testing static support ===")
    z0 = float(unwrapped.root_pos[0, 2].item() - origin[2])
    print(f"spawn z = {z0:+.3f}")
    a = torch.zeros(args_cli.num_envs, num_act, device=device)
    for s in range(steps_a):
        obs, _, term, trunc, _ = env.step(a)
        if s % max(1, int(0.5 / dt)) == 0:
            report("A", s)
        if (term | trunc).any():
            print("  episode terminated during phase A")
            obs, _ = env.reset()
            origin = unwrapped.scene.env_origins[0].cpu().numpy()
    z_after_a = float(unwrapped.root_pos[0, 2].item() - origin[2])
    print(f"phase A Δz = {z_after_a - z0:+.3f} m  (negative = sank)")

    print("\n=== Phase B: constant forward drive — testing traction ===")
    x_start = float(unwrapped.root_pos[0, 0].item() - origin[0])
    y_start = float(unwrapped.root_pos[0, 1].item() - origin[1])
    a = torch.zeros(args_cli.num_envs, num_act, device=device)
    a[:, :6] = args_cli.drive
    for s in range(steps_b):
        obs, _, term, trunc, _ = env.step(a)
        if s % max(1, int(0.5 / dt)) == 0:
            report("B", s)
        if (term | trunc).any():
            print("  episode terminated during phase B")
            break
    x_end = float(unwrapped.root_pos[0, 0].item() - origin[0])
    y_end = float(unwrapped.root_pos[0, 1].item() - origin[1])
    dx, dy = x_end - x_start, y_end - y_start
    dist = math.hypot(dx, dy)
    print(f"\nphase B displacement: dx={dx:+.3f} dy={dy:+.3f} "
          f"dist={dist:.3f} m in {args_cli.phase_b}s "
          f"=> avg speed {dist/args_cli.phase_b:.3f} m/s")
    print("\nVerdict:")
    if dist > 0.3:
        print("  PHYSICS OK — rover moves with hardcoded action. Issue is RL/policy side.")
    else:
        print("  FLUIDIZATION CONFIRMED — physics cannot produce motion at this drive.")
        print("  Next levers: lower effort_limit_sim, raise wheel mu, or switch to mesh wheels.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
