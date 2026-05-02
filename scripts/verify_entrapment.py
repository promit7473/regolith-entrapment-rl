import argparse
import math
import os
import sys


import platform as _platform
_platform._sys_version_cache[sys.version] = (
    "CPython",
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "", "", "", "",
    sys.version.split("[", 1)[-1].rstrip("]") if "[" in sys.version else "",
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Entrapment realism check")
parser.add_argument("--num_envs",        type=int,   default=16)
parser.add_argument("--hold_seconds",    type=float, default=3.0,
                    help="Seconds of zero-action hold (default: 3.0)")
parser.add_argument("--bury_tolerance",  type=float, default=0.10,
                    help="Max passive displacement [m] to count as entrapped (default: 0.10)")
parser.add_argument("--episodes",        type=int,   default=3,
                    help="Reset + hold cycles to aggregate over (default: 3)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import warp as wp
import gymnasium as gym

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs
from envs.entrapment_env import EntrapmentEnvCfg, SPAWN_X_OFFSET


def main():
    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg, render_mode=None)
    unwrapped = env.unwrapped

    policy_hz    = 1.0 / (cfg.decimation * cfg.sim.dt)
    hold_steps   = int(args_cli.hold_seconds * policy_hz)
    device       = unwrapped.device
    zero_action  = torch.zeros(args_cli.num_envs, cfg.action_space, device=device)

    print(f"\n[Verify] {args_cli.num_envs} envs × {args_cli.episodes} episodes")
    print(f"[Verify] hold = {args_cli.hold_seconds}s ({hold_steps} policy steps)")
    print(f"[Verify] sinkage range = {cfg.dr_sinkage_range} m")
    print(f"[Verify] bury tolerance = {args_cli.bury_tolerance} m\n")

    all_displacements = []
    all_sinkages      = []
    all_trapped       = []

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()

        root_pos_0 = wp.to_torch(unwrapped.robot.data.root_link_pos_w).clone()
        pos_xy_0  = root_pos_0[:, :2] - unwrapped.scene.env_origins[:, :2]
        sinkages  = unwrapped._sinkage.clone()

        for _ in range(hold_steps):
            obs, rew, term, trunc, info = env.step(zero_action)

        root_pos_f = wp.to_torch(unwrapped.robot.data.root_link_pos_w)
        pos_xy_f   = root_pos_f[:, :2] - unwrapped.scene.env_origins[:, :2]
        disp       = torch.norm(pos_xy_f - pos_xy_0, dim=-1)

        trapped = (disp < args_cli.bury_tolerance).float()
        all_displacements.append(disp.cpu())
        all_sinkages.append(sinkages.cpu())
        all_trapped.append(trapped.cpu())

        print(f"[Ep {ep+1}] trapped: {trapped.mean().item()*100:5.1f}%  "
              f"disp mean={disp.mean().item():.3f} m  max={disp.max().item():.3f} m  "
              f"sinkage mean={sinkages.mean().item():.3f} m")

    disp_all    = torch.cat(all_displacements)
    sink_all    = torch.cat(all_sinkages)
    trapped_all = torch.cat(all_trapped)

    print("\n" + "=" * 60)
    print(f"[Summary] overall trap rate: {trapped_all.mean().item()*100:.1f}%")
    print(f"[Summary] mean displacement: {disp_all.mean().item():.3f} m")
    print(f"[Summary] max  displacement: {disp_all.max().item():.3f} m")


    median_sink = sink_all.median().item()
    shallow = sink_all <= median_sink
    deep    = ~shallow
    print(f"\n[Tier] shallow sinkage (< {median_sink:.3f} m): "
          f"trap rate {trapped_all[shallow].mean().item()*100:.1f}%  "
          f"mean disp {disp_all[shallow].mean().item():.3f} m")
    print(f"[Tier] deep    sinkage (>={median_sink:.3f} m): "
          f"trap rate {trapped_all[deep].mean().item()*100:.1f}%  "
          f"mean disp {disp_all[deep].mean().item():.3f} m")

    print("\n[Verdict]")
    rate = trapped_all.mean().item()
    if rate >= 0.90:
        print(f"  ✓ Real entrapment confirmed ({rate*100:.1f}% >= 90%). Safe to train.")
    elif rate >= 0.70:
        print(f"  ⚠ Borderline ({rate*100:.1f}%). Consider raising dr_sinkage_range min"
              f" from {cfg.dr_sinkage_range[0]:.2f} to {cfg.dr_sinkage_range[0]+0.03:.2f}.")
    else:
        print(f"  ✗ Fake entrapment problem ({rate*100:.1f}% < 70%). DO NOT train yet.")
        print(f"    Options: raise dr_sinkage_range, raise sand µ, raise ke.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
