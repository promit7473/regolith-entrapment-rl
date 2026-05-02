"""Rocking-maneuver baseline — scientific comparison for the paper.

Implements the classical wheel-extrication rocking strategy: alternate max-forward
and max-backward drive commands at a fixed period, no steering.  This is the
standard hand-crafted recovery primitive used in field robotics for stuck rovers.

The script runs N_EPISODES episodes with DR-matched sinkage + friction (same
ranges as PPO training) and records per-episode outcomes.  A sinkage-level sweep
provides per-difficulty escape rate data for comparison with the PPO policy.

Run:
    ./launch.sh scripts/rocking_baseline.py --num_envs 64
    ./launch.sh scripts/rocking_baseline.py --num_envs 64 --episodes 400
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rocking baseline for regolith entrapment recovery")
parser.add_argument("--num_envs",    type=int,   default=64)
parser.add_argument("--episodes",    type=int,   default=200,
                    help="Total episodes to run (split across sinkage levels)")
parser.add_argument("--half_period", type=float, default=2.0,
                    help="Duration of each forward/backward half-cycle in seconds")
parser.add_argument("--drive_mag",   type=float, default=1.0,
                    help="Drive command magnitude [-1, 1] (1.0 = max)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv

import torch
import gymnasium as gym

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # noqa: F401 — registers MarsRover-RegolithEscape-v0
from envs.entrapment_env import EntrapmentEnvCfg

# Sinkage levels to sweep — matches v9 curriculum range endpoints + mid-points.
SINKAGE_LEVELS = [0.15, 0.20, 0.25, 0.28]
EPISODES_PER_LEVEL = 50   # episodes per sinkage level (200 total)


def run_level(env, sinkage: float, n_eps: int, device: str,
              num_act: int, half_period_steps: int, drive_mag: float):
    """Run n_eps episodes at a fixed sinkage depth and return outcome list."""
    unwrapped = env.unwrapped
    # Override sinkage to a fixed value for this level
    unwrapped.cfg.dr_sinkage_range = (sinkage, sinkage)

    results = []
    completed = 0

    obs, _ = env.reset()

    # Per-env step counter and phase tracker
    step_counter = torch.zeros(unwrapped.num_envs, device=device)

    while completed < n_eps:
        # Rocking action: sign alternates with period = 2 × half_period_steps
        phase = (step_counter // half_period_steps) % 2  # 0 = forward, 1 = backward
        drive_sign = torch.where(phase == 0,
                                 torch.ones_like(phase),
                                 -torch.ones_like(phase))  # (N,)

        action = torch.zeros(unwrapped.num_envs, num_act, device=device)
        action[:, :6] = (drive_sign * drive_mag).unsqueeze(-1).expand(-1, 6)
        # steer_cmd stays 0 — pure forward/backward rocking

        obs, _, terminated, truncated, _ = env.step(action)
        step_counter += 1.0
        done_mask = terminated | truncated

        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=True)[0]
            # Read out per-episode data before reset wipes state
            root_pos  = unwrapped.root_pos
            spawn_pos = unwrapped._spawn_pos
            esc_dir   = unwrapped._escape_dir

            for ei in done_ids.tolist():
                if completed >= n_eps:
                    break
                rel = root_pos[ei, :2] - spawn_pos[ei]
                final_dist = float((rel * esc_dir[ei]).sum().item())
                escaped = bool(terminated[ei].item()) and (final_dist > 2.5)
                ep_steps = int(unwrapped._episode_step[ei].item())
                results.append({
                    "sinkage":       sinkage,
                    "escaped":       int(escaped),
                    "steps":         ep_steps,
                    "final_dist":    final_dist,
                    "curriculum":    float(unwrapped._curriculum_progress.mean().item()),
                })
                completed += 1

            # Reset per-env step counter for done envs
            step_counter[done_mask] = 0.0
            # env auto-resets done envs internally (DirectRLEnv)

    return results


def main():
    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # Disable failure-mode CSV logging from the env itself (this script does its own)
    cfg.log_failure_modes = False

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg)
    unwrapped = env.unwrapped
    device    = unwrapped.device
    num_act   = cfg.action_space

    dt = float(cfg.sim.dt) * float(cfg.decimation)
    half_period_steps = max(1, int(args_cli.half_period / dt))

    print(f"[rocking_baseline] dt={dt:.4f}s  half_period_steps={half_period_steps}")
    print(f"[rocking_baseline] drive_mag={args_cli.drive_mag}  episodes_per_level={EPISODES_PER_LEVEL}")
    print(f"[rocking_baseline] sinkage_levels={SINKAGE_LEVELS}")
    print()

    all_results = []
    for sinkage in SINKAGE_LEVELS:
        print(f"  Running sinkage={sinkage:.2f}m  ({EPISODES_PER_LEVEL} episodes)...", flush=True)
        level_results = run_level(
            env, sinkage, EPISODES_PER_LEVEL, device, num_act,
            half_period_steps, args_cli.drive_mag,
        )
        all_results.extend(level_results)
        esc_rate = sum(r["escaped"] for r in level_results) / max(1, len(level_results))
        avg_dist = sum(r["final_dist"] for r in level_results) / max(1, len(level_results))
        print(f"    escape_rate={esc_rate:.3f}  avg_final_dist={avg_dist:.3f}m")

    print()
    print("=" * 60)
    print("  ROCKING BASELINE RESULTS")
    print("=" * 60)
    print(f"  {'Sinkage (m)':<15} {'Escape Rate':<14} {'N'}")
    print(f"  {'-'*15} {'-'*14} {'-'*4}")
    for sinkage in SINKAGE_LEVELS:
        lvl = [r for r in all_results if r["sinkage"] == sinkage]
        rate = sum(r["escaped"] for r in lvl) / max(1, len(lvl))
        print(f"  {sinkage:<15.2f} {rate:<14.3f} {len(lvl)}")
    overall = sum(r["escaped"] for r in all_results) / max(1, len(all_results))
    print(f"  {'Overall':<15} {overall:<14.3f} {len(all_results)}")
    print("=" * 60)

    # Save CSV
    out_dir = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rocking_baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sinkage", "escaped", "steps", "final_dist", "curriculum",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[rocking_baseline] Results saved to {csv_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
