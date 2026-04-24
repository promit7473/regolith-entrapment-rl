"""
Sim2Sim Baseline — A→B Navigation WITHOUT Entrapment Recovery

Same setup as run_validation.py but the escape primitive is never triggered.
The PD controller drives toward B and gets stuck — this is the control group
that shows what happens without the recovery primitive.

Usage:
  ./launch.sh sim2sim_validation/run_baseline.py \\
      --num_envs 8 --num_trials 20

Metrics saved to: experiments/sim2sim/baseline_<timestamp>.json
Compare with run_validation.py output for the paper table.
"""

import argparse
import os
import sys
import json
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Sim Baseline: A→B navigation, no recovery")
parser.add_argument("--num_envs",   type=int,   default=8)
parser.add_argument("--num_trials", type=int,   default=20)
parser.add_argument("--goal_x",     type=float, default=6.0)
parser.add_argument("--goal_y",     type=float, default=0.0)
parser.add_argument("--max_steps",  type=int,   default=2000)
parser.add_argument("--seed",       type=int,   default=123)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-init imports ──────────────────────────────────────────────────────────
import torch
import gymnasium as gym

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils import set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs
from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg

from sim2sim_validation.nav_controller import PDNavController
from sim2sim_validation.metrics import MetricsTracker


def get_pos_yaw(env_unwrapped):
    root = env_unwrapped.robot.data.root_state_w
    pos_xy = root[:, :2].clone()
    qx, qy, qz, qw = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return pos_xy, yaw


def main():
    set_seed(args_cli.seed)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = args_cli.num_envs

    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sinkage_min    = 0.18
    cfg.sinkage_max    = 0.22

    raw_env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg)
    env     = SkrlVecEnvWrapper(raw_env)
    unwrapped: EntrapmentEnv = raw_env.unwrapped

    goal_world   = torch.tensor([args_cli.goal_x, args_cli.goal_y],
                                 dtype=torch.float32, device=device)
    env_origins_xy = unwrapped.scene.env_origins[:, :2]
    goal_xy      = env_origins_xy + goal_world.unsqueeze(0)

    nav = PDNavController(num_envs=num_envs, device=device,
                          drive_speed=0.6, heading_gain=1.2, arrival_radius=0.5)
    nav.set_goal(goal_xy)

    # MetricsTracker with no escape events — all escaped=False, time_to_escape=-1
    metrics = MetricsTracker(num_envs=num_envs, device=device, goal_xy=goal_xy)

    num_trials  = args_cli.num_trials
    max_steps   = args_cli.max_steps
    trials_done = 0
    trial_step  = torch.zeros(num_envs, dtype=torch.long, device=device)
    all_env_ids = torch.arange(num_envs, device=device)

    obs, _ = env.reset()
    pos_xy, _ = get_pos_yaw(unwrapped)
    metrics.begin_trial(all_env_ids, pos_xy)

    # Dummy escape dir for metrics (pointing toward goal — heading error will be 0)
    unwrapped._escape_dir = (goal_xy - pos_xy)
    unwrapped._escape_dir /= (torch.norm(unwrapped._escape_dir, dim=-1, keepdim=True) + 1e-6)

    never_escaped = torch.zeros(num_envs, dtype=torch.bool, device=device)

    print(f"\n[baseline] Running {num_trials} trials — PD nav only, no recovery\n")

    while trials_done < num_trials:
        pos_xy, yaw = get_pos_yaw(unwrapped)
        action, arrived = nav.step(pos_xy, yaw)

        obs, _, terminated, truncated, _ = env.step(action)

        # No escape primitive — pass never_escaped every step
        metrics.step(
            pos_xy=pos_xy,
            newly_escaped=never_escaped,
            arrived=arrived,
            escape_dir=unwrapped._escape_dir,
        )

        trial_step += 1
        done_mask = terminated | truncated | (trial_step >= max_steps) | arrived

        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=True)[0]
            finished = metrics.end_trial(done_ids)
            for r in finished:
                trials_done += 1
                status = "GOAL" if r.reached_goal else "STUCK"
                print(f"  Trial {r.trial_id:3d} [{status:5s}] "
                      f"goal={r.time_to_goal:4d}s  "
                      f"eff={r.path_efficiency:.3f}")
                if trials_done >= num_trials:
                    break

            if trials_done < num_trials:
                trial_step[done_ids] = 0
                metrics.begin_trial(done_ids, pos_xy)

    metrics.print_summary()
    summary = metrics.summary()

    out_dir = os.path.join(REPO_ROOT, "experiments", "sim2sim")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"baseline_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args_cli), "summary": summary}, f, indent=2)
    print(f"[baseline] Results saved → {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
