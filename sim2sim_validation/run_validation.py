"""
Sim2Sim Validation — A→B Navigation with Entrapment Recovery

Architecture (dual-brain):
  1. PDNavController drives rover from A toward goal B
  2. ModeSwitcher detects entrap_flag and hands off to the PPO escape primitive
  3. After escape, navigation resumes from new position toward B

Usage:
  ./launch.sh sim2sim_validation/run_validation.py \\
      --checkpoint experiments/.../best_agent.pt \\
      --num_envs 8 --num_trials 20

Metrics written to: experiments/sim2sim/summary_<timestamp>.json
"""

import argparse
import os
import sys
import json
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Sim Validation: A→B with entrapment recovery")
parser.add_argument("--checkpoint",   type=str,   required=True,
                    help="Path to trained PPO checkpoint (.pt)")
parser.add_argument("--num_envs",     type=int,   default=8,
                    help="Parallel environments (must be ≤ training num_envs for PPO_RNN states)")
parser.add_argument("--num_trials",   type=int,   default=20,
                    help="Total trials to run (spread across envs)")
parser.add_argument("--goal_x",       type=float, default=6.0,
                    help="Goal position X in world frame (m) — B")
parser.add_argument("--goal_y",       type=float, default=0.0,
                    help="Goal position Y in world frame (m) — B")
parser.add_argument("--max_steps",    type=int,   default=2000,
                    help="Max policy steps per trial before timeout")
parser.add_argument("--seed",         type=int,   default=123)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-init imports ──────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO_RNN
from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # registers "MarsRover-RegolithEscape-v0"
from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg

from sim2sim_validation.nav_controller import PDNavController
from sim2sim_validation.mode_switcher import ModeSwitcher, Mode
from sim2sim_validation.metrics import MetricsTracker

# ── Model definitions (must match train.py exactly) ───────────────────────────

GRU_HIDDEN     = 256
GRU_LAYERS     = 1
SEQ_LEN        = 32
ROLLOUTS       = 64
POLICY_OBS_DIM = 29


class GRUPolicyNet(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)
        self._num_envs = num_envs
        self._hidden   = GRU_HIDDEN
        self._layers   = GRU_LAYERS
        self._seq_len  = SEQ_LEN

        self.encoder = nn.Sequential(nn.Linear(POLICY_OBS_DIM, 128), nn.ELU())
        self.gru     = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head    = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states = inputs["states"][:, :POLICY_OBS_DIM]
        rnn_list = inputs.get("rnn", [None])
        hidden   = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states).view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        output = self.head(x)
        return output, self.log_std.expand_as(output), {"rnn": [h_n]}


class GRUValueNet(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._num_envs = num_envs
        self._hidden   = GRU_HIDDEN
        self._layers   = GRU_LAYERS
        self._seq_len  = SEQ_LEN

        self.encoder = nn.Sequential(nn.Linear(self.num_observations, 128), nn.ELU())
        self.gru     = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head    = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states   = inputs["states"]
        rnn_list = inputs.get("rnn", [None])
        hidden   = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states).view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        return self.head(x), {"rnn": [h_n]}


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_agent(env, device, num_envs, checkpoint_path):
    """Load the PPO_RNN agent from checkpoint."""
    obs_space = env.observation_space
    act_space = env.action_space

    policy = GRUPolicyNet(obs_space, act_space, device, num_envs)
    value  = GRUValueNet(obs_space, act_space, device, num_envs)

    models = {"policy": policy, "value": value}

    cfg = PPO_RNN_DEFAULT_CONFIG.copy()
    cfg["state_preprocessor"]       = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": device}
    cfg["value_preprocessor"]       = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=num_envs, device=device)

    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent


def get_pos_yaw(env_unwrapped):
    """Extract (pos_xy, yaw) from root_state [x,y,z, qx,qy,qz,qw, ...]."""
    root = env_unwrapped.robot.data.root_state_w   # (N, 13+)
    pos_xy = root[:, :2].clone()
    qx, qy, qz, qw = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return pos_xy, yaw


def override_escape_dir(env_unwrapped, goal_xy_tensor):
    """
    Override the env's internal _escape_dir to point toward the GPS goal B.
    This makes the PPO primitive (trained with random headings) orient toward B.
    Must be called after env reset, before the escape primitive fires.
    """
    pos_xy = env_unwrapped.robot.data.root_state_w[:, :2]
    rel = goal_xy_tensor - pos_xy                           # (N, 2)
    dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)
    env_unwrapped._escape_dir = rel / dist


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(args_cli.seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs  = args_cli.num_envs

    # ── Build env ─────────────────────────────────────────────────────────────
    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = num_envs
    # Disable curriculum for validation — use a fixed, representative sinkage
    cfg.sinkage_min    = 0.18
    cfg.sinkage_max    = 0.22

    raw_env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg)
    env     = SkrlVecEnvWrapper(raw_env)
    unwrapped: EntrapmentEnv = raw_env.unwrapped

    # ── Build and load agent ───────────────────────────────────────────────────
    print(f"[sim2sim] Loading checkpoint: {args_cli.checkpoint}")
    agent = build_agent(env, device, num_envs, args_cli.checkpoint)

    # ── Goal B (GPS) ──────────────────────────────────────────────────────────
    goal_world = torch.tensor([args_cli.goal_x, args_cli.goal_y],
                               dtype=torch.float32, device=device)
    # Per-env goal: each env is offset by Isaac Lab's env_origins
    # (N, 2) in world frame = env_origin[:, :2] + local goal
    env_origins_xy = unwrapped.scene.env_origins[:, :2]           # (N, 2)
    goal_xy = env_origins_xy + goal_world.unsqueeze(0)             # (N, 2)

    # ── Controllers & trackers ─────────────────────────────────────────────────
    nav = PDNavController(
        num_envs=num_envs, device=device,
        drive_speed=0.6, heading_gain=1.2, arrival_radius=0.5,
    )
    nav.set_goal(goal_xy)

    switcher = ModeSwitcher(
        num_envs=num_envs, device=device,
        escape_distance=3.0, trigger_steps=15,
    )

    metrics = MetricsTracker(
        num_envs=num_envs, device=device, goal_xy=goal_xy,
    )

    # ── Validation loop ────────────────────────────────────────────────────────
    num_trials    = args_cli.num_trials
    max_steps     = args_cli.max_steps
    trials_done   = 0
    trial_step    = torch.zeros(num_envs, dtype=torch.long, device=device)

    obs, _ = env.reset()
    all_env_ids = torch.arange(num_envs, device=device)

    # Initialize metrics for first episode
    pos_xy, _ = get_pos_yaw(unwrapped)
    metrics.begin_trial(all_env_ids, pos_xy)
    switcher.reset(all_env_ids)

    # Override escape_dir to point toward B for every env
    override_escape_dir(unwrapped, goal_xy)

    agent.init()
    states = obs

    print(f"\n[sim2sim] Running {num_trials} trials across {num_envs} envs "
          f"(goal B = [{args_cli.goal_x:.1f}, {args_cli.goal_y:.1f}] m from env origin)\n")

    while trials_done < num_trials:
        pos_xy, yaw = get_pos_yaw(unwrapped)

        # ── Mode FSM ──────────────────────────────────────────────────────────
        # Extract entrap_flag from obs (index 26 in 29D obs)
        entrap_flag = states[:, 26].clamp(0.0, 1.0)
        mode, newly_escaped, _ = switcher.update(pos_xy, entrap_flag, goal_xy)

        # ── Override escape_dir when escape mode just triggered ───────────────
        # switcher already computed escape_dir = toward B at trigger time,
        # but we force the env's internal buffer too so the PPO dist_norm obs
        # is computed correctly against the GPS-directed heading.
        just_triggered = (mode == Mode.ESCAPE) & (switcher.steps_in_escape == 1)
        if just_triggered.any():
            for i in just_triggered.nonzero(as_tuple=True)[0].tolist():
                unwrapped._escape_dir[i] = switcher.escape_dir[i]

        # ── Choose action ──────────────────────────────────────────────────────
        nav_action, arrived = nav.step(pos_xy, yaw)

        with torch.no_grad():
            ppo_action, _, _ = agent.act({"states": states}, role="policy")

        # NAVIGATE / REPLANNED → use PD nav;  ESCAPE → use PPO primitive
        navigating_mask = (mode != Mode.ESCAPE).unsqueeze(-1).float()
        action = nav_action * navigating_mask + ppo_action * (1.0 - navigating_mask)

        # ── Step environment ───────────────────────────────────────────────────
        obs_next, _, terminated, truncated, _ = env.step(action)
        states = obs_next

        # ── Metrics update ────────────────────────────────────────────────────
        metrics.step(
            pos_xy=pos_xy,
            newly_escaped=newly_escaped,
            arrived=arrived,
            escape_dir=switcher.escape_dir,
        )

        trial_step += 1
        done_mask = terminated | truncated | (trial_step >= max_steps) | arrived

        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=True)[0]
            finished = metrics.end_trial(done_ids)
            for r in finished:
                trials_done += 1
                status = ("GOAL" if r.reached_goal else
                          "ESCAPED" if r.escaped else "TIMEOUT")
                print(f"  Trial {r.trial_id:3d} [{status:7s}] "
                      f"esc={r.time_to_escape:4d}s  "
                      f"goal={r.time_to_goal:4d}s  "
                      f"eff={r.path_efficiency:.3f}  "
                      f"hdg_err={r.escape_heading_error:.1f}°")
                if trials_done >= num_trials:
                    break

            if trials_done < num_trials:
                # Reset finished envs
                trial_step[done_ids] = 0
                metrics.begin_trial(done_ids, pos_xy)
                switcher.reset(done_ids)
                override_escape_dir(unwrapped, goal_xy)

    # ── Summary ────────────────────────────────────────────────────────────────
    metrics.print_summary()
    summary = metrics.summary()

    # Save JSON
    out_dir = os.path.join(REPO_ROOT, "experiments", "sim2sim")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"summary_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args_cli), "summary": summary}, f, indent=2)
    print(f"[sim2sim] Results saved → {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
