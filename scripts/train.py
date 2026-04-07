"""
Mars Rover — Regolith Escape Recovery — Training Script
Newton DirectRLEnv + skrl PPO

Usage (via launch.sh from repo root):
    ./launch.sh scripts/train.py --num_envs 64
    ./launch.sh scripts/train.py --num_envs 64 --timesteps 200000
    ./launch.sh scripts/train.py --num_envs 64 --checkpoint experiments/.../best_agent.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mars Rover Regolith Escape Recovery RL Training")
parser.add_argument("--num_envs",   type=int,   default=64)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--checkpoint", type=str,   default=None)
parser.add_argument("--timesteps",  type=int,   default=200_000)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
# Always headless for training.  LAUNCH_OV_APP=1 forces SimulationApp creation
# without loading the heavy RTX rendering stack (enable_cameras loads viewport +
# replicator + shaders which adds >60s startup and wastes VRAM).
os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-init imports ──────────────────────────────────────────────────────────
import math
import gymnasium as gym
import torch

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # registers "MarsRover-RegolithEscape-v0"
from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg


# ── Model definitions ──────────────────────────────────────────────────────────

class PolicyNet(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128),                   nn.ELU(),
            nn.Linear(128, 64),                    nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        x = self.net(inputs["states"])
        return x, self.log_std.expand_as(x), {}


class ValueNet(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128),                   nn.ELU(),
            nn.Linear(128, 64),                    nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    set_seed(args_cli.seed)

    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device    = env.device
    num_obs   = env_cfg.observation_space
    num_act   = env_cfg.action_space
    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))

    print(f"\n{'='*55}")
    print(f"  Mars Rover Regolith Escape — PPO Training")
    print(f"  Obs: {num_obs}D | Act: {num_act}D | Envs: {env.num_envs}")
    print(f"  Device: {device} | Seed: {args_cli.seed}")
    print(f"{'='*55}\n")

    models  = {
        "policy": PolicyNet(obs_space, act_space, device),
        "value":  ValueNet(obs_space, act_space, device),
    }
    rollouts = 24
    memory   = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    exp_dir = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts":           rollouts,
        "learning_epochs":    5,
        "mini_batches":       4,
        "discount_factor":    0.99,
        "lambda":             0.95,
        "learning_rate":      3e-4,
        "grad_norm_clip":     1.0,
        "ratio_clip":         0.2,
        "value_clip":         0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.02,
        "value_loss_scale":   1.0,
        "state_preprocessor":             RunningStandardScaler,
        "state_preprocessor_kwargs":      {"size": num_obs, "device": device},
        "value_preprocessor":             RunningStandardScaler,
        "value_preprocessor_kwargs":      {"size": 1, "device": device},
        "experiment": {
            "directory":          exp_dir,
            "experiment_name":    "ppo_regolith",
            "write_interval":     100,
            "checkpoint_interval": 2000,
            "wandb":              False,
        },
    })

    agent = PPO(
        models=models, memory=memory, cfg=ppo_cfg,
        observation_space=obs_space, action_space=act_space, device=device,
    )

    if args_cli.checkpoint:
        print(f"Resuming from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # Patch: forward Isaac Lab extras["log"] entries to TensorBoard each rollout.
    # skrl's PPO doesn't auto-log these; we inject via agent.track_data().
    _raw_env = env.unwrapped
    _orig_post = agent.post_interaction

    def _post_interaction_with_extras(timestep, timesteps):
        _orig_post(timestep=timestep, timesteps=timesteps)
        try:
            log = _raw_env.extras.get("log", {})
            for k, v in log.items():
                val = float(v.mean().item()) if hasattr(v, "mean") else float(v)
                agent.track_data(f"Info / {k}", val)
        except Exception:
            pass

    agent.post_interaction = _post_interaction_with_extras

    trainer = SequentialTrainer(
        cfg={"timesteps": args_cli.timesteps, "headless": args_cli.headless},
        agents=agent, env=env,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    train()
