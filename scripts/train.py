"""
Mars Rover — Regolith Escape Recovery — Training Script
Newton DirectRLEnv + skrl Recurrent PPO (GRU)

Usage (via launch.sh from repo root):
    ./launch.sh scripts/train.py --num_envs 64
    ./launch.sh scripts/train.py --num_envs 64 --timesteps 200000
    ./launch.sh scripts/train.py --num_envs 64 --checkpoint experiments/.../best_agent.pt
"""

import argparse
import os
import sys

# Workaround: Python 3.11's platform._sys_version regex fails on some
# conda-forge sys.version strings ('... | packaged by conda-forge | ... Oct 22 2025 ...').
# wandb.init triggers the parse via settings.to_proto() → platform.python_implementation().
# Pre-seed platform's cache so the parse is never attempted.
import platform as _platform
_platform._sys_version_cache[sys.version] = (
    "CPython",
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "", "", "", "",
    sys.version.split("[", 1)[-1].rstrip("]") if "[" in sys.version else "",
)

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
import torch.nn as nn

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO_RNN
from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # registers "MarsRover-RegolithEscape-v0"
from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg

# ── GRU hyperparameters ────────────────────────────────────────────────────────
GRU_HIDDEN  = 256   # GRU hidden units
GRU_LAYERS  = 1     # GRU stacked layers
SEQ_LEN     = 32    # BPTT sequence length (~1.3s at 25Hz — covers entrap detection window)
ROLLOUTS    = 64    # Steps stored per env per update (64 / 32 = 2 seqs per env, 2× more data)

# Asymmetric actor-critic: policy reads the first POLICY_OBS_DIM dims, critic
# reads the full tensor (policy obs + privileged oracle features). Must match
# envs/entrapment_env.py EntrapmentEnvCfg.policy_observation_space.
POLICY_OBS_DIM = 29


# ── Model definitions ──────────────────────────────────────────────────────────

class GRUPolicyNet(GaussianMixin, Model):
    """
    Recurrent policy: Linear encoder → GRU → MLP head → Gaussian action.

    Architecture (29D obs → 10D action):
        encoder : Linear(29→128) + ELU
        gru     : GRU(128 → 256, 1 layer)
        head    : Linear(256→64) + ELU + Linear(64→num_actions)
        log_std : learnable parameter (shared across actions)

    PPO_RNN passes hidden states via inputs["rnn"] = [h_t].
    The model returns updated states in the output dict: {"rnn": [h_{t+1}]}.
    During training (BPTT), states arrive as (seq*batch, obs_dim) and are
    reshaped to (seq, batch, obs_dim) before the GRU call.
    """

    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)

        self._num_envs  = num_envs
        self._hidden    = GRU_HIDDEN
        self._layers    = GRU_LAYERS
        self._seq_len   = SEQ_LEN

        # Policy encoder sees only the on-board observation slice, never the
        # privileged features — those are critic-only.
        self.encoder = nn.Sequential(
            nn.Linear(POLICY_OBS_DIM, 128), nn.ELU(),
        )
        self.gru = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # GRU has 1 state (hidden h). Shape: (num_layers, num_envs, hidden_size).
        return {
            "rnn": {
                "sequence_length": self._seq_len,
                "sizes": [(self._layers, self._num_envs, self._hidden)],
            }
        }

    def compute(self, inputs, role=""):
        # Slice off privileged features; the policy sees only the first POLICY_OBS_DIM
        # dims so the learned policy stays deployable on the real rover.
        states = inputs["states"][:, :POLICY_OBS_DIM]
        rnn_list = inputs.get("rnn", [None])
        hidden = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        # Determine batch/sequence dimensions.
        # During rollout: hidden shape is (layers, num_envs, hidden); states is (num_envs, obs).
        # During training: hidden shape is (layers, mini_batch, hidden); states is (mini_batch*seq, obs).
        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states)               # (batch*seq, 128)
        x = x.view(seq, batch, -1)             # (seq, batch, 128)
        x, h_n = self.gru(x, hidden)           # x: (seq, batch, hidden), h_n: (layers, batch, hidden)
        x = x.reshape(seq * batch, -1)         # (seq*batch, hidden)
        output = self.head(x)                  # (seq*batch, num_actions)

        return output, self.log_std.expand_as(output), {"rnn": [h_n]}


class GRUValueNet(DeterministicMixin, Model):
    """
    Recurrent value critic: same GRU architecture as policy, scalar output.
    Shares no weights with policy.

    Reads the full observation tensor (policy obs + privileged oracle features),
    so the critic gets a lower-variance value estimate than the actor. The
    privileged slice is discarded at deployment — only the policy is exported.
    """

    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self._num_envs  = num_envs
        self._hidden    = GRU_HIDDEN
        self._layers    = GRU_LAYERS
        self._seq_len   = SEQ_LEN

        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, 128), nn.ELU(),
        )
        self.gru = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self._seq_len,
                "sizes": [(self._layers, self._num_envs, self._hidden)],
            }
        }

    def compute(self, inputs, role=""):
        states = inputs["states"]
        rnn_list = inputs.get("rnn", [None])
        hidden = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states)
        x = x.view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        value = self.head(x)

        return value, {"rnn": [h_n]}


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    set_seed(args_cli.seed)

    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Pass total timesteps to env so curriculum denominator scales correctly
    # with --timesteps argument instead of assuming a fixed 200k default.
    env.unwrapped._total_timesteps = args_cli.timesteps

    device    = env.device
    num_envs  = env.num_envs
    num_obs   = env_cfg.observation_space
    num_act   = env_cfg.action_space
    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))

    print(f"\n{'='*60}")
    print(f"  Mars Rover Regolith Escape — Recurrent PPO (GRU)")
    print(f"  Obs: {num_obs}D | Act: {num_act}D | Envs: {num_envs}")
    print(f"  GRU hidden: {GRU_HIDDEN} | Seq len: {SEQ_LEN} | Rollouts: {ROLLOUTS}")
    print(f"  Device: {device} | Seed: {args_cli.seed}")
    print(f"{'='*60}\n")

    models = {
        "policy": GRUPolicyNet(obs_space, act_space, device, num_envs=num_envs),
        "value":  GRUValueNet(obs_space, act_space, device, num_envs=num_envs),
    }
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=num_envs, device=device)

    exp_dir = os.path.join(REPO_ROOT, "experiments", "regolith_recovery")
    ppo_cfg = PPO_RNN_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts":           ROLLOUTS,
        "learning_epochs":    3,   # reduced from 5 — prevents critic overfit on easy curriculum episodes
        "mini_batches":       8,   # raised from 4 — smaller batches, more gradient steps per rollout
        "discount_factor":    0.99,
        "lambda":             0.95,
        "learning_rate":      3e-4,
        "grad_norm_clip":     1.0,
        "ratio_clip":         0.2,
        "value_clip":         0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.03,  # 0.02 caused collapse in prior run; 0.03 maintains exploration
        "value_loss_scale":   1.0,
        "state_preprocessor":             RunningStandardScaler,
        "state_preprocessor_kwargs":      {"size": num_obs, "device": device},
        "value_preprocessor":             RunningStandardScaler,
        "value_preprocessor_kwargs":      {"size": 1, "device": device},
        "experiment": {
            "directory":          exp_dir,
            "experiment_name":    "ppo_gru_regolith",
            "write_interval":     100,
            "checkpoint_interval": 2000,
            "wandb":              True,
            "wandb_kwargs": {
                "project":  "regolith-entrapment-rl",
                "name":     "ppo_gru_asymmetric_v1",
                "tags":     ["asymmetric-critic", "gru", "5070ti"],
                "sync_tensorboard": True,   # mirror TB scalars into W&B automatically
            },
        },
    })

    agent = PPO_RNN(
        models=models, memory=memory, cfg=ppo_cfg,
        observation_space=obs_space, action_space=act_space, device=device,
    )

    if args_cli.checkpoint:
        if not os.path.exists(args_cli.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
        print(f"Resuming from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # Patch: forward Isaac Lab extras["log"] entries to TensorBoard each rollout.
    # skrl's PPO_RNN doesn't auto-log these; we inject via agent.track_data().
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
