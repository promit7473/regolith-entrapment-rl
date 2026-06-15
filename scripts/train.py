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
parser.add_argument("--timesteps",  type=int,   default=200_000,
                    help="Training timesteps per env. 200k default (5070Ti/64 envs). "
                         "4M recommended for RTX 4090 with 512 envs.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
# Capture the user's --headless intent BEFORE we force the Isaac stack off
# below (AppLauncher.add_app_launcher_args added the --headless flag).
_user_headless = bool(getattr(args_cli, "headless", False))

# The Isaac/Omniverse RTX GUI is ALWAYS off for training (broken in conda-Python;
# the heavy stack wastes >60s + VRAM). LAUNCH_OV_APP=1 forces SimulationApp
# creation without loading viewport/replicator/shaders.
os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True

# The lightweight Newton ViewerGL (the live sand/rover window) is SEPARATE and
# is controlled by --headless the intuitive way:
#   ./launch.sh scripts/train.py ...              -> GUI shows (default)
#   ./launch.sh scripts/train.py --headless ...   -> no GUI
# setdefault lets an explicit ENTRAPMENT_NO_VIEWER env var still override.
# Note: the viewer costs little speed (MPM dominates) but the multi-env view is
# cluttered — use a small --num_envs (e.g. 4) if you actually want to watch.
os.environ.setdefault("ENTRAPMENT_NO_VIEWER", "1" if _user_headless else "0")
print(f"[train] Newton viewer: "
      f"{'OFF (--headless)' if os.environ['ENTRAPMENT_NO_VIEWER']=='1' else 'ON (pass --headless to disable)'}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-init imports ──────────────────────────────────────────────────────────
import math
import gymnasium as gym
import numpy as np
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
# v9=0.015, v10=0.05. Seeds 0&2 still collapsed (std≈0.56) while seed 1 thrived
# (std=1.70). Raised to 0.08 to reliably sustain std≥0.8 through early curriculum.
# Single source of truth — used by ppo_cfg, the entropy floor, and W&B config.
ENTROPY_SCALE = 0.08

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
    # Seed all RNG sources for reproducibility.
    # skrl's set_seed handles torch + numpy internally; we also call them
    # directly so any code path that doesn't go through skrl is covered.
    set_seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    ablation = os.environ.get("ABLATION", "").strip()
    if ablation:
        if ablation == "no_priv_critic":
            env_cfg.use_privileged_critic = False
            env_cfg.observation_space = env_cfg.policy_observation_space
        elif ablation == "no_dr":
            for k in ("dr_noise_wheel_vel", "dr_noise_slip", "dr_noise_steer_pos",
                      "dr_noise_imu_acc", "dr_noise_grav_z",
                      "dr_noise_drive_torque", "dr_noise_dist_norm"):
                setattr(env_cfg, k, 0.0)
            env_cfg.dr_friction_range = (0.75, 0.75)
        elif ablation == "no_pen_grind":
            env_cfg.pen_grind = 0.0
        elif ablation == "no_pen_hop":
            env_cfg.pen_hop = 0.0
        else:
            raise ValueError(f"Unknown ABLATION={ablation!r}")
        print(f"[train] Ablation active: {ablation}")

    print(f"[train] Calling gym.make (Newton init starts here — first cold launch can take 10-15 min)…", flush=True)
    env = gym.make("MarsRover-RegolithEscape-v0", cfg=env_cfg)
    print(f"[train] gym.make returned. Wrapping env…", flush=True)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Pass total timesteps to env so curriculum denominator scales correctly
    # with --timesteps argument instead of assuming a fixed 200k default.
    env.unwrapped._total_timesteps = args_cli.timesteps
    print(f"[train] Env ready (num_envs={args_cli.num_envs}). Building agent…", flush=True)

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

    _tag = ablation if ablation else f"seed_{args_cli.seed}"
    exp_dir = os.path.join(REPO_ROOT, "experiments", "regolith_recovery", _tag)

    # W&B run identity — ablations get their own group, multi-seed runs share one group
    # so the W&B UI can plot mean±std across seeds automatically.
    if ablation:
        _wb_group   = f"ablation_{ablation}"
        _wb_name    = f"ablation_{ablation}_s{args_cli.seed}"
        _wb_tags    = ["ablation", ablation, "gru", "rtx4090"]
    else:
        _wb_group   = f"multiseed_{args_cli.timesteps//1000}k"
        _wb_name    = f"seed_{args_cli.seed}"
        _wb_tags    = ["multiseed", "gru", "asymmetric-critic", "rtx4090", f"seed_{args_cli.seed}"]

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
        "entropy_loss_scale": ENTROPY_SCALE,
        "value_loss_scale":   1.0,
        "state_preprocessor":             RunningStandardScaler,
        "state_preprocessor_kwargs":      {"size": num_obs, "device": device},
        "value_preprocessor":             RunningStandardScaler,
        "value_preprocessor_kwargs":      {"size": 1, "device": device},
        "experiment": {
            "directory":          exp_dir,
            "experiment_name":    _tag,
            "write_interval":     100,
            "checkpoint_interval": 2000,
            "wandb":              True,
            "wandb_kwargs": {
                "project":          "regolith-entrapment-rl",
                "group":            _wb_group,
                "name":             _wb_name,
                "tags":             _wb_tags,
                "config": {
                    "seed":         args_cli.seed,
                    "num_envs":     args_cli.num_envs,
                    "timesteps":    args_cli.timesteps,
                    "ablation":     ablation or None,
                    "gru_hidden":   GRU_HIDDEN,
                    "gru_layers":   GRU_LAYERS,
                    "seq_len":      SEQ_LEN,
                    "policy_obs_dim": POLICY_OBS_DIM,
                    "lr":           3e-4,
                    "entropy_scale": ENTROPY_SCALE,
                },
                "sync_tensorboard": True,
            },
        },
    })

    # Opt-in KL-adaptive LR (KL_ADAPTIVE_LR=1): standard PPO stabiliser (holds
    # the policy-update KL near a target by scaling lr) and the textbook remedy
    # for the seed-collapse mode seen in seeds 0/2. Off by default so the v12
    # retrain stays comparable to the documented config; enable if collapse
    # recurs despite the (now-functional) entropy floor.
    if os.environ.get("KL_ADAPTIVE_LR") == "1":
        from skrl.resources.schedulers.torch import KLAdaptiveLR
        ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        print("[train] KL-adaptive LR enabled (kl_threshold=0.008)")

    agent = PPO_RNN(
        models=models, memory=memory, cfg=ppo_cfg,
        observation_space=obs_space, action_space=act_space, device=device,
    )

    # Shrink reward tracking window to num_envs (default 100 is for large-env runs;
    # at 16 envs early episodes are long and the window fills too slowly to plot).
    import collections
    agent._track_rewards   = collections.deque(maxlen=args_cli.num_envs)
    agent._track_timesteps = collections.deque(maxlen=args_cli.num_envs)

    if args_cli.checkpoint:
        if not os.path.exists(args_cli.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
        print(f"Resuming from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # Patch: forward Isaac Lab extras["log"] entries to TensorBoard each rollout.
    # skrl's PPO_RNN doesn't auto-log these; we inject via agent.track_data().
    _raw_env = env.unwrapped
    _orig_post = agent.post_interaction

    # Entropy floor: if policy std drops below threshold before step 150k, boost
    # entropy coefficient temporarily until exploration recovers. Prevents the
    # silent early collapse seen in seeds 0&2 (std≈0.56, grind_rate→99%).
    _STD_FLOOR      = 0.7
    _STD_FLOOR_STEP = 150_000
    _ENT_BASE       = ppo_cfg["entropy_loss_scale"]          # 0.08
    _ENT_BOOSTED    = _ENT_BASE * 2.0                         # 0.16
    _ent_boosted    = False

    def _post_interaction_with_extras(timestep, timesteps):
        nonlocal _ent_boosted
        _orig_post(timestep=timestep, timesteps=timesteps)
        try:
            log = _raw_env.extras.get("log", {})
            for k, v in log.items():
                val = float(v.mean().item()) if hasattr(v, "mean") else float(v)
                agent.track_data(f"Info / {k}", val)
        except Exception:
            pass

        # Adaptive entropy floor check (only during early training)
        if timestep < _STD_FLOOR_STEP:
            try:
                current_std = float(agent.policy.log_std.exp().mean().item())
                # NOTE: must set the PRIVATE attribute — skrl's PPO_RNN caches
                # cfg["entropy_loss_scale"] into self._entropy_loss_scale at
                # __init__ and the update loop only reads the cached attribute.
                # Writing agent.cfg[...] alone is a silent no-op (bug found
                # 2026-06-12: the floor never actually fired in any run to date).
                if current_std < _STD_FLOOR and not _ent_boosted:
                    agent.cfg["entropy_loss_scale"] = _ENT_BOOSTED
                    agent._entropy_loss_scale = _ENT_BOOSTED
                    _ent_boosted = True
                    agent.track_data("Info / entropy_boost_active", 1.0)
                elif current_std >= _STD_FLOOR and _ent_boosted:
                    agent.cfg["entropy_loss_scale"] = _ENT_BASE
                    agent._entropy_loss_scale = _ENT_BASE
                    _ent_boosted = False
                    agent.track_data("Info / entropy_boost_active", 0.0)
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
