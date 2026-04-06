"""
AAU Mars Rover — Evaluation / Live Preview

Runs the trained policy (or random actions) with Newton ViewerGL for
live 3D visualization, and prints episode metrics.

Usage:
    # Random actions (sanity check)
    ./launch.sh scripts/eval.py --num_envs 1

    # Trained checkpoint
    ./launch.sh scripts/eval.py --num_envs 4 --checkpoint experiments/.../best_agent.pt

    # Headless evaluation (no viewer, just metrics)
    ./launch.sh scripts/eval.py --num_envs 64 --checkpoint <path> --headless --episodes 50
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AAU Mars Rover — Evaluation")
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--episodes",   type=int, default=0,
                    help="Run N episodes then exit (0 = infinite / interactive)")
parser.add_argument("--no-mpm", action="store_true",
                    help="Skip MPM sand physics (viewer-only, shows rover without sand)")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
# Force SimulationApp creation without heavy RTX rendering stack.
# LAUNCH_OV_APP=1 avoids "standalone mode" (no SimulationApp) without loading
# viewport/replicator extensions. Newton ViewerGL handles visualisation.
import os as _os
_os.environ["LAUNCH_OV_APP"] = "1"
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import torch
import gymnasium as gym

from isaaclab_rl.skrl import SkrlVecEnvWrapper

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # registers AAURover-MarsEntrapment-v0
from envs.entrapment_env import EntrapmentEnvCfg


def load_agent(device, num_obs, num_act, num_envs, checkpoint_path):
    """Load a trained PPO agent from checkpoint."""
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from train import PolicyNet, ValueNet

    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))
    models = {
        "policy": PolicyNet(obs_space, act_space, device),
        "value":  ValueNet(obs_space, act_space, device),
    }
    memory = RandomMemory(memory_size=1, num_envs=num_envs, device=device)
    agent = PPO(models=models, memory=memory, cfg=PPO_DEFAULT_CONFIG.copy(),
                observation_space=obs_space, action_space=act_space, device=device)
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent


def random_actions(num_envs, num_act, device):
    """Gentle forward drive with slight random steering."""
    a = torch.zeros(num_envs, num_act, device=device)
    a[:, :6] = 0.4 + 0.1 * torch.randn(num_envs, 6, device=device)
    a[:, 6:] = 0.05 * torch.randn(num_envs, 4, device=device)
    return a.clamp(-1.0, 1.0)


def main():
    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, 'no_mpm', False):
        env_cfg.skip_mpm = True
    # For viewer mode (non-headless with MPM), limit VRAM usage
    if not args_cli.headless and not getattr(args_cli, 'no_mpm', False):
        # Override to limit memory: restrict max_nodes for collision impulse buffer
        print("[Eval] Viewer mode: consider --no-mpm if you get VRAM OOM")

    env = gym.make("AAURover-MarsEntrapment-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env.device
    num_act = env_cfg.action_space

    # Load trained policy if checkpoint given
    agent = None
    if args_cli.checkpoint:
        agent = load_agent(device, env_cfg.observation_space, num_act,
                           env.num_envs, args_cli.checkpoint)
        print(f"[Eval] Loaded: {args_cli.checkpoint}")

    _viewer = getattr(env.unwrapped, "_viewer", None)

    mode = "trained" if agent else "random"
    print(f"\n{'='*55}")
    print(f"  AAU Mars Rover — Eval ({mode} actions)")
    print(f"  Envs: {env.num_envs}  |  Viewer: {'open' if _viewer else 'off'}")
    if args_cli.episodes > 0:
        print(f"  Episodes: {args_cli.episodes}")
    print(f"{'='*55}\n")

    obs, _ = env.reset()
    step = 0
    ep_count = 0
    ep_rewards = []
    ep_reward = torch.zeros(env.num_envs, device=device)

    try:
        while True:
            # Check viewer
            _viewer = getattr(env.unwrapped, "_viewer", None)
            if _viewer is not None and not _viewer.is_running():
                break

            # Act
            if agent is not None:
                with torch.no_grad():
                    actions, _, _ = agent.act({"states": obs}, timestep=step, timesteps=0)
            else:
                actions = random_actions(env.num_envs, num_act, device)

            obs, reward, terminated, truncated, info = env.step(actions)
            ep_reward += reward.squeeze(-1) if reward.dim() > 1 else reward
            step += 1

            _term = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
            _trunc = truncated.squeeze(-1) if truncated.dim() > 1 else truncated
            done = _term | _trunc
            if done.any():
                for i in range(env.num_envs):
                    if done[i]:
                        ep_rewards.append(float(ep_reward[i]))
                        ep_count += 1
                ep_reward[done] = 0.0
                obs, _ = env.reset()

            # Periodic log
            if step % 250 == 0:
                log = info.get("log", {})
                vx  = float(log.get("mean_vx", torch.tensor(0.0)))
                esc = float(log.get("escape_rate", torch.tensor(0.0)))
                print(f"  step={step:6d} | episodes={ep_count} | "
                      f"v_x={vx:.3f} | escape={esc:.0%}")

            # Episode limit
            if args_cli.episodes > 0 and ep_count >= args_cli.episodes:
                break

    except KeyboardInterrupt:
        print("\n[Eval] Stopped.")

    # Summary
    if ep_rewards:
        import statistics
        print(f"\n{'='*55}")
        print(f"  Episodes: {len(ep_rewards)}")
        print(f"  Mean reward: {statistics.mean(ep_rewards):.3f}")
        print(f"  Std reward:  {statistics.stdev(ep_rewards):.3f}" if len(ep_rewards) > 1 else "")
        print(f"{'='*55}\n")

    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
