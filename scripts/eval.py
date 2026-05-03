import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mars Rover (6-wheel) — Evaluation")
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--episodes",   type=int, default=0,
                    help="Run N episodes then exit (0 = infinite / interactive)")
parser.add_argument("--no-mpm", action="store_true",
                    help="Skip MPM sand physics (viewer-only, shows rover without sand)")
parser.add_argument("--save-data",  type=str, default=None, metavar="PATH",
                    help="Save episode data to PATH (.npz) for offline plotting "
                         "with: python3 scripts/plot_episode.py --from-file PATH")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video via gym.wrappers.RecordVideo (saves to recordings/)")
parser.add_argument("--video-length", type=int, default=300,
                    help="Frames per video clip (default: 300 = ~12s at 25Hz)")
parser.add_argument("--video-interval", type=int, default=1,
                    help="Record every N episodes (default: 1 = record all)")
parser.add_argument("--num_trials", type=int, default=0,
                    help="Headless eval: run N total trials and exit. Alias for --episodes "
                         "but also enables success-rate aggregation when used with --out_json.")
parser.add_argument("--out_json", type=str, default=None,
                    help="If set, write {success_rate, num_trials, mean_reward} JSON here at exit.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()


os.environ["LAUNCH_OV_APP"] = "1"
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import gymnasium as gym

from isaaclab_rl.skrl import SkrlVecEnvWrapper

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs
from envs.entrapment_env import EntrapmentEnvCfg


def load_agent(device, num_obs, num_act, num_envs, checkpoint_path):
    from skrl.agents.torch.ppo import PPO_RNN
    from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from train import GRUPolicyNet, GRUValueNet, ROLLOUTS

    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))
    models = {
        "policy": GRUPolicyNet(obs_space, act_space, device, num_envs=num_envs),
        "value":  GRUValueNet(obs_space, act_space, device, num_envs=num_envs),
    }
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=num_envs, device=device)
    agent = PPO_RNN(models=models, memory=memory, cfg=PPO_RNN_DEFAULT_CONFIG.copy(),
                    observation_space=obs_space, action_space=act_space, device=device)
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent


def random_actions(num_envs, num_act, device):
    a = torch.zeros(num_envs, num_act, device=device)
    a[:, :6] = 0.4 + 0.1 * torch.randn(num_envs, 6, device=device)
    a[:, 6:] = 0.05 * torch.randn(num_envs, 4, device=device)
    return a.clamp(-1.0, 1.0)


def main():
    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    _ood_s = os.environ.get("OOD_SINKAGE")
    _ood_f = os.environ.get("OOD_FRICTION")
    if _ood_s is not None:
        env_cfg.sinkage_override = float(_ood_s)
        print(f"[Eval] OOD sinkage override: {env_cfg.sinkage_override} m")
    if _ood_f is not None:
        env_cfg.friction_override = float(_ood_f)
        print(f"[Eval] OOD friction override: {env_cfg.friction_override}")

    if args_cli.num_trials > 0 and args_cli.episodes == 0:
        args_cli.episodes = args_cli.num_trials
    if getattr(args_cli, 'no_mpm', False):
        env_cfg.skip_mpm = True

    if not args_cli.headless and not getattr(args_cli, 'no_mpm', False):

        print("[Eval] Viewer mode: consider --no-mpm if you get VRAM OOM")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("MarsRover-RegolithEscape-v0", cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        import numpy as np
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
        os.makedirs(out_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep: ep % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            name_prefix="entrapment",
        )
        print(f"[Eval] Video recording → {out_dir}/")

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env.device
    num_act = env_cfg.action_space


    agent = None
    if args_cli.checkpoint:
        agent = load_agent(device, env_cfg.observation_space, num_act,
                           env.num_envs, args_cli.checkpoint)
        print(f"[Eval] Loaded: {args_cli.checkpoint}")

    _viewer = getattr(env.unwrapped, "_viewer", None)

    mode = "trained" if agent else "random"
    print(f"\n{'='*55}")
    print(f"  Mars Rover (6-wheel) — Eval ({mode} actions)")
    print(f"  Envs: {env.num_envs}  |  Viewer: {'open' if _viewer else 'off'}")
    if args_cli.episodes > 0:
        print(f"  Episodes: {args_cli.episodes}")
    print(f"{'='*55}\n")

    obs, _ = env.reset()
    step = 0
    ep_count = 0
    ep_rewards = []
    ep_successes = 0  # episodes that terminated (escape) rather than timed out
    ep_reward = torch.zeros(env.num_envs, device=device)


    import numpy as np
    _SAVE_PATH = getattr(args_cli, "save_data", None)
    _EP_KEYS   = ["t", "wheel_vel", "drive_torque", "slip",
                  "entrap_flag", "torque_anomaly", "imu_acc",
                  "pos_xy", "reward", "action"]
    _VEL_LIM, _TRQ_LIM = 6.0, 3.0
    _DT = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    saved_episodes, _cur_ep = [], {k: [] for k in _EP_KEYS}
    _ep_step = 0
    _env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()[:2]

    def _collect_step(obs_t, act_t, rew_t):
        nonlocal _ep_step
        ob = obs_t[0].cpu().numpy()
        rp = env.unwrapped.root_pos[0].cpu().numpy()
        _cur_ep["t"].append(_ep_step * _DT)
        _cur_ep["wheel_vel"].append(ob[0:6]   * _VEL_LIM)
        _cur_ep["drive_torque"].append(ob[20:26] * _TRQ_LIM)
        _cur_ep["slip"].append(ob[6:12])
        _cur_ep["entrap_flag"].append(ob[26])
        _cur_ep["torque_anomaly"].append(ob[27])
        _cur_ep["imu_acc"].append(ob[16:19])
        _cur_ep["pos_xy"].append(rp[:2] - _env_origin)
        _cur_ep["reward"].append(float(rew_t[0]))
        _cur_ep["action"].append(act_t[0].cpu().numpy())
        _ep_step += 1

    def _end_episode():
        nonlocal _cur_ep, _ep_step, _env_origin
        if _cur_ep["t"]:
            saved_episodes.append({k: np.array(v) for k, v in _cur_ep.items()})
        _cur_ep   = {k: [] for k in _EP_KEYS}
        _ep_step  = 0
        _env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()[:2]


    try:
        while True:

            _viewer = getattr(env.unwrapped, "_viewer", None)
            if _viewer is not None and not _viewer.is_running():
                break


            if agent is not None:
                with torch.no_grad():
                    actions, _, _ = agent.act({"states": obs}, timestep=step, timesteps=0)
            else:
                actions = random_actions(env.num_envs, num_act, device)

            obs, reward, terminated, truncated, info = env.step(actions)
            ep_reward += reward.squeeze(-1) if reward.dim() > 1 else reward
            step += 1

            if _SAVE_PATH:
                _collect_step(obs, actions, reward)

            _term = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
            _trunc = truncated.squeeze(-1) if truncated.dim() > 1 else truncated
            done = _term | _trunc
            if done.any():
                for i in range(env.num_envs):
                    if done[i]:
                        ep_rewards.append(float(ep_reward[i]))
                        ep_count += 1
                        if bool(_term[i]) and not bool(_trunc[i]):
                            ep_successes += 1
                ep_reward[done] = 0.0
                if _SAVE_PATH and bool(done[0]):
                    _end_episode()
                obs, _ = env.reset()


            if step % 250 == 0:
                log = info.get("log", {})
                vx  = float(log.get("mean_vx", torch.tensor(0.0)))
                esc = float(log.get("escape_rate", torch.tensor(0.0)))
                print(f"  step={step:6d} | episodes={ep_count} | "
                      f"v_x={vx:.3f} | escape={esc:.0%}")


            if args_cli.episodes > 0 and ep_count >= args_cli.episodes:
                break

    except KeyboardInterrupt:
        print("\n[Eval] Stopped.")


    if _SAVE_PATH:
        if _cur_ep["t"]:
            _end_episode()
        if saved_episodes:
            out_dir = os.path.dirname(os.path.abspath(_SAVE_PATH))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            arrays = {
                "n_episodes": np.array([len(saved_episodes)]),
                "mode":       np.array([mode]),
            }
            for ei, ep in enumerate(saved_episodes):
                for k, v in ep.items():
                    arrays[f"ep{ei}_{k}"] = v
            np.savez(_SAVE_PATH, **arrays)
            print(f"\n[Eval] Episode data saved → {_SAVE_PATH}")
            print(f"       Plot offline with:")
            print(f"       python3 scripts/plot_episode.py --from-file {_SAVE_PATH}")


    if ep_rewards:
        import statistics
        print(f"\n{'='*55}")
        print(f"  Episodes: {len(ep_rewards)}")
        print(f"  Mean reward: {statistics.mean(ep_rewards):.3f}")
        print(f"  Std reward:  {statistics.stdev(ep_rewards):.3f}" if len(ep_rewards) > 1 else "")
        print(f"{'='*55}\n")

    if args_cli.out_json:
        import json as _json
        n = max(ep_count, 1)
        payload = {
            "num_trials": ep_count,
            "successes": ep_successes,
            "success_rate": ep_successes / n,
            "mean_reward": (sum(ep_rewards) / n) if ep_rewards else 0.0,
            "ood_sinkage": _ood_s,
            "ood_friction": _ood_f,
            "checkpoint": args_cli.checkpoint,
        }
        _outp = os.path.abspath(args_cli.out_json)
        os.makedirs(os.path.dirname(_outp) or ".", exist_ok=True)
        with open(_outp, "w") as f:
            _json.dump(payload, f, indent=2)
        print(f"[Eval] Wrote {_outp}: {payload}")

    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
