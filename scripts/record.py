"""
AAU Mars Rover — Wheel Entrapment Recovery — Video Recorder

Uses Isaac Sim's RTX renderer (headless, no GUI window) to capture
research-quality video and images of the rover in the regolith.

Usage:
    # Random actions — just visualise the environment
    ./launch.sh scripts/record.py --num_envs 1 --num_steps 500

    # Trained policy
    ./launch.sh scripts/record.py --num_envs 1 --num_steps 500 \\
        --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt

    # Side-view camera
    ./launch.sh scripts/record.py --num_envs 1 --camera side

    # Overhead / top-down camera
    ./launch.sh scripts/record.py --num_envs 1 --camera top

Output (default: recordings/):
    recordings/entrapment_<camera>.mp4   — full video at 25 fps
    recordings/frame_NNNN.png            — every 10th frame as PNG
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AAU Mars Rover — Video Recorder")
parser.add_argument("--num_envs",  type=int, default=1,         help="Envs to run (1 recommended)")
parser.add_argument("--num_steps", type=int, default=500,       help="Frames to record")
parser.add_argument("--out_dir",   type=str, default="recordings", help="Output directory")
parser.add_argument("--camera",    type=str, default="side",
                    choices=["side", "top", "front", "iso"],
                    help="Camera angle preset")
parser.add_argument("--width",     type=int, default=1920)
parser.add_argument("--height",    type=int, default=1080)
parser.add_argument("--fps",       type=int, default=25)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Trained checkpoint path (omit for random actions)")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
# Headless RTX rendering — no GUI window, but full quality render
args_cli.headless       = True
args_cli.enable_cameras = True   # forces SimulationApp + RTX renderer
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym

import omni.replicator.core as rep
from isaaclab_rl.skrl import SkrlVecEnvWrapper

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # registers AAURover-MarsEntrapment-v0
from envs.entrapment_env import EntrapmentEnvCfg


# ── Camera presets (relative to env origin) ────────────────────────────────────
# All coords are offsets added to env origin [0] after reset.
CAMERA_PRESETS = {
    # Good for showing wheels in sand, sand deformation on the side
    "side":  dict(pos_offset=( 0.0,  -3.5,  1.5), look_offset=( 0.0,  0.0,  0.1)),
    # 45-degree isometric — rover + surrounding sand visible
    "iso":   dict(pos_offset=( 2.5,  -2.5,  2.0), look_offset=( 0.0,  0.0,  0.15)),
    # Top-down — shows all 6 wheels and sand deformation
    "top":   dict(pos_offset=( 0.0,   0.0,  4.0), look_offset=( 0.0,  0.0,  0.0)),
    # Front-facing — shows rover head-on driving out of sand
    "front": dict(pos_offset=( 3.0,   0.0,  0.8), look_offset=( 0.0,  0.0,  0.2)),
}


def make_video(frames: list, out_path: str, fps: int):
    """Write frames (list of H×W×3 uint8 numpy arrays) to an MP4 file."""
    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def record():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env_cfg = EntrapmentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("AAURover-MarsEntrapment-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env.device
    num_act = env_cfg.action_space

    # ── Load policy if checkpoint given ───────────────────────────────────────
    agent = None
    if args_cli.checkpoint:
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train import PolicyNet, ValueNet

        obs_space = gym.spaces.Box(low=-math.inf, high=math.inf,
                                   shape=(env_cfg.observation_space,))
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))
        models = {
            "policy": PolicyNet(obs_space, act_space, device),
            "value":  ValueNet(obs_space, act_space, device),
        }
        memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)
        agent  = PPO(models=models, memory=memory, cfg=PPO_DEFAULT_CONFIG.copy(),
                     observation_space=obs_space, action_space=act_space, device=device)
        agent.load(args_cli.checkpoint)
        agent.set_running_mode("eval")
        print(f"[Record] Loaded checkpoint: {args_cli.checkpoint}")

    # ── First reset to get env origins ────────────────────────────────────────
    obs, _ = env.reset()
    simulation_app.update()  # let the stage settle

    origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    preset = CAMERA_PRESETS[args_cli.camera]
    cam_pos  = tuple(float(origin[i] + preset["pos_offset"][i])  for i in range(3))
    look_pos = tuple(float(origin[i] + preset["look_offset"][i]) for i in range(3))

    # ── Replicator camera ──────────────────────────────────────────────────────
    with rep.new_layer():
        camera = rep.create.camera(
            position=cam_pos,
            look_at=look_pos,
            focal_length=18.0,   # ~wide-angle for close-up rover shots
            focus_distance=4.0,
            f_stop=0.0,          # infinite depth of field — everything sharp
        )
        render_product = rep.create.render_product(
            camera, (args_cli.width, args_cli.height)
        )

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])

    # Warm up — let replicator initialise and RTX shaders compile
    print("[Record] Warming up RTX renderer (may take ~30s first run)...")
    for _ in range(5):
        simulation_app.update()

    # ── Recording loop ────────────────────────────────────────────────────────
    frames    = []
    video_tag = args_cli.camera
    print(f"\n{'='*55}")
    print(f"  Recording: {args_cli.num_steps} steps  |  camera: {args_cli.camera}")
    print(f"  Policy: {'trained' if agent else 'random actions'}")
    print(f"  Output: {args_cli.out_dir}/")
    print(f"{'='*55}\n")

    for step in range(args_cli.num_steps):
        # Actions
        if agent is not None:
            with torch.no_grad():
                actions, _, _ = agent.act(
                    {"states": obs}, timestep=step, timesteps=args_cli.num_steps
                )
        else:
            actions = torch.zeros(env.num_envs, num_act, device=device)
            actions[:, :6] = 0.5 + 0.1 * torch.randn(env.num_envs, 6, device=device)
            actions[:, 6:] = 0.08 * torch.randn(env.num_envs, 4, device=device)
            actions = actions.clamp(-1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(actions)

        # Render + grab frame
        simulation_app.update()
        raw = rgb_annot.get_data()
        if raw is not None and raw.size > 0:
            # raw is RGBA uint8 (H, W, 4)
            frames.append(raw[:, :, :3].copy())

        done = terminated | truncated
        if done.any():
            obs, _ = env.reset()

        if step % 50 == 0:
            log = info.get("log", {})
            vx  = float(log.get("mean_vx", torch.tensor(0.0)))
            esc = float(log.get("escape_rate", torch.tensor(0.0)))
            print(f"  step {step:4d}/{args_cli.num_steps}"
                  f"  |  frames captured: {len(frames)}"
                  f"  |  v_x={vx:.3f} m/s  escape={esc:.0%}")

    # ── Save output ────────────────────────────────────────────────────────────
    if not frames:
        print("[Record] No frames captured — check RTX renderer setup.")
        env.close()
        simulation_app.close()
        return

    import cv2

    # MP4 video
    video_path = os.path.join(args_cli.out_dir, f"entrapment_{video_tag}.mp4")
    make_video(frames, video_path, args_cli.fps)
    print(f"\n[Record] Video  → {video_path}  ({len(frames)} frames @ {args_cli.fps} fps)")

    # PNG keyframes (every 10th frame)
    saved_imgs = 0
    for i, frame in enumerate(frames[::10]):
        img_path = os.path.join(args_cli.out_dir, f"frame_{i:04d}_{video_tag}.png")
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        saved_imgs += 1
    print(f"[Record] Images → {args_cli.out_dir}/frame_*_{video_tag}.png  ({saved_imgs} files)")

    env.close()
    render_product.destroy()
    simulation_app.close()


if __name__ == "__main__":
    record()
