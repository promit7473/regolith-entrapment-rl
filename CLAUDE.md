# Regolith Wheel Entrapment Recovery Research

## What This Is
RL-trained Mars rover that detects + recovers from wheel entrapment in granular
regolith (sand/soil). Uses Newton MPM for particle physics + Isaac Lab for robot
simulation + skrl PPO for training.

## Quick Reference

### Run training (5070Ti — 64 envs, 200k steps)
```bash
cd ~/regolith_entrapment_research
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000
```

### Run eval with viewer (needs DISPLAY)
```bash
./launch.sh scripts/eval.py --num_envs 1
./launch.sh scripts/eval.py --num_envs 4 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt
```

### Record video (headless RTX rendering)
```bash
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500
```

### TensorBoard
```bash
tensorboard --logdir experiments/
```

## Architecture

```
Robot:  AAU 6-wheel rocker-bogie rover (Mars_Rover.usd from RLRoverLab)
Physics: Newton XPBD (rigid bodies) + SolverImplicitMPM (granular sand)
Coupling: Two-way — sand impulses → body forces, wheel SDF → sand collision
RL:     PPO via skrl, 64 parallel envs, 50Hz physics / 25Hz policy
```

**Obs (27D)**: wheel_vel(6) + slip(6) + steer_pos(4) + imu_acc(3) + gravity_z(1) + drive_torque(6) + entrap_flag(1)
**Act (10D)**: drive_cmd(6) ±6 rad/s + steer_cmd(4) ±0.6 rad

## Repo Layout

```
scripts/
  train.py              # PPO training (main entry point)
  eval.py               # evaluate checkpoint or run random actions
  record.py             # RTX headless → MP4 + PNG
envs/
  entrapment_env.py     # DirectRLEnv with full MPM coupling
  mpm_kernels.py        # 3 Warp kernels (body forces, double-count fix, reset)
  __init__.py            # registers AAURover-MarsEntrapment-v0
robots/
  aau_rover_cfg.py      # ArticulationCfg for AAU Mars rover
configs/
  ppo_aau_v2.yaml       # hyperparameters (reference)
detection/              # Phase 1: sinkage detection (CNN-GRU)
sim2real/               # Phase 3: ONNX export + RPi5 controller
experiments/            # training outputs (auto-created, gitignored)
archived/               # old Jackal code, ManagerBased prototype
```

## Critical Technical Notes

### AppLauncher standalone mode
Isaac Lab's AppLauncher SKIPS SimulationApp when `headless=True` unless one of:
- `--enable_cameras` is set (preferred)
- `LAUNCH_OV_APP=1` env var
- `--visualizer omniverse`
All our scripts set `args_cli.enable_cameras = True` to force SimulationApp creation.

### Isaac Sim GUI is broken
`omni.kit.actions.core` doesn't register in conda-Python. The viewport is always
blank. Use Newton ViewerGL for live preview (via eval.py) or RTX headless for
video (via record.py). Do NOT use `--visualizer omniverse`.

### Newton quaternion convention
`[x, y, z, qx, qy, qz, qw]` — w at index 6 (NOT wxyz like PhysX).

### Ground plane
`spawn_ground_plane()` fails (needs remote Nucleus). Use Newton callback:
```python
NewtonManager.add_on_init_callback(lambda: NewtonManager._builder.add_ground_plane())
```

### Warp version
Must use `warp-lang>=1.11.0`. Newton's FEM interpolate needs `at`/`reduction` params.

### First-run startup time
First launch takes 10-15 min: Isaac Sim extension loading + Newton USD scene build +
Warp kernel JIT compilation. Subsequent runs reuse caches and are much faster (~1-2 min).
The scene clone step is the slowest part (AAU rover USD is complex). Monitor with
`PYTHONUNBUFFERED=1` to see progress in real-time. CPU at 100%+ during this phase is
normal (not a hang).

## Reward v2 (current)
| Term | Weight | Formula |
|------|--------|---------|
| r_progress | 2.0 | v_x × dt |
| r_escape | 5.0 | time_scale (once when >1.5m from origin) |
| p_slip | -1.0 | mean_slip × dt |
| p_tilt | -0.3 | ‖ang_vel_xy‖₂ × dt |
| p_smooth | -0.05 | ‖Δaction‖₂ × dt |

## PPO hyperparams
entropy_loss_scale=0.02 (prevents collapse), rollouts=24, lr=3e-4,
mini_batches=4, learning_epochs=5, discount=0.99, lambda=0.95

## Environment Details
- Python: 3.11 (conda: env_isaaclab)
- GPU: RTX 5070 Ti (training at 64 envs)
- Solver: XPBOSolverCfg, 4 iterations, 4 substeps
- MPM: voxel_size=0.05, sand 1.2m×1.2m×0.15m, µ=0.7

## External Dependencies
- Newton: ~/newton/
- Isaac Lab: ~/IsaacLab/source/
- Isaac Sim: ~/isaac-sim/
- RLRoverLab assets: ~/RLRoverLab/rover_envs/assets/
- pxr ext: ~/.local/share/ov/data/exts/v2/omni.usd.libs-*/
