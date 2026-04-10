# Regolith Wheel Entrapment Recovery

**RL-trained escape policy for Mars rovers stuck in granular regolith**  
Newton MPM + Isaac Lab + skrl PPO with LSTM, dual-sensor detection, curriculum learning. Designed to integrate with a high-level navigator.

## [-] Project Structure
```
regolith_entrapment_research/
├── README.md                 # This file
├── .gitignore               # Git exclusions
├── launch.sh                # Main executor (sources paths.sh)
├── view.sh                  # Standalone Newton viewer
├── paths.py                 # Python path configuration (env var overrides)
├── paths.sh                 # Bash path configuration (env var overrides)
│
├── configs/                 # Configuration
│   └── ppo_mars_rover_v1.yaml  # PPO hyperparameters
│
├── envs/                    # Environment (core)
│   ├── entrapment_env.py    # RL environment with MPM coupling
│   ├── mpm_kernels.py       # Newton-MPM coupling kernels
│   └── __init__.py
│
├── robots/                  # Robot model & USD assets
│   ├── mars_rover_cfg.py    # AAU 6-wheel rocker-bogie config
│   ├── Mars_Rover.usd       # Main rover USD
│   ├── SubUSDs/             # Rover sub-assets (materials, textures)
│   └── __init__.py
│
├── scripts/                 # Executables
│   ├── train.py             # PPO training
│   ├── eval.py              # Evaluation & visualization (Newton ViewerGL)
│   ├── record.py            # RTX headless recording (MP4/PNG)
│   ├── plot_episode.py      # Episode dashboard plotting
│   ├── plot_training.py     # Training curve visualization
│   └── view_rover.py        # Standalone Newton viewer
│
├── sim2real/                # Sim-to-real prep
│   ├── onnx_export/
│   │   └── export_model.py  # Export policy to ONNX (29D obs, 10D actions)
│   └── rpi5_deploy/
│       └── rover_controller.py  # RPi5 inference loop
│
├── detection/               # Phase 1: Sinkage detection (CNN-GRU)
│   ├── models/
│   │   └── cnn_gru.py       # Detector architecture
│   └── scripts/
│       └── train_detector.py  # Train sinkage detector
│
├── terrain/                 # Terrain configs
└── stubs/                   # External library stubs
```

## [-] What This Solves

This repository provides a **standalone escape policy subsystem** trained to detect and recover from wheel entrapment in granular regolith. It is *designed* to plug into a high-level navigator:

```
[High-Level Navigator] 
    → Monitors slip/torque sensors 
    → On entrapment detection: SWITCHES to [Escape Policy] 
    → Escape policy executes recovery maneuvers 
    → On escape (>1.5m from origin): RETURNS control to navigator 
    → Navigator resumes point A → B path following
```

**What's included**:
- [x] Trained escape policy (`*.pt` checkpoints) - 10D action space (6 drive + 4 steer)
- [x] **Recurrent PPO (GRU)**: policy carries temporal memory across steps (GRU hidden 256, seq_len 16)
- [x] Dual entrapment detection: slip-based (low v_x + high slip) + torque-based (high motor torque)
- [x] Observation space: 29D (wheel states, slip, IMU, joint torques, entrapment/torque flags, escape progress)
- [x] Reward shaping: progress + shaped escape - penalties + rocking bonus
- [x] Curriculum learning: sinkage depth increases with training progress
- [x] Visualization: Newton ViewerGL with particle rendering and debug overlays
- [x] Configurable paths via environment variables

**What's required for integration** (external wrapper):
- Logic to switch between navigator and escape policy based on entrapment detection
- Waypoint tracking to resume navigation post-escape
- (Optional) Shared terrain context for escape direction selection

## [-] Installation Guide

### 1. Install Dependencies

```bash
# Newton Physics Engine
git clone https://github.com/NVIDIA-Newton/newton.git ~/newton

# Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab

# Isaac Sim (download from NVIDIA Omniverse Launcher)

# RLRoverLab Assets (optional, for Mars terrain)
git clone https://github.com/abmoRobotics/RLRoverLab.git ~/RLRoverLab
```

### 2. Create Conda Environment

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

### 3. Install Python Packages

```bash
pip install torch torchvision torchaudio gymnasium wandb scikit-learn
pip install pyglet imgui_bundle  # For Newton ViewerGL
pip install onnxruntime          # For ONNX export/deployment
```

### 4. Configure Paths (Optional)

Default paths assume the following locations. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ISAAC_SIM_PATH` | `~/isaac-sim` | Isaac Sim installation |
| `ISAACLAB_SRC_PATH` | `~/IsaacLab/source` | Isaac Lab source directory |
| `NEWTON_PATH` | `~/newton` | Newton installation |
| `CONDA_ENV_PATH` | `~/miniconda3/envs/env_isaaclab` | Conda environment path |
| `RLROVER_ASSETS` | `~/RLRoverLab/rover_envs/assets` | RLRoverLab assets directory |
| `PXR_EXT_PATH` | (auto-detected) | PXR extension path |

Example:
```bash
export ISAAC_SIM_PATH=/custom/path/to/isaac-sim
export NEWTON_PATH=/custom/path/to/newton
./launch.sh scripts/train.py --num_envs 64
```

### 5. Verify Installation

```bash
./launch.sh scripts/train.py --num_envs 1 --timesteps 100
```

## [-] Quick Start

### GUI Mode (Newton ViewerGL — needs DISPLAY)

Opens a live 3D viewer window with the rover + sand particles rendered in real-time.
Isaac Sim's GUI is broken in conda-Python; all visualization uses Newton ViewerGL instead.

```bash
# GUI eval with random actions (sanity check — no checkpoint needed)
./launch.sh scripts/eval.py --num_envs 1

# GUI eval with trained policy
./launch.sh scripts/eval.py --num_envs 1 \
    --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

# GUI eval without MPM sand (faster, viewer-only)
./launch.sh scripts/eval.py --num_envs 1 --no-mpm

# Standalone rover viewer (Newton only, no Isaac Lab / no RL)
./view.sh
./view.sh --no-sand
```

### Headless Mode (no display needed)

Runs without any GUI window. Use for training, batch evaluation, and video recording.

```bash
# Train with 64 parallel environments (headless, ~200k steps)
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000

# Smoke test (quick sanity check)
./launch.sh scripts/train.py --num_envs 4 --timesteps 500

# Resume from checkpoint
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000 \
    --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

# Headless evaluation (metrics only, no viewer)
./launch.sh scripts/eval.py --num_envs 64 --headless --episodes 50 \
    --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

# RTX headless rendering → MP4 video
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500 \
    --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt

# Monitor training (separate terminal)
tensorboard --logdir experiments/

# Generate training curve plots
python3 scripts/plot_training.py --compare
```

### Offline Plotting (save episode data, plot later without Isaac Sim)

```bash
# Save episode data during eval
./launch.sh scripts/eval.py --num_envs 1 --episodes 5 \
    --checkpoint experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt \
    --save-data experiments/regolith_recovery/episode_data/run1.npz

# Plot saved data anywhere (no Isaac Sim needed)
conda run -n env_isaaclab python3 scripts/plot_episode.py \
    --from-file experiments/regolith_recovery/episode_data/run1.npz
```

### Background Processes & Job Control

Training and eval can run for hours. Use shell job control to manage them:

```bash
# Run training in background
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000 &

# List background jobs (shows job numbers like [1], [2], etc.)
jobs

# Kill a background job by number
kill %1        # kill job [1]
kill %2        # kill job [2]
kill %%        # kill the most recent background job

# Bring a background job to foreground
fg %1

# Send foreground job to background
# Press Ctrl+Z first (suspends), then:
bg %1

# Kill by PID (if you know it)
kill 12345

# Find PIDs for training processes
ps aux | grep train.py
```

## [-] ONNX Export & Deployment

### Export Trained Policy

```bash
# Export to ONNX (29D obs, 10D actions for 6-wheel Mars rover)
python sim2real/onnx_export/export_model.py \
    --policy_ckpt experiments/regolith_recovery/ppo_regolith/checkpoints/best_agent.pt \
    --out_dir sim2real/onnx_export/output \
    --num_obs 29 --num_actions 10
```

### Deploy on Hardware (RPi5)

**Important**: The rover controller supports configurable dimensions for different robot configurations:

```bash
# For 4-wheel differential drive (default: 12D obs, 4D actions)
python sim2real/rpi5_deploy/rover_controller.py \
    --policy_onnx recovery_policy.onnx \
    --detector_onnx sinkage_detector.onnx \
    --num_obs 12 --num_actions 4 \
    --run_time 300

# For 6-wheel Mars rover (29D obs, 10D actions)
python sim2real/rpi5_deploy/rover_controller.py \
    --policy_onnx recovery_policy.onnx \
    --detector_onnx sinkage_detector.onnx \
    --num_obs 29 --num_actions 10 \
    --run_time 300
```

**Note**: You must retrain the policy with appropriate observation/action dimensions for your specific hardware. The 6-wheel Mars rover simulation uses 29D observations and 10D actions.

## [-] Key Metrics to Watch (TensorBoard)

| Metric | Expected Behavior |
|--------|-------------------|
| `reward/mean` | Should trend upward |
| `episode/escape_rate` | Should increase from 0% |
| `Info/curriculum_progress` | Should increase 0→1.0 over training |
| `Info/entrap_flag_rate` | Should stabilize (not 0 or 1) |
| `Info/torque_anomaly_rate` | Should stabilize (not 0 or 1) |
| `entropy` | Should stay > 0.02 (prevents collapse) |

## [-] Core Enhancements

| Component | Improvement | Purpose |
|-----------|-------------|---------|
| **Policy Network** | Recurrent PPO + GRU (encoder 128→GRU 256→head 64) | Temporal memory for rocking maneuvers + sustained entrapment |
| **Entrapment Detection** | Dual-sensor (slip + torque) | More robust than slip-only |
| **Reward Function** | Shaped escape + abnormal penalty + rocking bonus | Better learning signals |
| **Curriculum Learning** | Episode-based sinkage increase | Consistent challenge as policy improves |
| **Visualization** | Newton ViewerGL with particle rendering | Real-time debugging |
| **Configuration** | Environment variable overrides | Portable across systems |

## [-] Environment Details

- **Python**: 3.11 (conda: env_isaaclab)
- **Physics (training)**: Isaac Lab `XPBOSolverCfg` (PhysX rigid bodies) + Newton `SolverImplicitMPM` (granular sand)
- **Physics (viewer)**: Newton `SolverMuJoCo` (rigid bodies) + Newton `SolverImplicitMPM` (sand)
- **Solver**: 4 substeps, 50 Hz physics / 25 Hz policy
- **Note**: Standalone viewer uses MuJoCo solver — Newton's XPBD cannot stably support an articulated rover on a ground plane (329 mesh collision shapes → contact buffer overflow). Mesh collision disabled; 6 invisible proxy spheres on wheel bodies used instead.
- **MPM**: voxel_size=0.05m, sand 1.2m×1.2m×0.15m, µ=0.7

**Observation Space (29D)**:
- `wheel_vel` (6) — drive joint velocities normalized by 6 rad/s
- `slip` (6) — per-wheel slip ratio
- `steer_pos` (4) — steering joint angles normalized by 0.6 rad
- `imu_acc` (3) — linear acceleration / g
- `gravity_z` (1) — tilt indicator
- `drive_torque` (6) — normalized drive joint torques
- `entrap_flag` (1) — binary entrapment indicator
- `torque_anomaly` (1) — sustained high-torque anomaly flag
- `dist_norm` (1) — distance from origin / escape threshold

**Action Space (10D)**:
- `drive_cmd` (6) — velocity targets [-1,1] → ±6 rad/s
- `steer_cmd` (4) — position targets [-1,1] → ±0.6 rad

## [-] Known Issues

- **Isaac Sim GUI broken in conda-Python**: Use Newton ViewerGL instead (`eval.py`, `view_rover.py`)
- **First-run startup time**: 10-15 min for Isaac Sim extension loading + Newton USD build + Warp JIT
- **Heightmap extraction disabled**: CUDA memory issues; use offline plotting mode

## [-] Documentation

- **[Configs]**: `configs/ppo_mars_rover_v1.yaml` — PPO hyperparameters
- **[Code]**: `envs/entrapment_env.py` — Main environment with detailed comments
- **[Paths]**: `paths.py` / `paths.sh` — Configurable path system

## [-] Troubleshooting

### OpenGL Context Error (Linux/Wayland)
```bash
export PYOPENGL_PLATFORM=glx
```

### CUDA Out of Memory
Reduce `--num_envs` or use `--no-mpm` flag for viewer-only mode.

### Missing PXR Extension
Set `PXR_EXT_PATH` environment variable to the correct path:
```bash
export PXR_EXT_PATH=~/.local/share/ov/data/exts/v2/omni.usd.libs-<hash>
```

---

*Last updated: April 2026*  
*Questions? Open an issue on GitHub.*
