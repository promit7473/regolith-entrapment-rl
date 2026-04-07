# Regolith Wheel Entrapment Recovery

**RL-trained escape policy for Mars rovers stuck in granular regolith**  
Newton MPM + Isaac Lab + skrl PPO with LSTM, dual-sensor detection, curriculum learning. Designed to integrate with a high-level navigator.

## [-] Project Structure
```
regolith_entrapment_research/
├── README.md                 # This file
├── .gitignore               # Git exclusions
├── launch.sh                # Main executor
├── view.sh                  # Alternative viewer
│
├── configs/                 # Configuration
│   └── ppo_aau_v2.yaml     # PPO hyperparameters (LSTM, rewards, curriculum)
│
├── envs/                    # Environment (core)
│   ├── entrapment_env.py   # Enhanced RL environment (viz: color coding, sand viz, action vectors)
│   ├── mpm_kernels.py      # Newton-MPM coupling kernels
│   └── __init__.py
│
├── robots/                  # Robot model
│   ├── aau_rover_cfg.py    # AAU 6-wheel rocker-bogie config
│   └── __init__.py
│
├── scripts/                 # Executables
│   ├── train.py            # PPO training
│   ├── eval.py             # Evaluation & visualization (Newton ViewerGL)
│   ├── record.py           # RTX headless recording (MP4/PNG)
│   └── view_rover.py       # Standalone Newton viewer
│
├── sim2real/                # Sim-to-real prep
│   ├── onnx_export.py      # Export policy to ONNX
│   ├── rpi5_controller.py  # RPi5 inference loop
│   └── [related files]
│
├── related_papers/          # References
│   └── Escape and path point tracking methods of skid-steered mobile...
│
├── detection/               # Phase 1: Sinkage detection (CNN-GRU)
│   └── [files]
│
├── terrain/                 # Terrain configs
└── stubs/                   # External library stubs
```

## [-] What This Solves
This repository provides a **standalone escape policy subsystem** trained to detect and recover from wheel entrapment in granular regolith. It is *designed* to plug into a high-level navigator:

```
[High-Level Navigator] 
    → Monitors slip/torque sensors 
    → On entrapment detection: SWITCHES to [YOUR Escape Policy] 
    → Escape policy executes recovery maneuvers 
    → On escape (>1.5m from origin): RETURNS control to navigator 
    → Navigator resumes point A → B path following
```

**What's included**:
- [x] Trained escape policy (`*.pt` checkpoints) - 10D action space (6 drive + 4 steer)
- [x] Dual entrapment detection: slip-based (low v_x + high slip) + torque-based (high motor torque)
- [x] Observation space: 29D (wheel states, slip, IMU, joint torques, entrapment/torque flags, escape progress)
- [x] Reward shaping: progress + shaped escape - penalties + rocking bonus
- [x] Curriculum learning: sinkage depth increases with training progress
- [x] Visualization enhancements: entrapment state color coding (red/orange), enhanced sand viz, action vectors

**What's required for integration** (external wrapper):
- Logic to switch between navigator and escape policy based on entrapment detection
- Waypoint tracking to resume navigation post-escape
- (Optional) Shared terrain context for escape direction selection

## [-] Installation Guide
Follow these steps to set up the environment:

1. **Install Dependencies**:
   - **Newton Physics Engine**: `git clone https://github.com/NVIDIA-Newton/newton.git ~/newton` + follow its installation guide
   - **Isaac Lab**: `git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab` + follow its installation guide  
   - **Isaac Sim**: Download from NVIDIA Omniverse Launcher
   - **RLRoverLab Assets**: `git clone https://github.com/abmoRobotics/RLRoverLab.git ~/RLRoverLab`

2. **Create Conda Environment**:
   ```bash
   conda create -n env_isaaclab python=3.11
   conda activate env_isaaclab
   ```

3. **Install Python Packages**:
   ```bash
   pip install torch torchvision torchaudio gymnasium wandb scikit-learn
   ```

4. **Verify Installation**:
   ```bash
   ./launch.sh --help
   ```

## [-] Quick Start

### Headless (no display needed — training server / SSH)
```bash
# Train — always headless by default (LAUNCH_OV_APP=1 set internally)
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000

# Headless eval — metrics only, no viewer window
./launch.sh scripts/eval.py --num_envs 64 --headless --episodes 50 \
    --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v2/checkpoints/best_agent.pt

# Smoke test (quick sanity check — 4 envs, 500 steps)
./launch.sh scripts/train.py --num_envs 4 --timesteps 500

# Monitor training
tensorboard --logdir experiments/

# Generate training curve plots (no Isaac Sim needed)
python3 scripts/plot_training.py --compare
```

### With display (Newton ViewerGL — needs DISPLAY)
```bash
# Evaluate with live 3D viewer
./launch.sh scripts/eval.py --num_envs 1 \
    --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v2/checkpoints/best_agent.pt

# Record video (RTX headless rendering → MP4)
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500 \
    --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v2/checkpoints/best_agent.pt
```

## [-] Key Metrics to Watch (TensorBoard)
- `reward/mean` → Should trend upward
- `episode/escape_rate` → Should increase from 0%  
- `extras/log/curriculum_progress` → Should increase 0→1.0 over training
- `extras/log/entrap_flag_rate` & `extras/log/torque_anomaly_rate` → Should stabilize (not 0 or 1)
- `entropy` → Should stay > 0.02 (prevents collapse)

## [-] Core Enhancements (vs. baseline)
| Component | Improvement | Purpose |
|-----------|-------------|---------|
| **Policy Network** | LSTM layer (128 units) | Temporal modeling of proprioceptive data |
| **Entrapment Detection** | Dual-sensor (slip + torque) | More robust than slip-only |
| **Reward Function** | Shaped escape + abnormal penalty + rocking bonus | Better learning signals |
| **Curriculum Learning** | Episode-based sinkage increase | Consistent challenge as policy improves |
| **Visualization** | Entrapment state color coding, enhanced sand viz, action vectors | Debugging & validation |

## [-] Documentation
- **[Configs]** - `configs/ppo_aau_v2.yaml` shows all hyperparameters
- **[Code]** - `envs/entrapment_env.py` contains all enhancements (well-commented)

## [-] Notes
- **Visualization**: Use Newton ViewerGL (via `eval.py`/`view_rover.py`), *not* Isaac Sim GUI (known broken in conda-Python)
- **Experiments/outputs**: Auto-gitignored (see `.gitignore`)
- **SSH key**: Your key `merajhossainpromit@gmail.com` is configured for GitHub pushes

*Last updated: April 2026*  
*Questions? Open an issue on GitHub.*