# Regolith Wheel Entrapment Recovery

**RL-trained escape policy for Mars rovers stuck in granular regolith**  
Newton MPM + Isaac Lab + skrl PPO with LSTM, dual-sensor detection, curriculum learning. Designed to integrate with RLRoverLab navigator.

## 📁 Project Structure
```
regolith_entrapment_research/
├── README.md                 # This file
├── CLAUDE.md                 # Quick reference & technical notes
├── ROADMAP.md                # Development plan
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

## 🔬 What This Solves
This repository provides a **standalone escape policy subsystem** trained to detect and recover from wheel entrapment in granular regolith. It is *designed* to plug into the RLRoverLab navigator:

```
[RLRoverLab Navigator] 
    → Monitors slip/torque sensors 
    → On entrapment detection: SWITCHES to [YOUR Escape Policy] 
    → Escape policy executes recovery maneuvers 
    → On escape (>1.5m from origin): RETURNS control to navigator 
    → Navigator resumes point A → B path following
```

**What's included**:
- ✅ Trained escape policy (`*.pt` checkpoints) - 10D action space (6 drive + 4 steer)
- ✅ Dual entrapment detection: slip-based (low v_x + high slip) + torque-based (high motor torque)
- ✅ Observation space: 28D (wheel states, slip, IMU, joint torques, entrapment/torque flags)
- ✅ Reward shaping: progress + shaped escape - penalties + rocking bonus
- ✅ Curriculum learning: sinkage depth increases with training progress
- ✅ Visualization enhancements: entrapment state color coding (red/orange), enhanced sand viz, action vectors

**What's required for integration** (external wrapper):
- Logic to switch between navigator and escape policy based on entrapment detection
- Waypoint tracking to resume navigation post-escape
- (Optional) Shared terrain context for escape direction selection

## 🚀 Quick Start
```bash
# 1. Train (64 envs, 200k timesteps - adjust for your GPU)
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000

# 2. Evaluate with visualization (Newton ViewerGL)
./launch.sh scripts/eval.py --num_envs 1 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt

# 3. Record video (RTX headless)
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt

# 4. Monitor training
tensorboard --logdir experiments/
```

## 📈 Key Metrics to Watch (TensorBoard)
- `reward/mean` → Should trend upward
- `episode/escape_rate` → Should increase from 0%  
- `extras/log/curriculum_progress` → Should increase 0→1.0 over training
- `extras/log/entrap_flag_rate` & `extras/log/torque_anomaly_rate` → Should stabilize (not 0 or 1)
- `entropy` → Should stay > 0.02 (prevents collapse)

## ⚙️ Core Enhancements (vs. baseline)
| Component | Improvement | Purpose |
|-----------|-------------|---------|
| **Policy Network** | LSTM layer (128 units) | Temporal modeling of proprioceptive data |
| **Entrapment Detection** | Dual-sensor (slip + torque) | More robust than slip-only |
| **Reward Function** | Shaped escape + abnormal penalty + rocking bonus | Better learning signals |
| **Curriculum Learning** | Episode-based sinkage increase | Consistent challenge as policy improves |
| **Visualization** | Entrapment state color coding, enhanced sand viz, action vectors | Debugging & validation |

## 📖 Documentation
- **[CLAUDE.md]** - Quick reference & technical notes
- **[ROADMAP.md]** - Development phases & planned features
- **[Configs]** - `configs/ppo_aau_v2.yaml` shows all hyperparameters
- **[Code]** - `envs/entrapment_env.py` contains all enhancements (well-commented)

## 🛠️ Setup
1. Install dependencies: Newton, Isaac Lab, Isaac Sim, RLRoverLab (see `CLAUDE.md` for details)
2. Create conda env: `conda create -n env_isaaclab python=3.11 && conda activate env_isaaclab`
3. Install packages: `pip install torch torchvision torchaudio gymnasium wandb scikit-learn`
4. Verify: `./launch.sh --help`

## ⚠️ Notes
- **Visualization**: Use Newton ViewerGL (via `eval.py`/`view_rover.py`), *not* Isaac Sim GUI (known broken in conda-Python)
- **Experiments/outputs**: Auto-gitignored (see `.gitignore`)
- **SSH key**: Your key `merajhossainpromit@gmail.com` is configured for GitHub pushes

*Last updated: April 2026*  
*Questions? Check CLAUDE.md or open an issue on GitHub.*