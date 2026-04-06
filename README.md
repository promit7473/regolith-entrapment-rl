# Regolith Wheel Entrapment Recovery Research

RL-trained Mars rover that detects + recovers from wheel entrapment in granular regolith (sand/soil). Uses Newton MPM for particle physics + Isaac Lab for robot simulation + skrl PPO for training.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Reward Structure](#reward-structure)
- [Observation and Action Spaces](#observation-and-action-spaces)
- [Training](#training)
- [Evaluation](#evaluation)
- [Recording](#recording)
- [TensorBoard Monitoring](#tensorboard-monitoring)
- [Sim2Real Transfer](#sim2real-transfer)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

This project implements a reinforcement learning solution for detecting and recovering from wheel entrapment in extraterrestrial regolith environments. The system trains an AAU 6-wheel rocker-bogie Mars rover to autonomously escape from sandy traps using proximal policy optimization (PPO).

**Current Status**: This repository contains a **standalone escape policy subsystem** designed to integrate with the RLRoverLab navigator for point A → B navigation in regolith environments. The policy has been trained to detect entrapment and execute recovery maneuvers, after which control is returned to the high-level navigator.

## Integration Status

This repository provides a **plug-and-play escape policy subsystem** for regolith entrapment recovery. It is *designed* to integrate with the RLRoverLab navigator but does not include the integration wrapper itself.

**How Integration Would Work**:
1. **Detection Trigger**: RLRoverLab navigator monitors for entrapment using slip/torque sensors (matching your detection logic)
2. **Control Switch**: When entrapped, navigator suspends path-following and activates this escape policy
3. **Execution**: This policy outputs 10D actions (6 drive velocities + 4 steer positions) to execute recovery maneuvers
4. **Return to Navigator**: Upon escape detection (>1.5m from entrapment point), control returns to RLRoverLab navigator to resume point A → B navigation

**What's Included**:
- ✅ Trained escape policy (`*.pt` checkpoints) compatible with AAU rover action space
- ✅ Entrapment detection logic (slip-based + torque-based temporal windows)
- ✅ Observation space matching rover state (wheel velocities, slip, IMU, joint torques)
- ✅ Reward shaping optimized for entrapment recovery (not navigation)

**What's Required for Full Integration**:
- External wrapper logic to switch between navigator and escape policy based on entrapment detection
- Waypoint tracking to resume navigation from correct location post-escape
- (Optional) Shared terrain context for informed escape direction selection

Key Features:
- **Physics Simulation**: Newton XPBD (rigid bodies) + SolverImplicitMPM (granular sand) with two-way coupling
- **Robot Model**: AAU 6-wheel rocker-bogie rover with individual drive/steer control
- **Learning Algorithm**: PPO via skrl with LSTM-enhanced policy network
- **Entrapment Detection**: Dual-sensor approach (wheel slip + motor torque monitoring)
- **Curriculum Learning**: Progressive difficulty increase in entrapment scenarios
- **Integration-Ready Design**: Matching action/observation spaces with RLRoverLab for seamless subsystem integration

## System Architecture

```
Robot:  AAU 6-wheel rocker-bogie rover (Mars_Rover.usd from RLRoverLab)
Physics: Newton XPBD (rigid bodies) + SolverImplicitMPM (granular sand)
Coupling: Two-way — sand impulses → body forces, wheel SDF → sand collision
RL:     PPO via skrl, 64 parallel envs, 50Hz physics / 25Hz policy
```

### Observation Space (28D)
- `wheel_vel` (6): drive joint velocities normalized by 6 rad/s
- `slip` (6): per-wheel slip ratio
- `steer_pos` (4): steering joint angles normalized by 0.6 rad
- `imu_acc` (3): linear acceleration / g
- `gravity_z` (1): projected gravity z (tilt indicator)
- `drive_torque_norm` (6): drive motor torques normalized by effort limits
- `entrapment_flag` (1): binary detector (low v_x + high slip for 10+ steps)
- `torque_anomaly_flag` (1): binary detector (high torque for 10+ steps)

### Action Space (10D)
- `drive_cmd` (6): velocity targets [-1,1] → ±6 rad/s
- `steer_cmd` (4): position targets [-1,1] → ±0.6 rad

### Reward Function
The reward function combines multiple terms to encourage effective entrapment escape:

```
r_total = r_progress + r_escape - p_slip - p_tilt - p_smooth - p_abnormal + r_rocking
```

Where:
- **r_progress** = w_progress × v_x × dt
  - Encourages forward movement (v_x = forward velocity in body frame)
  - Weight: w_progress = 3.0

- **r_escape** = r_escape_binary + r_escape_shaped
  - **r_escape_binary** = w_escape × time_scale if dist > ESCAPE_DISTANCE else 0
  - **r_escape_shaped** = 0.1 × (dist - ESCAPE_DISTANCE/2) × dt (clamped ≥ 0)
  - Encourages moving away from episode start point
  - Weight: w_escape = 5.0

- **p_slip** = w_slip × mean(|slip|) × dt
  - Penalizes wheel slip (energy wasting spinning)
  - Weight: w_slip = 1.0

- **p_tilt** = w_tilt × ||ang_vel_xy||₂ × dt
  - Penalizes lateral/tipping motion
  - Weight: w_tilt = 0.3

- **p_smooth** = w_smooth × ||Δaction||₂ × dt
  - Penalizes abrupt action changes
  - Weight: w_smooth = 0.05

- **p_abnormal** = w_abnormal × mean(max(0, -v_x) × torque_anomaly_flag) × dt
  - Penalizes sustained high torque with no/negative progress
  - Weight: w_abnormal = 0.5

- **r_rocking** = w_rocking × mean(|Δv_x| × entrapment_flag) × dt
  - Rewards alternating forward/backward motion when trapped
  - Weight: w_rocking = 0.1

### Curriculum Learning
Entrapment difficulty increases with training progress:
```
sinkage_min = base_min + (base_max - base_min) × progress
sinkage_depth ~ Uniform(sinkage_min, sinkage_max)
progress = min(1.0, curriculum_progress)
curriculum_progress += episodes_completed / expected_total_episodes
```
- Early training: shallow sinkage (0.02m) for easy learning
- Late training: full range (0.02-0.10m) for challenging scenarios

## Installation

### Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Dependencies
The project requires several external repositories:

1. **Newton Physics Engine**
   ```bash
   git clone https://github.com/NVIDIA-Newton/newton.git ~/newton
   cd ~/newton
   # Follow installation instructions in the repository
   ```

2. **Isaac Lab**
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
   cd ~/IsaacLab
   # Follow installation instructions
   ```

3. **Isaac Sim**
   ```bash
   # Download from NVIDIA Omniverse Launcher
   # or follow manual installation instructions
   ```

4. **RLRoverLab Assets**
   ```bash
   git clone https://github.com/abmoRobotics/RLRoverLab.git ~/RLRoverLab
   ```

### Python Environment
```bash
# Create conda environment
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

# Install required packages
pip install -r requirements.txt  # If provided
# Otherwise install key packages:
pip install torch torchvision torchaudio
pip install gymnasium
pip install wandb
pip install scikit-learn  # For TensorBoard
```

### Project Setup
```bash
# Clone this repository
git clone <repository-url> regolith_entrapment_research
cd regolith_entrapment_research

# Verify setup
./launch.sh --help  # Should show usage information
```

## Usage

### Quick Start Commands

#### 1. Training (64 environments, 200k timesteps)
```bash
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000
```

#### 2. Evaluation with Visualization
```bash
./launch.sh scripts/eval.py --num_envs 1
./launch.sh scripts/eval.py --num_envs 4 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt
```

#### 3. Recording Videos (Headless RTX Rendering)
```bash
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500
./launch.sh scripts/record.py --num_envs 1 --camera iso --num_steps 500 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt
```

#### 4. TensorBoard Monitoring
```bash
tensorboard --logdir experiments/
# Then open http://localhost:6006 in your browser
```

### Detailed Usage

#### Launch Script
The `launch.sh` script handles environment setup and launches Isaac Lab applications:
```bash
./launch.sh <script.py> [script arguments]
```

#### Training Script
```bash
scripts/train.py
  --num_envs N          # Number of parallel environments (default: 16)
  --timesteps T         # Total training timesteps (default: 200000)
  --seed S              # Random seed (default: 42)
  --checkpoint PATH     # Resume from checkpoint (optional)
```

#### Evaluation Script
```bash
scripts/eval.py
  --num_envs N          # Number of parallel environments (default: 1)
  --checkpoint PATH     # Path to model checkpoint (optional, uses random if not provided)
  --render              # Enable rendering (requires DISPLAY)
```

#### Recording Script
```bash
scripts/record.py
  --num_envs N          # Number of environments (typically 1 for video)
  --camera CAM          # Camera view: side, top, iso (default: side)
  --num_steps N         # Number of simulation steps to record
  --checkpoint PATH     # Path to model checkpoint
  --output DIR          # Output directory for MP4/PNG files
```

## Training

### Recommended Training Parameters
Based on hardware capabilities:
- **RTX 5070 Ti / RTX 4080**: 64 environments, 200k-500k timesteps
- **RTX 4090**: 512 environments, 2M+ timesteps
- **RTX 3060 / 3070**: 16-32 environments, 100k-200k timesteps

### Monitoring Training Progress
Key metrics to watch in TensorBoard:
1. **reward/mean**: Should show overall upward trend
2. **episode/escape_rate**: Percentage of episodes where rover escaped (>1.5m from origin)
3. **reward/r_progress**: Average forward progress reward per step
4. **reward/p_slip**: Average slip penalty (should decrease as policy improves)
5. **extras/log/entrap_flag_rate**: Frequency of entrapment detection (should stabilize)
6. **extras/log/torque_anomaly_rate**: Frequency of torque anomaly detection
7. **extras/log/curriculum_progress**: Should increase from 0→1.0 over training
8. **entropy**: Policy entropy (should remain > 0.02 to prevent collapse)

### Expected Learning Curve
- **Episodes 0-500**: High entrapment rate, low escape rate, negative/low progress reward
- **Episodes 500-2000**: Increasing escape rate, improving progress reward
- **Episodes 2000-5000**: Stable escape rate (>60%), refining escape maneuvers
- **Episodes 5000+**: High performance on challenging entrapment scenarios

## Evaluation

### Evaluation Metrics
During evaluation, monitor:
- **Escape Success Rate**: Percentage of episodes where rover escapes sand
- **Average Escape Time**: Steps taken to escape when successful
- **Control Effort**: Average action magnitude during escape
- **Maneuver Variety**: Different escape strategies employed (backing, rocking, steering)

### Checkpoint Evaluation
To evaluate a specific checkpoint:
```bash
./launch.sh scripts/eval.py --num_envs 4 --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/agent_50000.pt
```

### Rendering During Evaluation
For visual inspection:
```bash
./launch.sh scripts/eval.py --num_envs 1 --render
```
Note: Requires active DISPLAY environment variable or X11 forwarding.

## Recording

### Camera Options
- **side**: Side-angle view showing rover and sand interaction
- **top**: Top-down view showing movement patterns
- **iso**: Isometric 3D view
- **front**: Front-facing view

### Recording Command Examples
```bash
# Record 500 steps from side view
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500

# Record with specific checkpoint
./launch.sh scripts/record.py --num_envs 1 --camera top --num_steps 1000 \
    --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v2/checkpoints/best_agent.pt

# Record multiple angles for comprehensive analysis
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500 --output recordings/side
./launch.sh scripts/record.py --num_envs 1 --camera top --num_steps 500 --output recordings/top
```

### Output Files
- **MP4 Video**: High-quality recording of the simulation
- **PNG Sequences**: Individual frames for detailed analysis
- **Metadata**: JSON file with recording parameters

## TensorBoard Monitoring

### Key Tabs to Monitor
1. **Scalars**: Reward components, escape rates, curriculum progress
2. **Histograms**: Observation and action distributions
3. **Images**: If any image logging is implemented
4. **Charts**: Custom visualizations of learning progress

### Important Metrics Explained
- **reward/mean**: Average total reward per step (primary learning indicator)
- **episode/escape_rate**: % of episodes achieving escape condition (>1.5m from origin)
- **reward/r_progress**: Progress reward component (should be positive)
- **reward/p_slip**: Slip penalty component (should decrease over time)
- **extras/log/entrap_flag_rate**: How often entrapment is detected (balance needed)
- **extras/log/torque_anomaly_rate**: How often torque anomalies are detected
- **extras/log/curriculum_progress**: Training progress metric (0→1.0)
- **entropy**: Policy exploration measure (should stay above entropy_loss_scale)

### Debugging Training Issues
- **Reward not increasing**: Check if rover is spawning correctly in sand
- **Entrapment rate → 0**: Curriculum may be too aggressive or rewards misconfigured
- **High slip, low progress**: Wheels spinning without forward motion (common early)
- **Policy entropy collapsing**: Increase entropy_loss_scale in config
- **NaN values**: Check for numerical instability in physics simulation

## Sim2Real Transfer

### Deployment Pipeline
1. **Training**: Train policy in simulation as described above
2. **Export to ONNX**: Convert trained policy to Open Neural Network Exchange format
3. **RPi5 Deployment**: Run on physical rover with Raspberry Pi 5 controller

### Exporting to ONNX
```bash
# In sim2real directory
python onnx_export.py --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt --output policy.onnx
```

### RPi5 Controller
The sim2real directory contains:
- `onnx_export.py`: Export policy to ONNX format
- `rpi5_controller.py`: Inference loop for Raspberry Pi 5
- `requirements.txt`: Dependencies for deployment

### Safety Considerations for Real-World Deployment
1. **Conservative Triggering**: Require both slip AND torque detectors to agree before invoking escape policy
2. **Watchdog Timers**: Limit maximum escape policy execution time
3. **Fallback Behaviors**: Default to conservative motions if policy fails
4. **Hardware Limits**: Ensure commanded velocities/torques are within physical capabilities
5. **Emergency Stop**: Implement physical or software-based emergency stop

### Transfer Gap Mitigation Techniques
Implemented in training to improve real-world performance:
- **Domain Randomization**: 
  - Motor gain: 0.8-1.2× multiplicative noise
  - Observation noise: σ=0.02 additive Gaussian
  - Sinkage range: 0.02-0.10m randomized per episode
  - Friction randomization: μ=0.4-1.0 for sand particles
- **Action Delay**: Not yet implemented but recommended for sim2real gap
- **System Identification**: Measure actual motor response for better modeling

## Troubleshooting

### Common Issues and Solutions

#### 1. "No CUDA GPUs are available" Error
**Symptoms**: RuntimeError during environment initialization
**Solutions**:
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check GPU is detected: `lspci | grep -i nvidia`
- Ensure conda environment has proper CUDA toolkit
- Reinstall PyTorch with CUDA support: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

#### 2. Training Stuck at Low Rewards
**Symptoms**: reward/mean remains negative or flat
**Checks**:
- Verify rover is spawning in sand (check initial z-position)
- Confirm MPM sand is active (look for "[MPM] Sand:" print during init)
- Check if escape bonus is being awarded (monitor escape_rate)
- Try increasing forward_progress weight temporarily
- Verify curriculum learning is functioning (check curriculum_progress metric)

#### 3. Policy Not Learning Escape Behavior
**Symptoms**: entrapment_flag_rate stays high, escape_rate stays near 0%
**Checks**:
- Ensure rewards are being computed correctly (add debug prints)
- Verify both entrapment detectors are functioning
- Check if curriculum is progressing too fast/slow
- Try reducing environment complexity (--num_envs 4) to debug
- Examine if actions are being applied correctly to joints

#### 4. Isaac Sim / Viewer Issues
**Symptoms**: Blank screen, crashes during rendering
**Solutions**:
- Use `--enable_cameras` flag in launch arguments (required for headless mode with rendering)
- Do NOT use `--visualizer omniverse` (known to be broken in this setup)
- Use Newton ViewerGL for live preview (via eval.py) or RTX headless for video (via record.py)
- For GUI issues, ensure proper display setup or use headless mode with recording

#### 5. Memory Issues (VRAM/ RAM)
**Symptoms**: Out of memory errors, slow performance
**Solutions**:
- Reduce `--num_envs` parameter
- Decrease sand resolution (increase voxel_size in config)
- Use simplified USD (already implemented: Mars_Rover_Simplified.usd)
- Close other memory-intensive applications
- Monitor memory usage: `nvidia-smi` for VRAM, `top` or `htop` for RAM

#### 6. Curriculum Learning Not Working
**Symptoms**: sinkage depth not increasing over training
**Checks**:
- Verify `_curriculum_progress` is being updated in `_get_rewards()`
- Check TensorBoard for `extras/log/curriculum_progress` metric
- Ensure total episodes estimate is reasonable (default assumes ~2000 episodes)
- Manual verification: run short training and check if later episodes have deeper sinkage

### Getting Help
When encountering issues:
1. Check the terminal output for error messages
2. Consult TensorBoard for metric trends
3. Review the code at the indicated file/line in error messages
4. Search for similar issues in Isaac Lab and Newton documentation
5. If reproducible, create a minimal test case

## Project Structure

```
regolith_entrapment_research/
├── README.md                 # This file
├── CLAUDE.md                 # Project reference and quick start guide
├── ROADMAP.md                # Development roadmap and planned features
├── .gitignore               # Git ignore rules
├── launch.sh                # Main launcher script for Isaac Lab
├── view.sh                  # Alternative viewing script
│
├── configs/                 # Configuration files
│   └── ppo_aau_v2.yaml     # PPO hyperparameters and settings
│
├── detection/               # Phase 1: Sinkage detection (CNN-GRU)
│   └── [files for detection model]
│
├── envs/                    # Environment definitions
│   ├── entrapment_env.py   # Main entrapment recovery environment
│   ├── mpm_kernels.py      # Warp kernels for MPM coupling
│   └── __init__.py         # Gym registration
│
├── robots/                  # Robot configurations
│   ├── aau_rover_cfg.py    # AAU Mars rover articulation config
│   └── __init__.py
│
├── scripts/                 # Main executable scripts
│   ├── train.py            # PPO training script
│   ├── eval.py             # Evaluation and visualization
│   ├── record.py           # Video recording (RTX headless)
│   └── view_rover.py       # Standalone Newton viewer
│
├── sim2real/                # Phase 3: Sim-to-real transfer
│   ├── onnx_export.py      # Export policy to ONNX
│   ├── rpi5_controller.py  # RPi5 inference loop
│   └── [related files]
│
├── related_papers/          # Reference papers
│   └── Escape and path point tracking methods of skid-steered mobile...
│
├── experiments/             # Training outputs (auto-generated, gitignored)
│   └── aau_mars_entrapment/
│       ├── ppo_aau_mars_v1/
│       └── ppo_aau_mars_v2/
│
├── recordings/              # Output from record.py (gitignored)
├── terrain/                 # Terrain configurations
└── stubs/                   # External library stubs
```

## References

### Key Papers and Resources
1. **Bi & Ding 2026** - Primary inspiration for reward shaping and detection methods
   - "Escape and path point tracking methods of skid-steered mobile robots under undulating terrain conditions"
   - Computers and Electronics in Agriculture 241 (2026) 111247

2. **Newton Physics Engine**
   - NVIDIA's unified physics engine for rigid bodies, soft bodies, and particles
   - https://github.com/NVIDIA-Newton/newton

3. **Isaac Lab & Isaac Sim**
   - NVIDIA's robotics simulation framework
   - https://github.com/isaac-sim/IsaacLab

4. **skrl PPO Implementation**
   - Modular library for reinforcement learning algorithms
   - https://github.com/Toni-Sm/skrl

5. **RLRoverLab**
   - Source of AAU rover USD models and terrain assets
   - https://github.com/abmoRobotics/RLRoverLab

### Equations Reference

#### Physics Simulation
- **XPBD Solver**: Extended Position-Based Dynamics for rigid body simulation
- **MPM Solver**: Material Point Method for granular material simulation
- **Two-Way Coupling**: 
  - Sand → Robot: Contact impulses converted to body forces
  - Robot → Sand: Wheel SDF used as collision boundary for sand grid

#### Reinforcement Learning
- **PPO Objective**: 
  ```
  L^{CLIP}(θ) = 𝔼_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
  ```
  where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

- **Entropy Bonus**: 
  ```
  H(π_θ)(s_t) = -∑_a π_θ(a|s_t) log π_θ(a|s_t)
  ```

#### Entrapment Detection
- **Slip Ratio** (per wheel):
  ```
  slip_i = (ω_i * r - v_x) / max(|ω_i * r|, |v_x|, ε)
  ```
  where ω_i = wheel angular velocity, r = wheel radius, v_x = body forward velocity

- **Torque Anomaly**:
  ```
  anomalous_i = (|τ_i| > τ_threshold) for N consecutive steps
  ```
  where τ_i = motor torque, τ_threshold = 0.95 × τ_rated

#### Curriculum Learning
- **Progress-Based Sinkage**:
  ```
  sinkage_min(t) = sinkage_base_min + (sinkage_base_max - sinkage_base_min) × progress(t)
  progress(t) = min(1.0, episodes_completed(t) / expected_total_episodes)
  ```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing_feature`
3. Commit your changes: `git commit -m 'Add amazing_feature'`
4. Push to the branch: `git push origin feature/amazing_feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for Newton physics engine, Isaac Lab, and Isaac Sim
- The researchers behind the Bi & Ding 2026 paper for foundational concepts
- The RLRoverLab team for providing rover models and terrain assets
- The open-source robotics and ML communities for tools and frameworks

---

*Last updated: April 2026*
*For questions or issues, please refer to the CLAUDE.md file or open an issue on GitHub.*