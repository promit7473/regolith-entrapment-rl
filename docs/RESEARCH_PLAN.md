# Regolith Wheel Entrapment Recovery Research

## One-Line Summary
Train a mobile robot to detect and autonomously recover from wheel entrapment
in granular regolith using proprioceptive sensing + deep RL, then deploy on RPi5.

## Why NOT starting with RLRoverLab's Mars environment
RLRoverLab uses rigid-body USD terrain meshes — no particle physics.
Your research requires granular (MPM) simulation for realistic sinkage.
Newton MPM (already set up) provides this without any USD terrain dependency.
The Martian "aesthetic" can be added later by texturing the terrain.

## Three-Phase Research Plan

### Phase 1 — Sinkage Detection (2–3 months)
**Goal**: Classify rover state (normal / sinking / entrapped) in real-time
from proprioceptive sensors alone (no vision, no external localization).

| Item | Detail |
|------|--------|
| Simulator | Newton MPM (granular terrain) + Isaac Lab (robot) |
| Robot | 4WD skid-steer (simple, buildable) |
| Input features | wheel_torque(4) + wheel_vel(4) + imu_accel(3) = **11D** |
| Sequence window | 50 steps @ 10Hz = 5 seconds |
| Model | **CNN-GRU hybrid** (see phase1_detection/models/cnn_gru.py) |
| Output | 3-class probability: normal / sinking / entrapped |
| Training data | Simulated episodes with varied terrain density and slope |
| Key metric | F1 score on "entrapped" class (precision-recall tradeoff) |

**Deliverable**: Trained detector with >90% entrapped recall, <1s detection latency.

### Phase 2 — Recovery Policy (2–3 months)
**Goal**: RL agent learns escape maneuvers from entrapped state using
only proprioceptive observations (same 11D as Phase 1).

| Item | Detail |
|------|--------|
| Algorithm | PPO (stable, sample-efficient for continuous control) |
| Observation | 11D proprioceptive vector |
| Action | 4D wheel velocity commands (independent per wheel) |
| Reward | +100 escape bonus, +progress, -slip, -energy |
| Curriculum | Start: shallow sinkage → hard: deep entrapment + slope |
| Parallel envs | 512–2048 envs via Isaac Lab vectorization |
| Key technique | Domain randomization (friction, terrain density, robot mass) |

**Recovery maneuvers the policy may discover:**
- Rocking (alternating forward/backward)
- Differential thrust (spin one side faster)
- Wheel oscillation (high-freq small motions to compact particles)
- Dig-and-push (bury one side, lever off)

**Deliverable**: Policy achieving >80% escape rate across varied entrapment scenarios.

### Phase 3 — Sim-to-Real on RPi5 (2–3 months)
**Goal**: Deploy both models on a physical robot with Raspberry Pi 5.

| Item | Detail |
|------|--------|
| Hardware | RPi5 + 4x DC motors + encoders + MPU6050 IMU |
| Inference | ONNX Runtime (CPU, ~5ms per inference at 10Hz) |
| Sim2Real gap | Domain randomization + System ID (match motor friction) |
| Test surface | Sand tray (indoor), gravel (outdoor) |
| Metrics | Escape rate (real), detection accuracy (real), latency |

**Deliverable**: Physical robot that autonomously detects and recovers from
entrapment in sand/gravel without human intervention.

## Directory Structure
```
regolith_entrapment_research/
├── phase1_detection/
│   ├── data/               ← npz files from simulation
│   ├── models/
│   │   └── cnn_gru.py      ← CNN-GRU detector architecture
│   ├── scripts/
│   │   ├── collect_data.py ← Run sim, record proprioceptive sequences
│   │   └── train_detector.py
│   └── notebooks/          ← EDA, confusion matrices, SHAP analysis
│
├── phase2_recovery/
│   ├── envs/
│   │   └── entrapment_env.py  ← IsaacLab + Newton RL environment
│   ├── policies/           ← Saved PPO checkpoints
│   ├── rewards/            ← Reward function implementations
│   └── scripts/
│       ├── train_ppo.py    ← RL training entrypoint
│       └── eval_policy.py  ← Evaluate escape rate
│
├── phase3_sim2real/
│   ├── onnx_export/
│   │   └── export_model.py ← PT → ONNX conversion
│   ├── rpi5_deploy/
│   │   └── rover_controller.py  ← Onboard 10Hz control loop
│   └── hardware_config/    ← Wiring diagrams, motor calibration
│
├── shared/
│   ├── robot_assets/       ← URDF / USD files
│   ├── terrain/
│   │   └── granular_terrain.py  ← Newton MPM terrain class
│   ├── utils/              ← Slip ratio, logging, visualization
│   └── configs/
│       └── research_config.yaml
│
└── experiments/            ← Logged runs (wandb / tensorboard)
```

## Key Design Decisions

### Why CNN-GRU over plain LSTM?
- 20-30% faster training and inference (fewer GRU gates)
- CNN front-end learns per-wheel patterns automatically
- Better for short sequences (50 steps) — LSTM excels at 1000+ steps
- Empirically outperforms LSTM on proprioceptive robotics benchmarks

### Why skid-steer over Ackermann (like the AAU rover)?
- Simpler dynamics (no steering joints → fewer DOF to control)
- Physically buildable with 2 L298N motor drivers + 4 motors
- Skid-steer rovers can spin in place → critical for escape maneuvers
- Results transfer to more complex rovers (AAU, ExoMy) in future work

### Why Newton MPM over PhysX PBD?
- MPM is physically accurate for regolith (Coulomb friction, yield surface)
- Differentiable → future: gradient-based trajectory optimization
- PhysX PBD is position-based (not force-based) → overestimates traction

### When to bring in RLRoverLab?
After Phase 2. Once the escape policy works, port it to RLRoverLab's
AAU rover USD + Mars terrain for the "Mars rover" presentation/paper.
The science stays the same; only the visual context changes.

## Related Work to Read
1. Inotsume et al. (2020) - Slip prediction for planetary rovers
2. Ding et al. (2011) - Wheel-terrain interaction modeling
3. Shibly et al. (2005) - Equivalent sinkage concept
4. Jain et al. (2019) - RL for planetary rover locomotion
5. Fankhauser et al. (2018) - ANYmal recovery (inspiration for multi-phase)
