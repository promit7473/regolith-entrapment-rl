# Roadmap — Regolith Entrapment Recovery Research

Status as of 2026-04-03: Step 1 complete, Step 2 in progress.

---

## Step 1: First AAU Training Run ✅ DONE (2026-04-03)
**Goal**: Verify the full pipeline works end-to-end.
**Result**: Pipeline verified — 500-step smoke test passed at 4 envs, ~18 it/s.

```bash
./launch.sh scripts/train.py --num_envs 64 --timesteps 200000
```

**What to check**:
- [ ] No crashes during env init, MPM setup, or first rollout
- [ ] TensorBoard: `tensorboard --logdir experiments/`
  - reward should trend upward (even slowly)
  - entropy should NOT collapse to 0 (entropy_loss_scale=0.02 prevents this)
  - episode length should vary (not all hitting max)
- [ ] Checkpoint saved to `experiments/aau_mars_entrapment/ppo_aau_mars_v1/`

**If it crashes**: most likely causes:
- VRAM OOM → reduce `--num_envs` to 32 or 16
- Warp kernel error → check warp-lang version (must be ≥1.11.1)
- Newton model error → check that rover USD and ground plane initialize correctly

**Expected wall time**: ~30-60 min for 200k steps on 5070Ti at 64 envs.

---

## Step 1.5: Obs Enhancement (Bi & Ding 2026 paper) ✅ DONE (2026-04-03)
**Goal**: Add torque observations and entrapment detection inspired by related paper.
**Changes**:
- Obs space: 20D → 27D (added 6 drive torques + 1 entrapment flag)
- Domain randomization: added motor gain noise (0.8–1.2×), obs noise (σ=0.02), wider sinkage range (0.02–0.10m)
- Entrapment flag: binary detector (v_x < 0.05 AND slip > 0.5 for 10+ consecutive steps)
- Experiment name: ppo_aau_mars_v1 → ppo_aau_mars_v2

---

## Step 2: Diagnose Reward Signal ← CURRENT (training launched 2026-04-03)
**Goal**: Is the rover learning anything?

Check TensorBoard:
- `reward/mean` → should increase over time
- `reward/r_progress` → is v_x positive? rover moving forward?
- `reward/p_slip` → high slip means wheels spinning in sand (expected early)
- `episode/escape_rate` → 0% initially is fine, should start rising after ~100k

**If reward is flat/negative**:
- Increase `rew_forward_progress` from 2.0 → 4.0
- Check if rover spawns correctly (z height, not buried too deep)
- Check if sand particles are actually affecting the rover (MPM coupling working?)

**If entropy collapses**:
- Increase `entropy_loss_scale` from 0.02 → 0.05
- Reduce learning rate from 3e-4 → 1e-4

---

## Step 3: Reward Tuning + Curriculum (Week 1-2)
**Goal**: Achieve >10% escape rate.

Ideas to try (one at a time, measure in TensorBoard):
1. **Curriculum on sinkage depth**: start shallow (0.02m), ramp to 0.15m
   - Modify `_reset_idx` to sample sinkage based on training progress
2. **Shaped escape reward**: distance-based shaping instead of binary escape bonus
   - `r_dist = 1.0 * (dist_now - dist_prev) / dt` (reward for moving away from origin)
3. **Rocking bonus**: reward for alternating forward/backward velocity
4. **Asymmetric drive bonus**: reward for differential wheel speeds (helps turn out)

---

## Step 4: Scale to 4090 / Supercomputer (Week 2-3)
**Goal**: 2M-step run at 512 envs.

**What changes for 4090**:
- `--num_envs 512` (8× more parallel envs)
- `--timesteps 2000000`
- May need to increase `rollouts` to 48 (more data per update)
- VRAM: 4090 has 24GB vs 5070Ti 16GB — can handle 512 envs easily

**For supercomputer (multi-GPU)**:
- skrl supports distributed PPO (check skrl docs for `--multi_gpu`)
- Or simply run independent seeds on separate GPUs and pick best

---

## Step 5: Recording + Visualization (Week 2-3)
**Goal**: Research-quality video for paper/presentation.

```bash
# After training completes with a good checkpoint:
./launch.sh scripts/record.py --num_envs 1 --camera side --num_steps 500 \
    --checkpoint experiments/aau_mars_entrapment/ppo_aau_mars_v1/checkpoints/best_agent.pt

./launch.sh scripts/record.py --num_envs 1 --camera iso --num_steps 500 \
    --checkpoint <best_checkpoint>

./launch.sh scripts/record.py --num_envs 1 --camera top --num_steps 500 \
    --checkpoint <best_checkpoint>
```

**Known issue**: record.py uses `omni.replicator` for camera — may need debugging
on first run. The RTX renderer needs shader compilation (~30s first time).

**If RTX recording doesn't work**: fall back to Newton ViewerGL screenshots
(eval.py with viewer open, then screenshot externally with `gnome-screenshot`).

---

## Step 6: Phase 1 — Sinkage Detection Data Collection (Week 3-4)
**Goal**: Build training dataset for CNN-GRU sinkage classifier.

**Approach**:
1. Run trained RL policy for ~10k episodes, logging per-step data
2. Label each step: normal (moving), sinking (slowing down), entrapped (stuck)
3. Labels come from ground-truth sim state (wheel slip > threshold, v_x < threshold)
4. Train CNN-GRU classifier (detection/models/cnn_gru.py)
5. Target: >90% recall on "entrapped" class, <1s latency

**Script needed**: `scripts/collect_detection_data.py` — run policy, save
(obs, label) pairs to disk. Label logic:
- normal: v_x > 0.1 m/s AND mean_slip < 0.3
- sinking: v_x < 0.1 AND mean_slip > 0.3 AND distance_from_origin < 0.5m
- entrapped: v_x < 0.02 for >2 consecutive seconds

---

## Step 7: Phase 3 — Sim-to-Real (Month 2-3)
**Goal**: Deploy on physical rover with RPi5.

**Pipeline**:
1. Export trained policy to ONNX: `sim2real/onnx_export.py`
2. Export detection model to ONNX
3. Test on RPi5: `sim2real/rpi5_controller.py` (10Hz inference loop)
4. Physical test on sand tray

**Sim-to-real gap mitigation** (add during training):
- Domain randomization on friction, motor gain, IMU noise
- Action delay randomization (0-40ms)
- System identification from physical rover (measure actual motor response)

---

## Hardware Notes
- **Current dev**: RTX 5070 Ti (16GB VRAM) — good for 64 envs
- **Planned scale**: RTX 4090 (24GB) — 512 envs
- **Deployment**: RPi5 + ONNX Runtime (CPU inference ~5ms at 10Hz)
