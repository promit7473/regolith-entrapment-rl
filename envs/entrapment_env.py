"""
Mars Rover — Wheel Entrapment Recovery — Newton DirectRLEnv.

Physics stack:
  • MuJoCo Warp (MJWarpSolverCfg) — rigid-body dynamics for the 6-wheel Mars rover
  • SolverImplicitMPM      — sparse-grid MPM for granular regolith
  • Two-way coupling        — sand impulses → robot bodies, robot SDF → sand grid

Mars environment:
  • Static terrain mesh  : RLRoverLab terrain_merged.usd (photogrammetry Mars analog)
  • Rock scatter         : 10 rock USDs placed randomly each episode reset
  • Regolith bed         : 3.5 m × 3.5 m × 0.60 m MPM particle bed per env

Robot: 6-wheel rocker-bogie Mars rover (Mars_Rover.usd)
  Drive joints  (6) : .*Drive_Continuous   — velocity control, ±6 rad/s
  Steer joints  (4) : .*Steer_Revolute     — position control, ±0.6 rad
  Passive joints(N) : Rocker/Differential  — free

Obs (29D):
  wheel_vel     (6)  — drive joint velocities normalised by 6 rad/s
  slip          (6)  — per-wheel slip ratio
  steer_pos     (4)  — steering joint angles normalised by 0.6 rad
  imu_acc       (3)  — linear acceleration / g
  gravity_z     (1)  — projected gravity z (tilt indicator)
  drive_torque_delta(6) — per-wheel step-wise torque change (struggle dynamics indicator)
  entrap_flag   (1)  — binary entrapment indicator
  slip_anomaly  (1)  — sustained high-slip + low-velocity anomaly flag
  dist_norm     (1)  — distance from env origin / escape threshold (0=spawn, 1=escaped)

Action (10D):
  drive_cmd     (6)  — velocity targets [-1,1] → ±6 rad/s
  steer_cmd     (4)  — position targets [-1,1] → ±0.6 rad
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

import newton as nt
from newton.solvers import SolverImplicitMPM

from robots import MARS_ROVER_CFG, ROVER_WHEEL_RADIUS
from .mpm_kernels import (
    compute_body_forces, subtract_body_force, reset_particle_range,
    clamp_escaped_particles, clamp_sand_forces_during_settle, decrement_settle_counter,
    set_env_friction,
)
from paths import RLROVER_ASSETS


# ── Constants ─────────────────────────────────────────────────────────────────

WHEEL_RADIUS    = ROVER_WHEEL_RADIUS        # 0.10 m
# Offset from chassis root (Body) to wheel-center in Z (from USD geometry).
# Drive joints sit at z = -0.167 m relative to Body.
CHASSIS_TO_WHEEL_Z = 0.167                 # m
DRIVE_JOINTS    = [".*Drive_Continuous"]    # regex — 6 wheels
STEER_JOINTS    = [".*Steer_Revolute"]      # regex — 4 corners
DRIVE_VEL_LIMIT = 6.0                       # rad/s
STEER_POS_LIMIT = 0.6                       # rad  (~34°)
ESCAPE_DISTANCE = 3.0                       # m — projected travel along escape heading from spawn
# Derived from physical rover geometry + 0.25 m safety margin:
#   SAND_HALF(1.75) + spawn_offset(0.5) + rover_half_length(0.5) + margin(0.25) = 3.0 m
# At escape: body centre is 2.5 m from env centre (0.75 m past sand edge);
# rear wheels (0.5 m behind centre) are 0.25 m past the sand far edge — fully clear.
# Clean round number; strong paper definition: "rover has fully exited the entrapment zone."
# Milestones at 0.5 / 1.0 / 2.0 / 3.0 m give four bands for escape-rate plots.

# Regolith pit geometry — SQUARE bed (per env, centred at env origin).
# 3.5 m × 3.5 m × 0.30 m: large enough for meaningful omnidirectional escape
# runway in every heading. Particle count ≈ 140×140×12 × PPC=2 ≈ 235 000 per env.
SAND_HALF    = 1.75            # m — half-side of the square sand bed
SAND_HALF_X  = SAND_HALF      # kept for kernel call compatibility
SAND_HALF_Y  = SAND_HALF
SAND_DEPTH   = 0.60           # v8: 0.30 → 0.60. Curriculum sinkage tops out at 0.28 m (wheel bottom 0.28 below the SETTLED surface by definition); with the measured wheel r=0.0994 the bed leaves >0.25 m (≈2.5× wheel radius) of compliant sand below the deepest wheel position — removes the rigid-floor confound that fed the v7-era bulldoze exploit at 0.30 m bed depth. Particle count scales ~2× (z cells 12 → 24).
VOXEL_SIZE   = 0.05
PPC          = 2.0    # particles per cell; voxel=0.05 preserved so SDF coupling quality unchanged

# Spawn offset (world X from env origin). Placing the rover in the -X half of the
# bed gives 2.0 m of runway ahead and ensures that at the escape moment the front
# wheels are still ~0.11 m inside the bed edge (no cliff drop triggering flipped/sunk).
SPAWN_X_OFFSET = -0.5

# Mars terrain / rock asset paths (from RLRoverLab)
MARS_TERRAIN_USD = str(RLROVER_ASSETS / "terrains/mars/terrain1/terrain_merged.usd")
ROCK_USDS = [
    str(RLROVER_ASSETS / f"objects/rocks/rock_{i}/rock_{i}.usd")
    for i in range(10)
]
ROCKS_PER_ENV   = 6    # how many rocks scattered per env per episode reset
ROCK_SCATTER_R  = 0.9  # scatter radius [m] around env origin (inside sand patch)


# ── Env Config ─────────────────────────────────────────────────────────────────

@configclass
class EntrapmentEnvCfg(DirectRLEnvCfg):
    # 6 drive vel + 6 slip + 4 steer pos + 3 imu acc + 1 gravity_z
    # + 6 drive torque DELTA + 1 entrapment flag + 1 slip anomaly flag
    # + 1 dist_norm (distance from origin / escape threshold) = 29
    # NOTE: torque_norm (absolute) was removed — motors always saturate in sand
    # (ratio=1.0 always), carrying zero information. Replaced with per-wheel
    # torque step-change (delta), which varies with struggle dynamics.
    policy_observation_space     = 29
    privileged_observation_space = 8   # oracle features (sinkage, burial, body vel, ...)
    use_privileged_critic        = True
    # Concatenated obs: policy slices [:29], critic reads full [:37].
    observation_space = 29 + 8
    # 6 drive cmd + 4 steer cmd = 10
    action_space      = 10
    state_space       = 0

    episode_length_s = 40.0  # v8: 30 → 40. Real recovery (Bi & Ding 2026: ~12–15 s per attempt) needs headroom for multiple attempts before the timeout clips a converging recovery. (Original note cited the v8 22 Nm effort limit — reverted to 40 Nm in v9; the headroom rationale stands on its own.)
    decimation       = 2     # policy at 25 Hz
    skip_mpm         = False  # set True for viewer-only mode (no sand physics)

    # Reward weights
    # r_progress kept moderate so escape bonus stays dominant; agent must escape,
    # not just drive laps on the sand surface.
    rew_forward_progress = 1.5   # reduced: forward velocity alone shouldn't beat escape bonus
    rew_escape_bonus     = 20.0  # raised: escaping must be the highest-value action
    pen_slip             = 0.5   # halved: slip is unavoidable in sand, not a primary signal
    pen_tilt             = 0.3
    pen_action_delta     = 0.01  # reduced 0.12→0.01: rocking requires large rapid action changes (forward↔backward); 0.12 dominated the reward (-0.016/step vs rew_rocking +0.001/step), teaching the policy that zero actions beat rocking. 0.01 keeps smoothness signal without suppressing recovery behaviour.
    pen_abnormal         = 0.3   # reduced: anomaly flag is still imperfect
    rew_rocking          = 0.5   # additive bootstrap bonus during entrapment; primary escape signal is now r_progress + Δdist shaping in r_escape (un-gated 2026-04-24)
    # Vertical-hop penalty: -|v_z| × dt. Directly punishes chassis vertical motion
    # (bouncing/hopping). Was inert before — locomotion-time bouncing in v6 was
    # emergent from saturated torque, not from any hop reward. This term makes the
    # hop-and-grab strategy explicitly costly so the policy must find a smoother gait.
    pen_hop              = 0.5
    # Slip-aware grind penalty (v9): replaces the v8 `pen_action_mag` term, which
    # was theatre — it penalised commanded action magnitude in [-1,1] units, but
    # realised motor torque saturates at effort_limit_sim regardless of command
    # magnitude (because v_error in sand is always huge). The policy could keep
    # commanding max velocity AND realise the same torque, so the penalty just
    # marked max-command as costly without forcing reduced realised torque.
    # Replacement: penalise drive command magnitude ONLY when wheels are slipping
    # (mean_slip > slip_grind_thresh). Targets the specific failure mode — wheels
    # spinning uselessly at high slip — and rewards "ease off the throttle when
    # stuck" instead of "use less action everywhere".
    pen_grind            = 0.05   # reduced 0.30→0.05: 0.30 causes policy collapse in real 0.60m sand — cumulative grind penalty (-12/episode) dominates escape bonus (20), teaching the policy that zero drive command is optimal. 0.05 keeps the signal without making inaction the best strategy.
    slip_grind_thresh    = 0.70   # mean |slip| above which grinding penalty fires
    pen_reverse          = 1.0    # weight on world-frame negative-progress penalty (was hardcoded inline)

    # Domain randomization ranges (inspired by Bi & Ding 2026, Table 9)
    dr_motor_gain_range  = (0.8, 1.2)   # multiplicative gain on drive velocity targets

    # Per-sensor observation noise (RPi5 deployment realism).
    # Structured noise: different sensors have different noise characteristics.
    # These values match real sensor datasheets for the deployment hardware.
    # Replaces the old flat dr_obs_noise_std = 0.02 across all 29 dims.
    # Physical-unit sigmas (rad/s, rad, m/s²) are divided by the obs channel's
    # normaliser at the injection site so datasheet values apply verbatim.
    dr_noise_wheel_vel    = 0.05   # rad/s — encoder quantization + bearing noise (0.8% of 6 rad/s range)
    dr_noise_slip         = 0.03   # slip ratio (dimensionless) — derived from wheel velocity noise
    dr_noise_steer_pos    = 0.01   # rad — potentiometer / encoder noise
    dr_noise_imu_acc      = 0.15   # m/s² — MEMS IMU white noise (MPU-6050 class: 0.1–0.2 m/s²)
    dr_noise_grav_z       = 0.02   # gravity projection noise from IMU attitude error
    dr_noise_drive_torque = 0.05   # normalised — current sensing noise (~5% of torque range)
    dr_noise_dist_norm    = 0.02   # normalised distance — GPS/odometry drift

    # Failure mode logging: write per-episode metadata to CSV for post-hoc analysis.
    log_failure_modes     = True   # set False to disable I/O overhead during profiling
    # Curriculum range calibrated on the DEFINITIVE honest-spawn ladder
    # (2026-06-13, pinned μ=0.75, full throttle, single draws):
    #   0.05 easy escape (0.24 m/s) | 0.10 marginal (~25 s to escape)
    #   0.15 stable trap | 0.20/0.25 terminal trap (self-burial)
    # Floor 0.05 = clearly escapable bootstrap level; 0.10 is the first hard
    # rung; ≥0.15 is where the naive baseline fails (hub-level burial — the
    # real-world failure depth). Paper-grade numbers come from the N=50
    # escape_eval sweep, not these single draws.
    dr_sinkage_range     = (0.05, 0.28) # m — full curriculum range
    # Sand cohesion (MPM shear yield stress, Pa). 0 = purely frictional sand,
    # which only moderately traps a powered rover (constant-drive escapes ~35%
    # in the in-engine sweep). Real regolith cohesion is ~0.1–10 kPa.
    # MPM_YIELD_STRESS env var overrides this field (kept for the 2026-06-06
    # cohesion-study workflow).
    sand_yield_stress    = 0.0
    dr_friction_range    = (0.6, 0.9)   # μ_s sand friction DR per reset.
    # Eval overrides: when set, pin sinkage / friction to a single value.
    sinkage_override     = None         # float | None — m
    friction_override    = None         # float | None — μ_s

    # Post-reset MPM settle window. Teleporting the chassis into the sand
    # volume creates an instantaneous SDF/particle overlap; MPM resolves it
    # as a huge impulse that can launch the rover ("bouncing" behaviour).
    # ROOT CAUSE of 37.5% trivial escapes: short settle (20 steps) + hard cap
    # (800N) → clamp expires before penetration resolves → abrupt spike → launch.
    # FIX: longer window (60 steps = 1.2s) + gentler cap (250N) so particles
    # push out organically through the clamped-but-nonzero force. No abrupt spike.
    # RETUNED for full-strength coupling (2026-06-13, post substep-dilution
    # fix). The historical 250 N/body was calibrated when applied forces were
    # silently ÷8 (effective ≈31 N — gentle). At full strength, 250 N/body
    # LIFTED the rover out of its prescribed burial during the settle window
    # (observed: rover "floating" on the bed surface at spawn; z rising during
    # settle). 40 N/body ≈ the old effective value with margin: per-wheel
    # static support needs ~22 N (130 N Mars weight / 6 wheels), so full
    # support fits under the cap while teleport-overlap spikes stay bounded.
    settle_steps             = 60     # 1.2 s at 50 Hz physics
    settle_force_cap         = 40.0   # N per body (≈ old 250/8 effective, +margin)
    settle_torque_cap        = 6.0    # N·m per body — proportional

    # Entrapment detection thresholds (v_x < vx AND mean_slip > slip for N steps)
    entrap_vx_thresh    = 0.15   # m/s
    entrap_slip_thresh  = 0.4
    entrap_steps_thresh = 15     # ~0.6 s at 25 Hz

    # Post-reset burial grace: force entrap_flag=1 for up to N policy steps after
    # reset, OR until the rover has moved beyond `burial_grace_dist` from spawn
    # (whichever comes first).
    # FIX (entrap_flag_rate was 4.2%): burial grace was too short (25 steps / 0.15m).
    # Bouncing rover exited 0.15m before grace expired, clearing the flag instantly.
    # Extended to 75 steps (3s) / 0.5m — flag stays on through rocking cycles.
    burial_grace_steps = 50      # v8: 75 → 50 (3s → 2s). At 75 the flag-rate floor (~0.30) was so high that real re-entrapment signal was buried in the grace plateau (entrap_flag_rate ≈ 0.33 ≈ floor). 50 still covers the initial rocking window but lets the flag clear faster so re-entrap is observable.
    burial_grace_dist  = 0.50    # m — must leave spawn zone meaningfully before flag clears

    # Slip-based anomaly detection: sustained high slip + low progress = struggling.
    # REPLACES torque-ratio anomaly: motor torque is ALWAYS saturated at 80 Nm when
    # wheels fight sand (target=6 rad/s, actual≈0 → error=6 → τ=24000 → capped).
    # torque_ratio was always ~1.0 → flag always on → zero information.
    # Slip+velocity formulation is informative regardless of actuator tuning:
    #   stuck:    slip>0.65 AND v_x<0.20 → fires ✓
    #   escaping: slip high BUT v_x>0.20 → does not fire ✓
    # Threshold raised 0.65 → 0.92: at 0.65 the flag fired on baseline sand-driving
    # (slip_anomaly_rate ≈ 0.98 saturated → 0 information). 0.92 only triggers on
    # genuinely pathological slip, restoring discriminative power.
    slip_anomaly_thresh       = 0.92   # mean |slip| above free-progress driving
    slip_anomaly_vx_thresh    = 0.20   # m/s — still counts as making progress
    slip_anomaly_steps_thresh = 15     # ~0.6 s at 25 Hz

    # Trivial-escape detection: clearing 0.3 m within this many policy steps flags
    # a bounce-exploit (rover popping out of spawn). Was 125 (5 s) — false-positived
    # a converged policy escaping in ~4.76 s. 50 (2 s) only catches genuine instant pops.
    trivial_escape_steps_thresh = 50

    # Lateral-drift termination: episode terminates if perpendicular distance from
    # the escape axis exceeds env_spacing/2 - safety margin. Prevents inter-env
    # collisions when the policy drifts sideways instead of progressing.
    lateral_oob_margin = 1.0  # m — subtract from env_spacing/2

    # Sunk termination: wheel-bottom burial below the settled sand surface at
    # which the episode ends (catastrophic swallowing). Must exceed the max
    # curriculum sinkage (0.28) by a healthy dynamic-dig-in margin.
    sunk_burial_thresh = 0.45  # m

    # Curriculum backoff: regress curriculum_progress when the policy collapses
    # (low escape rate) OR exploits bouncing (high trivial rate). Asymmetric — slower
    # decay than advance to avoid thrashing, but breaks the monotonic-progress trap.
    curriculum_backoff_rate_thresh    = 0.20
    curriculum_backoff_trivial_thresh = 0.60
    curriculum_backoff_scale          = 0.25  # × curriculum_speed

    # MuJoCo Warp solver — same choice as view_rover.py.
    # XPBD cannot stably support an articulated rover regardless of contact settings
    # (329 mesh shapes → contact buffer overflow → NaN; proxy spheres → slow sink → explosion).
    # MuJoCo handles articulated body contacts correctly at any env count.
    solver_cfg = MJWarpSolverCfg(
        use_mujoco_cpu=True, # bypass mujoco_warp GPU (conflicts with conda warp-lang)
        nconmax=50,          # contacts per env (6 wheel spheres + buffer)
        njmax=200,           # constraints per env
        iterations=4,
        ls_iterations=4,
        cone="elliptic",     # better friction model for wheel-ground contact
        impratio=100,        # high ratio prevents wheel sinking
        solver="newton",
        integrator="euler",
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        # was 4 (explicit euler blew up under stiff drive damping). 8 halves the
        # substep (0.005->0.0025s) and stabilises. Override via NEWTON_SUBSTEPS env
        # var for the integration-convergence study (2026-06-06).
        num_substeps=int(os.environ.get("NEWTON_SUBSTEPS", "8")),
        debug_mode=False,
        use_cuda_graph=False,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=decimation,
        newton_cfg=newton_cfg,
        gravity=(0.0, 0.0, -3.72),   # Mars gravity (3.72 m/s²) — Earth default is -9.81
    )

    robot_cfg: ArticulationCfg = MARS_ROVER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=14.0,    # 2×SAND_HALF(3.5m) + 2×ESCAPE_DISTANCE(3.0m) = 13.0m → 1.0m total buffer (0.5m/side); see lateral_oob_margin for termination bound
        replicate_physics=True,
        clone_in_fabric=True,
    )


# ── Env ────────────────────────────────────────────────────────────────────────

class EntrapmentEnv(DirectRLEnv):
    cfg: EntrapmentEnvCfg

    def __init__(self, cfg: EntrapmentEnvCfg, render_mode: str | None = None, **kwargs):
        self._mpm_ready = False
        super().__init__(cfg, render_mode, **kwargs)

        # Drive and steer joint index arrays
        _, _, self._drive_ids = self.robot.find_joints(DRIVE_JOINTS, preserve_order=True)
        _, _, self._steer_ids = self.robot.find_joints(STEER_JOINTS, preserve_order=True)

        # Cached warp→torch views
        self.joint_vel  = wp.to_torch(self.robot.data.joint_vel)
        self.joint_pos  = wp.to_torch(self.robot.data.joint_pos)
        self.root_pos   = wp.to_torch(self.robot.data.root_link_pos_w)
        self.root_vel_b = wp.to_torch(self.robot.data.root_com_lin_vel_b)
        self.ang_vel_b  = wp.to_torch(self.robot.data.root_com_ang_vel_b)
        self.grav_b     = wp.to_torch(self.robot.data.projected_gravity_b)
        self.lin_acc_w  = wp.to_torch(self.robot.data.body_com_lin_acc_w)[:, 0, :]

        # Gravity vector/magnitude for the IMU specific-force observation.
        self._g_vec_w = torch.tensor(self.cfg.sim.gravity, device=self.device)
        self._g_mag   = float(torch.linalg.norm(self._g_vec_w).clamp(min=1e-6))

        self.prev_action = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.prev_v_x = torch.zeros(self.num_envs, device=self.device)  # For rocking bonus calculation
        self._prev_v_x_world = torch.zeros(self.num_envs, device=self.device)  # World-frame v projected on escape_dir

        # One-time milestone tracking (4 milestones: 0.5m, 1.0m, 1.5m, 2.0m)
        # Each flag flips 0→1 the first time that distance is crossed in the episode.
        self._milestone_reached = torch.zeros(self.num_envs, 4, device=self.device)

        # Entrapment detection state (inspired by Bi & Ding 2026)
        # Counts consecutive steps where v_x < threshold AND mean_slip > threshold
        self._entrap_counter = torch.zeros(self.num_envs, device=self.device)
        self._entrap_flag    = torch.zeros(self.num_envs, device=self.device)

        # Per-env true sinkage (sampled each reset) — privileged critic signal.
        self._sinkage = torch.zeros(self.num_envs, device=self.device)
        # Per-env sand friction (sampled each reset) — cached for failure-mode CSV.
        # NaN until the first post-MPM-init reset writes a real DR sample.
        self._friction = torch.full((self.num_envs,), float("nan"), device=self.device)

        # Escape tracking — counted in _get_dones (before reset) so the metric
        # isn't wiped by the env reset that happens between dones and rewards.
        self._escape_count = 0       # total escapes across all envs
        self._episode_count = 0      # total episode terminations (escaped + timed-out + failed)
        # Rolling window of last N episode outcomes (1 = escaped, 0 = failed/timeout).
        # Used for competence-gated curriculum: only advance sinkage difficulty when
        # recent escape rate clears a threshold, preventing premature hard-setting
        # when the policy hasn't actually mastered the current level.
        self._recent_escapes      = torch.zeros(100, device=self.device)
        self._recent_escapes_idx  = 0
        self._recent_escapes_full = False
        # Per-env "time to first escape-distance threshold crossing" — used to flag
        # trivial escapes (entrapment that wasn't actually trapping).
        self._episode_step        = torch.zeros(self.num_envs, device=self.device)
        self._first_progress_step = torch.full((self.num_envs,), -1.0, device=self.device)
        # Rolling previous distance per env — for Δdist shaping (prevents reward
        # farming from lingering at high-dist without progressing).
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._ENTRAP_VX_THRESH    = self.cfg.entrap_vx_thresh
        self._ENTRAP_SLIP_THRESH  = self.cfg.entrap_slip_thresh
        self._ENTRAP_STEPS_THRESH = self.cfg.entrap_steps_thresh

        # Post-reset burial grace: counts down while the rover is close to spawn.
        # Forces entrap_flag=1 until either the counter expires or the rover moves
        # > burial_grace_dist from env origin.
        self._burial_grace_counter = torch.zeros(self.num_envs, device=self.device)

        # Slip-based anomaly detection (replaces torque-ratio which was always ~1.0).
        # Motors always saturate vs sand (target=6 rad/s, error=6 → τ=24kNm → capped at 80 Nm).
        # Slip+velocity is informative: fires when wheels spin without body motion.
        self._torque_anomaly_counter = torch.zeros(self.num_envs, device=self.device)
        self._torque_anomaly_flag    = torch.zeros(self.num_envs, device=self.device)
        self._prev_drive_torque_norm = torch.zeros(self.num_envs, 6, device=self.device)
        self._SLIP_ANOMALY_THRESH       = self.cfg.slip_anomaly_thresh
        self._SLIP_ANOMALY_VX_THRESH    = self.cfg.slip_anomaly_vx_thresh
        self._SLIP_ANOMALY_STEPS_THRESH = self.cfg.slip_anomaly_steps_thresh

        # Domain randomization: per-env motor gain (randomized at reset)
        self._motor_gain = torch.ones(self.num_envs, 1, device=self.device)
        
        # Curriculum learning progress tracker (across episodes).
        # _total_timesteps is set by train.py after env creation so the curriculum
        # denominator scales correctly with --timesteps. Default 200k if not set.
        self._curriculum_progress = torch.zeros(1, device=self.device)
        self._total_timesteps = 200_000  # overwritten by train.py via env._total_timesteps = args_cli.timesteps

        # Per-env rock pose storage (num_envs, ROCKS_PER_ENV, 3)
        self._rock_positions = torch.zeros(
            self.num_envs, ROCKS_PER_ENV, 3, device=self.device
        )

        # Omnidirectional escape: per-episode escape heading (world XY unit vector)
        # and world-frame spawn position. Sampled at reset so all 360° headings are
        # covered across episodes — the policy learns direction-agnostic recovery.
        self._escape_dir  = torch.zeros(self.num_envs, 2, device=self.device)
        self._escape_dir[:, 0] = 1.0   # default +X until first reset
        self._spawn_pos   = torch.zeros(self.num_envs, 2, device=self.device)

        # Failure mode logging (Task 2): global episode counter and CSV path.
        self._episode_counter = 0
        if self.cfg.log_failure_modes:
            exp_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "experiments", "regolith_recovery",
            )
            os.makedirs(exp_root, exist_ok=True)
            self._failure_csv_path = os.path.join(exp_root, "failure_modes.csv")
            # Write header if file doesn't exist yet
            if not os.path.exists(self._failure_csv_path):
                with open(self._failure_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "episode_id", "escaped", "sinkage", "friction",
                        "curriculum_level", "final_dist", "episode_steps",
                    ])
        else:
            self._failure_csv_path = None

    # ── Scene setup ─────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Override Isaac Lab's default USD parsing to skip convex hull computation
        # for all 368 mesh shapes in Mars_Rover.usd. Without this, add_usd(stage)
        # takes 30+ min computing mesh approximations we don't need (we use proxy
        # spheres for collision instead). With skip_mesh_approximation=True it's <30s.
        @classmethod
        def _fast_instantiate(cls):
            import time as _t
            print("[fast_instantiate] FIRED — using skip_mesh_approximation=True path", flush=True)
            from pxr import UsdGeom, UsdPhysics
            from isaaclab.sim._impl.newton_manager import get_current_stage
            stage = get_current_stage()
            up_axis = UsdGeom.GetStageUpAxis(stage)
            builder = nt.ModelBuilder(up_axis=up_axis)

            # ── Runtime USD collision-mesh strip ─────────────────────────────
            # Mars_Rover.usd carries 368 collision-mesh prims. With
            # `skip_mesh_approximation=True` Newton skips convex-hull build
            # but still calls `add_shape_mesh` for each, which builds a Warp
            # mesh + BVH per shape. On this hardware that grinds for ≥1 hour
            # (sometimes hangs across reboots).
            #
            # We never use these collision meshes: wheel contact uses proxy
            # cylinders added in `_newton_init_cb` below, and the chassis
            # talks to sand only via the MPM coupling kernels — nothing in
            # the env queries USD collision shapes.
            #
            # Deactivating CollisionAPI prims in-memory (not on disk) makes
            # add_usd skip them entirely. ~30s init instead of 1h+.
            # If you ever suspect this is hiding a real contact, set
            # STRIP_COLLISION_MESHES=False to revert.
            # STRIP_COLLISION_MESHES=0 keeps the 368 USD collision meshes — for
            # the "why not train on the real rover geometry?" experiment
            # (expect: very long add_usd / per-shape BVH build; see comment).
            STRIP_COLLISION_MESHES = os.environ.get("STRIP_COLLISION_MESHES", "1") != "0"
            if STRIP_COLLISION_MESHES:
                # IMPORTANT: don't `SetActive(False)` on the prim — Mars_Rover.usd
                # uses unified visual+collision mesh prims (same Mesh prim carries
                # both UsdGeom.Mesh schema and UsdPhysics.CollisionAPI). Deactivating
                # would also delete the visual mesh and the rover renders as just
                # proxy shapes.
                #
                # Instead: unbind the CollisionAPI so the prim still renders but
                # Newton's USD parser skips the expensive add_shape_mesh step.
                _t_strip = _t.time()
                stripped = 0
                for prim in stage.Traverse():
                    if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                        prim.RemoveAPI(UsdPhysics.CollisionAPI)
                        # Also remove MeshCollisionAPI if present (carries the
                        # approximation hint that triggers mesh-shape registration).
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        stripped += 1
                print(f"[fast_instantiate] Unbound CollisionAPI from {stripped} mesh prims "
                      f"(visuals preserved) in {_t.time()-_t_strip:.2f}s", flush=True)

            print("[fast_instantiate] Calling builder.add_usd(stage, skip_mesh_approximation=True)…", flush=True)
            _t0 = _t.time()
            builder.add_usd(
                stage,
                skip_mesh_approximation=True,
                collapse_fixed_joints=False,   # keep differential/rocker link bodies separate
                load_visual_shapes=True,        # needed for correct visual transforms
            )
            print(f"[fast_instantiate] add_usd done in {_t.time()-_t0:.1f}s", flush=True)
            NewtonManager.set_builder(builder)
        NewtonManager.instantiate_builder_from_stage = _fast_instantiate

        # Ground plane + proxy cylinder collision shapes via Newton builder callback.
        # The USD mesh collision shapes (329 of them) flood XPBD's contact buffer
        # → NaN. Real wheel mesh approaches hang (high-poly USD mesh parsing).
        # Proxy cylinders use the DEFINITIVE measured wheel geometry
        # (scripts/wheel_geometry.py; see ROVER_WHEEL_RADIUS): tire r=0.0939
        # + half of the 1.1 cm grouser depth → r_eff=0.0994; contact width =
        # rim width 0.1035. Same radius feeds the slip model + spawn formula.
        # Wheel axle is along X in USD frame → rotate cylinder 90° around Y so
        # its Z-axis (cylinder axis in Newton) aligns with wheel axle (X).
        # Width = RIM width 0.1035/2 (the continuous solid surface), not the
        # grouser-blade envelope 0.125: blades are discrete 8 mm plates with
        # gaps, so a solid cylinder at blade width overstates frontal/
        # bulldozing area. Equivalent-cylinder convention: radius from blade
        # penetration (r_eff=0.0994), width from the solid rim.
        WHEEL_HALF_WIDTH = 0.052

        def _newton_init_cb():
            builder = NewtonManager._builder
            n_shapes = getattr(builder, 'shape_count', 0)
            n_bodies = getattr(builder, 'body_count', 0)
            print(f"[Newton Init] bodies={n_bodies}  mesh_shapes={n_shapes}")

            # 90° rotation around Y so cylinder Z-axis aligns with wheel axle (X).
            # Computed inside callback — wp must be fully initialized first.
            _HALF_SQRT2 = 0.7071067811865476
            _WHEEL_ROT  = wp.quat(_HALF_SQRT2, 0.0, _HALF_SQRT2, 0.0)  # xyzw
            _WHEEL_XFORM = wp.transform(wp.vec3(0.0, 0.0, 0.0), _WHEEL_ROT)
            # Middle wheels (CL/CR) joint origin is ~56mm inboard of the actual wheel
            # centre along the axle (USD mesh centre at X≈0.113m vs FL/RL at X≈0.057m).
            # Offset +X in body frame to re-centre the cylinder on the wheel geometry.
            # Both CL and CR use the same offset because their body frames are mirrored
            # (CR quaternion is negated Z) so +X points outward for both.
            _WHEEL_XFORM_MID = wp.transform(wp.vec3(0.056, 0.0, 0.000223), _WHEEL_ROT)

            # Disable collision on all USD mesh shapes — keep VISIBLE for rendering.
            for s in range(n_shapes):
                builder.shape_flags[s] = builder.shape_flags[s] & ~nt.ShapeFlags.COLLIDE_SHAPES

            # Add proxy cylinder on each wheel body matching real wheel geometry.
            # density=0.0 (massless) auto-disables has_particle_collision, so we
            # must explicitly re-enable it so MPM setup_collider registers these
            # cylinders as SDF colliders (COLLIDE_PARTICLES flag).
            cfg = nt.ModelBuilder.ShapeConfig(
                ke=2e3, kd=1e2, kf=1e3, mu=0.75, density=0.0,
                has_shape_collision=True,
                has_particle_collision=True,   # ← critical: enables COLLIDE_PARTICLES
                is_visible=False,
            )
            body_keys = builder.body_key if hasattr(builder, 'body_key') else []
            n_wheels = 0
            n_chassis = 0
            for b in range(n_bodies):
                name = body_keys[b] if b < len(body_keys) else ''
                if 'Drive' in name:
                    xform = _WHEEL_XFORM_MID if ('CL' in name or 'CR' in name) else _WHEEL_XFORM
                    builder.add_shape_cylinder(
                        body=b,
                        xform=xform,
                        radius=WHEEL_RADIUS,
                        half_height=WHEEL_HALF_WIDTH,
                        cfg=cfg,
                    )
                    # Grouser blades (SAND-only colliders; rigid-ground contact
                    # stays on the smooth cylinder). MEASURED geometry
                    # (scripts/wheel_geometry.py): tire r=0.0939; fine shallow
                    # grousers to r=0.1053 (1.1 cm deep, 8 mm plate arc) — a
                    # knobby tread, not paddles. At 1.1 cm ≈ 0.22 voxel these
                    # are mostly sub-resolution; the blades give the SDF a
                    # mild non-axisymmetric texture (best-effort) while the
                    # equivalent-cylinder radius carries the main traction
                    # model. 8 blades (the densest spacing the 5 cm grid can
                    # distinguish; the real tread has dozens of fine grousers
                    # that alias into a ring). SAND_GROUSERS=0 → smooth only.
                    if os.environ.get("SAND_GROUSERS", "1") != "0":
                        n_paddles = int(os.environ.get("SAND_GROUSERS_N", "8"))
                        grouser_cfg = nt.ModelBuilder.ShapeConfig(
                            ke=2e3, kd=1e2, kf=1e3, mu=0.75, density=0.0,
                            has_shape_collision=False,     # no rigid contact
                            has_particle_collision=True,   # sand only
                            is_visible=False,
                        )
                        import math as _math
                        # Blade half-extents from measured geometry: radial
                        # 0.0055 (depth 1.1 cm), tangential 0.004 (8 mm plate),
                        # axial = rim half-width.
                        r_mid = 0.0939 + 0.0055   # blade mid-radius
                        base_off = xform.p
                        for k in range(n_paddles):
                            ang = 2.0 * _math.pi * k / n_paddles
                            # Wheel axle is body-X; blades lie in the body Y-Z
                            # plane, offset radially, long axis radial.
                            cy = r_mid * _math.cos(ang)
                            cz = r_mid * _math.sin(ang)
                            # Rotate the box about body-X so its local +Z
                            # (the hz=radial half-extent) points along the
                            # radial direction (0, cos ang, sin ang). Rotation
                            # about X by θ maps +Z → (0, −sin θ, cos θ), so
                            # θ = ang − π/2.
                            th = ang - _math.pi / 2.0
                            q = wp.quat(_math.sin(th / 2.0), 0.0, 0.0, _math.cos(th / 2.0))
                            p_xf = wp.transform(
                                wp.vec3(float(base_off[0]), float(base_off[1] + cy), float(base_off[2] + cz)),
                                q,
                            )
                            builder.add_shape_box(
                                body=b,
                                xform=p_xf,
                                hx=WHEEL_HALF_WIDTH,  # axial (along axle)
                                hy=0.004,             # tangential: 8 mm plate (measured)
                                hz=0.0055,            # radial: 1.1 cm blade depth (measured)
                                cfg=grouser_cfg,
                            )
                    n_wheels += 1
                elif (name.split('/')[-1] == 'Body' or name.endswith('Body')) \
                        and os.environ.get("SAND_HULL_COLLIDER") == "1":
                    # OPT-IN (2026-06-13). History: the v9 belly box was silently
                    # never added (`name == 'Body'` exact-match vs path-prefixed
                    # body keys → "added 0 chassis sand box" in every log), so
                    # ALL calibrated physics (convergence matrix, trap gradient)
                    # is wheels-only. Binding a hull-sized box turned out to
                    # detonate the bed at teleport-spawn (~20 kg of overlapping
                    # sand evicted instantly → rover shoved ~0.9 m/s, episodes
                    # die in the settle window) and erases the 0.20 m trap.
                    # Until spawn pocket-carving is implemented and the matrix
                    # recalibrated with hull contact, this stays OFF and the
                    # "sand inside the hull" issue is handled at RENDER time
                    # (grains inside the hull volume are hidden).
                    # Chassis sand collider — fills the missing body-soil interaction.
                    # Without this, sand particles pass through the chassis and only
                    # the 6 wheels feel sand resistance. Real granular entrapment
                    # includes belly-pan / hull drag (Bi & Ding 2026).
                    # v12: sized to the REAL hull (USD Body bbox: 0.665×0.490 m,
                    # z ∈ [0, +0.253] above the Body root) plus the belly pan down
                    # to −0.10. The old box (0.50×0.40, z ∈ [−0.10, 0]) covered
                    # only the belly — at deep burial the sand line is up to
                    # +0.11 above root, so sand flowed INSIDE the visual hull
                    # (under-modelled drag + "grains floating in the rover").
                    chassis_xform = wp.transform(
                        wp.vec3(0.0, 0.0, 0.075),   # box spans z ∈ [−0.10, +0.25]
                        wp.quat_identity(),
                    )
                    builder.add_shape_box(
                        body=b,
                        xform=chassis_xform,
                        hx=0.33, hy=0.245, hz=0.175,
                        cfg=cfg,
                    )
                    n_chassis += 1
            print(f"[Newton Init] Disabled mesh collision on {n_shapes} shapes, "
                  f"added {n_wheels} wheel proxy cylinders (r={WHEEL_RADIUS}m, hw={WHEEL_HALF_WIDTH}m), "
                  f"added {n_chassis} chassis sand box.")
            n_env_expected = max(1, n_bodies // 16)  # 16 bodies per env clone
            n_chassis_expected = n_env_expected if os.environ.get("SAND_HULL_COLLIDER") == "1" else 0
            if n_wheels != 6 * n_env_expected or n_chassis != n_chassis_expected:
                sample = [body_keys[b] for b in range(min(n_bodies, 16))]
                raise RuntimeError(
                    f"[Newton Init] Proxy collider binding failed: wheels={n_wheels} "
                    f"(want {6*n_env_expected}), chassis={n_chassis} (want {n_chassis_expected}). "
                    f"Body keys: {sample}. Sand would silently pass through unbound "
                    f"geometry — refusing to continue (this exact failure shipped "
                    f"unnoticed before)."
                )

            # Fix passive joint effort_limit=0 — MuJoCo Warp rejects actfrcrange=[0,0].
            # Rocker/Differential joints in the USD have effort_limit=0 (free joints).
            for i in range(len(builder.joint_effort_limit)):
                if builder.joint_effort_limit[i] <= 0.0:
                    builder.joint_effort_limit[i] = 1.0

            # Joint stiffness/damping — keeps passive joints (Rocker, Differential,
            # Boogie) stiff so the body doesn't collapse under gravity.
            # Same fix as view_rover.py lines 120-123.
            for i in range(builder.joint_dof_count):
                builder.joint_target_ke[i] = 150
                builder.joint_target_kd[i] = 5

            builder.add_ground_plane()
        NewtonManager.add_on_init_callback(_newton_init_cb)

        # Mars terrain static mesh — single copy at world origin (not per-env)
        # NOTE: terrain_merged.usd is 158 MB. Set LOAD_MARS_TERRAIN=1 to enable;
        # leave unset for fast startup during debugging / viewer preview.
        if os.environ.get("LOAD_MARS_TERRAIN") and os.path.exists(MARS_TERRAIN_USD):
            terrain_cfg = sim_utils.UsdFileCfg(usd_path=MARS_TERRAIN_USD)
            terrain_cfg.func("/World/MarsTerrain", terrain_cfg)
            print(f"[Terrain] Loaded: {MARS_TERRAIN_USD}")
        else:
            print(f"[Terrain] Skipped (set LOAD_MARS_TERRAIN=1 to enable).")

        # Spawn ONE set of rocks under the source env (env_0).
        # clone_environments() will replicate them to all envs automatically.
        # Rocks are spawned as visual-only USD prims (no physics colliders).
        # Set SPAWN_ROCKS=1 to enable; disabled by default to avoid USD parser
        # heap corruption on certain Newton versions with multi-mesh USD assets.
        if os.environ.get("SPAWN_ROCKS"):
            print(f"[Rocks] Spawning {ROCKS_PER_ENV} rocks under source env...")
            for rock_j in range(ROCKS_PER_ENV):
                usd_path = ROCK_USDS[rock_j % len(ROCK_USDS)]
                prim_path = f"/World/envs/env_0/Rock_{rock_j}"
                rock_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
                rock_cfg.func(prim_path, rock_cfg)
            print(f"[Rocks] Spawned.")
        else:
            print(f"[Rocks] Skipped (set SPAWN_ROCKS=1 to enable).")

        # MPM sand — initialised AFTER Newton finalises the rigid-body model
        if not self.cfg.skip_mpm:
            NewtonManager.add_on_start_callback(self._init_mpm)
        else:
            print("[Scene] MPM skipped (viewer-only mode)")
            NewtonManager.add_on_start_callback(self._init_viewer_only)

        # copy_from_source=True: Newton loads from USD stage directly (lower VRAM).
        print(f"[Scene] Cloning {self.num_envs} environments ...")
        self.scene.clone_environments(copy_from_source=True)
        print(f"[Scene] Clone done.")
        
        # NOTE: Robot-model gravity comes from SimulationCfg.gravity via NewtonManager.
        # The MPM sand builder needs its own explicit gravity (set in _init_mpm) —
        # nt.ModelBuilder defaults to Earth -9.81 otherwise.

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        print(f"[Scene] Articulations registered.")

        # Build prim path table (only populated when SPAWN_ROCKS is set)
        if os.environ.get("SPAWN_ROCKS"):
            self._rock_prim_paths: list[list[str]] = [
                [f"/World/envs/env_{ei}/Rock_{rj}" for rj in range(ROCKS_PER_ENV)]
                for ei in range(self.num_envs)
            ]
        else:
            self._rock_prim_paths = []

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.70, 0.55))
        light_cfg.func("/World/Light", light_cfg)

    # ── Rock scatter helper ──────────────────────────────────────────────────

    def _scatter_rocks(self, env_ids: Sequence[int]):
        """Randomise rock positions within the sand patch for the given envs."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        for ei in env_ids:
            origin = self.scene.env_origins[ei].cpu()
            for rj, prim_path in enumerate(self._rock_prim_paths[ei]):
                # Random XY within scatter radius, z just above ground
                angle = float(torch.rand(1) * 2.0 * torch.pi)
                r     = float(torch.rand(1) ** 0.5 * ROCK_SCATTER_R)
                x = float(origin[0]) + r * np.cos(angle)
                y = float(origin[1]) + r * np.sin(angle)
                z = float(origin[2]) + 0.02    # 2 cm above ground

                self._rock_positions[ei, rj] = torch.tensor([x, y, z])

                # Move the USD prim
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    from pxr import UsdGeom, Gf
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    xform_op = xformable.AddTranslateOp()
                    xform_op.Set(Gf.Vec3d(x, y, z))

    # ── MPM initialisation ──────────────────────────────────────────────────

    def _init_mpm(self):
        import sys
        def _p(msg): print(msg, flush=True)

        _p("[MPM] _init_mpm starting...")
        robot_model = NewtonManager._model
        device      = NewtonManager._device
        num_envs    = self.num_envs

        # Debug: dump Newton model structure
        _p(f"[Debug] Newton model: body_count={robot_model.body_count}, "
           f"shape_count={robot_model.shape_count}, "
           f"joint_count={robot_model.joint_count}")
        if hasattr(robot_model, 'body_name') and robot_model.body_name:
            _p(f"[Debug] Body names: {robot_model.body_name[:20]}...")
        if hasattr(robot_model, 'shape_geo_type') and robot_model.shape_geo_type is not None:
            _p(f"[Debug] Shape geo types: {robot_model.shape_geo_type}")

        # Ejecta z-headroom for the escaped-particle clamp. 0.10 = the value
        # the v12 matrix was calibrated with (default; training + reported
        # numbers). SAND_EJECTA_HEADROOM=0.45 for cinematic recordings only —
        # it measurably weakens the trap (thrown sand no longer refills the rut).
        self._ejecta_headroom = float(os.environ.get("SAND_EJECTA_HEADROOM", "0.10"))
        _p(f"[MPM] Ejecta headroom = {self._ejecta_headroom} m")

        lo  = np.array([-SAND_HALF_X, -SAND_HALF_Y, 0.0])
        hi  = np.array([ SAND_HALF_X,  SAND_HALF_Y, SAND_DEPTH])
        res = np.array(np.ceil(PPC * (hi - lo) / VOXEL_SIZE), dtype=int)
        cell_size = (hi - lo) / res
        radius    = float(np.max(cell_size) * 0.5)
        mass      = float(np.prod(cell_size) * 1700.0)
        _p(f"[MPM] Grid res={res}, cell_size={cell_size}, radius={radius:.4f}, mass={mass:.4f}")

        # CRITICAL: ModelBuilder defaults to Earth gravity (-9.81). The robot model
        # gets Mars gravity from SimulationCfg via NewtonManager, but this sand
        # builder is constructed by hand — without an explicit gravity it simulates
        # the bed at Earth weight (2.64× confining pressure → ~2.64× frictional
        # shear strength) while the rover weighs Mars. All runs before 2026-06-12
        # had this mismatch (sand: -9.81, rover: -3.72). Derive from cfg.sim so
        # both phases always share one gravity value.
        # SAND_GRAVITY env var reproduces the pre-2026-06-12 legacy behaviour
        # (Earth-gravity bed) for A/B attribution studies; default = matched.
        sand_gravity = float(os.environ.get("SAND_GRAVITY", self.cfg.sim.gravity[2]))
        sand_builder = nt.ModelBuilder(up_axis=nt.Axis.Z, gravity=sand_gravity)
        _p(f"[MPM] Sand gravity = {sand_gravity} m/s² (sim gravity = {self.cfg.sim.gravity[2]})")
        env_origins_np = self.scene.env_origins.cpu().numpy()

        self._particle_env_starts = []
        for i, origin in enumerate(env_origins_np):
            start = len(sand_builder.particle_q)
            self._particle_env_starts.append(start)
            sand_builder.add_particle_grid(
                pos=wp.vec3(
                    float(origin[0]) + lo[0],
                    float(origin[1]) + lo[1],
                    float(origin[2]) + lo[2],
                ),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=int(res[0]) + 1,
                dim_y=int(res[1]) + 1,
                dim_z=int(res[2]) + 1,
                cell_x=float(cell_size[0]),
                cell_y=float(cell_size[1]),
                cell_z=float(cell_size[2]),
                mass=mass,
                # v12: jitter 2.0r → 0.5r. Full-cell jitter (the Newton flowing-
                # sand demo value) builds an overlap/void disordered lattice that
                # consolidates ~36% under gravity — the bed couldn't statically
                # bear the rover (sank to the rigid floor with zero action).
                # particle_volume = (2r)³ exactly fills the cell, so a near-
                # lattice packing starts at volume fraction ≈ 1 and bears load
                # through the solver's unilateral incompressibility immediately.
                jitter=0.5 * radius,
                radius_mean=radius,
            )

        self._particle_env_starts = np.array(self._particle_env_starts, dtype=np.int32)
        self._particles_per_env   = len(sand_builder.particle_q) // num_envs

        # Warp array of env origins — used by clamp_escaped_particles kernel
        self._env_origins_wp = wp.array(
            env_origins_np[:, :3].astype(np.float32),
            dtype=wp.vec3, device=device,
        )

        _p(f"[MPM] Finalizing sand model ({len(sand_builder.particle_q)} particles)...")
        self.sand_model = sand_builder.finalize(device=device)
        # Per-particle friction array (size = total particles across all envs).
        # MUST be a wp.array BEFORE solver construction, because the solver copies
        # `model.particle_mu` into its internal `material_parameters.friction` array
        # exactly once at init and never re-reads `model.particle_mu` afterwards.
        # Setting `model.particle_mu = scalar` after init is a silent no-op.
        # Per-env friction DR is implemented by writing slices of the solver's
        # `material_parameters.friction` array directly (see _set_env_friction).
        total_particles = len(sand_builder.particle_q)
        self.sand_model.particle_mu = wp.full(total_particles, 0.9, dtype=float, device=device)
        # particle_ke = compression penalty stiffness (NOT a physical bulk modulus
        # — see v10 notes in CLAUDE.md). History: Newton reference uses 1e15
        # (near-incompressible → trampoline bounce); v7 tried 1e8 quoting compacted-
        # sand bulk modulus, still rubber-mat stiff; v10 restored 2e5 Pa (the
        # April-16 value) which gives visibly correct wheel-sand compliance.
        # Implicit MPM solver is unconditionally stable at any ke; pick for
        # visible compliance, not textbook modulus.
        self.sand_model.particle_ke = 2.0e5
        _p("[MPM] Sand model finalized.")

        mpm_opt = SolverImplicitMPM.Options()
        mpm_opt.voxel_size        = VOXEL_SIZE
        # v12 ROOT-CAUSE FIX: at the old 30 iterations / 1e-5 the rheology solve
        # exited unconverged on this 24-voxel-deep bed — the pressure constraint
        # never propagated to the ground, so the bed leaked volume EVERY step
        # (~30% column collapse; rover sank to the rigid floor with zero action).
        # Calibration (scripts/bed_calibration.py, 2026-06-13): leak is per-step
        # not per-sim-second, insensitive to particle volume/jitter/cf/basis;
        # 100 iters + 1e-7 → 27% → 6% compaction at +50% MPM cost (47 vs 31
        # ms/step @ 0.5M particles). Override via env vars to retune on the 4090.
        mpm_opt.tolerance         = float(os.environ.get("MPM_TOLERANCE", "1.0e-7"))
        mpm_opt.grid_type         = "sparse"
        mpm_opt.transfer_scheme   = "apic"   # angular-momentum-conserving PIC — better wheel↔sand momentum transfer
        mpm_opt.strain_basis      = "P0"     # Q1 measured WORSE (48% leak) — keep P0
        mpm_opt.max_iterations    = int(os.environ.get("MPM_MAX_ITERATIONS", "100"))
        mpm_opt.critical_fraction = 0.025
        mpm_opt.hardening         = 5.0
        mpm_opt.air_drag          = 1.0
        # Cohesion (shear yield stress, Pa) — see cfg.sand_yield_stress.
        # Env var takes precedence for sweep workflows.
        _ys_env = os.environ.get("MPM_YIELD_STRESS")
        mpm_opt.yield_stress = (
            float(_ys_env) if _ys_env is not None else float(self.cfg.sand_yield_stress)
        )
        _p(f"[MPM] yield_stress = {mpm_opt.yield_stress} Pa")

        _p("[MPM] Creating MPM model...")
        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_opt)
        _p("[MPM] Setting up collider...")
        mpm_model.setup_collider(model=robot_model, ground_height=0.0)
        _p("[MPM] Creating solver...")
        self.mpm_solver  = SolverImplicitMPM(mpm_model, mpm_opt)
        _p("[MPM] Creating state...")
        self.sand_state  = self.sand_model.state()
        _p("[MPM] Enriching state...")
        self.mpm_solver.enrich_state(self.sand_state)

        n_col = mpm_model.collider_body_count
        _p(f"[MPM] collider_body_count={n_col}")
        if n_col == 0:
            raise RuntimeError(
                "[MPM] collider_body_count == 0 — wheel↔sand coupling is DISABLED. "
                "Training would proceed on a rigid ground plane with 64k decorative "
                "particles, producing plausible-looking but physically meaningless "
                "results. Check that proxy cylinders in _setup_scene have "
                "has_particle_collision=True and that mpm_model.setup_collider was "
                "called with the correct robot model."
            )
        mpm_dt = NewtonManager._dt if hasattr(NewtonManager, "_dt") else 1.0 / 50.0

        # ── Bed pre-settle (v12) ─────────────────────────────────────────────
        # add_particle_grid builds a LOOSE jittered lattice that has never
        # consolidated. Under gravity the bed collapses for several seconds; a
        # rover standing on it rides the collapse down (measured 2026-06-13:
        # chassis −0.35 m in 5 s with ZERO action, identical at 0/1/2 kPa
        # cohesion — i.e. not a shear-strength issue). Pre-Mars-gravity-fix
        # runs masked this because Earth-weight sand consolidated faster and
        # was ~2.6× stronger relative to the Mars-weight rover.
        # Fix: settle the bed ONCE here with colliders parked far away, then
        # cache the consolidated state as the per-episode reset state.
        if self.sand_state.body_q is not None and n_col > 0:
            bq = wp.to_torch(self.sand_state.body_q)
            bq[:, :3]  = torch.tensor([0.0, 0.0, 1000.0], device=bq.device)
            bq[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=bq.device)
            if self.sand_state.body_qd is not None:
                wp.to_torch(self.sand_state.body_qd).zero_()
        presettle_max = int(os.environ.get("BED_PRESETTLE_STEPS", "600"))
        _p(f"[MPM] Pre-settling bed (max {presettle_max} steps @ dt={mpm_dt})...")
        for i in range(presettle_max):
            self.mpm_solver.step(self.sand_state, self.sand_state,
                                 contacts=None, control=None, dt=mpm_dt)
            # Keep the box walls during pre-settle — at runtime this clamp runs
            # every step, so the settled state must be consolidated WITHIN the
            # box, not a laterally-spread pile (first attempt without the clamp
            # measured 0.274 m of apparent "compaction", mostly sideways spill).
            wp.launch(
                clamp_escaped_particles,
                dim=int(self.sand_model.particle_count),
                inputs=[
                    self.sand_state.particle_q,
                    self._env_origins_wp,
                    int(self._particles_per_env),
                    float(SAND_HALF_X),
                    float(SAND_HALF_Y),
                    float(SAND_DEPTH),
                    self._ejecta_headroom,
                ],
                device=device,
            )
            if (i + 1) % 25 == 0:
                v = torch.linalg.norm(
                    wp.to_torch(self.sand_state.particle_qd), dim=1).mean().item()
                _p(f"[MPM]   pre-settle {i+1}: mean |v| = {v:.4f} m/s")
                if v < 0.005:
                    break
        wp.to_torch(self.sand_state.particle_qd).zero_()

        # Measure the settled surface height (99th-percentile particle z of
        # env 0, origin-relative). Compaction lowers the surface below the
        # as-built SAND_DEPTH; spawn-sinkage and the privileged wheel-burial
        # signal must reference the REAL surface or prescribed sinkage lies.
        q_torch = wp.to_torch(self.sand_state.particle_q)
        z_env0  = q_torch[: self._particles_per_env, 2] - float(env_origins_np[0][2])
        self._sand_surface_z = float(torch.quantile(z_env0, 0.99).item())
        _p(f"[MPM] Settled sand surface: {self._sand_surface_z:.3f} m "
           f"(as-built {SAND_DEPTH} m → compaction {SAND_DEPTH - self._sand_surface_z:.3f} m)")

        # Cache the CONSOLIDATED bed as the per-episode reset state (virgin
        # settled bed, zero velocities, no rover cavity baked in).
        self._sand_q0  = wp.clone(self.sand_state.particle_q)
        self._sand_qd0 = wp.clone(self.sand_state.particle_qd)

        # Now place the real collider poses and evict any particles that
        # overlap the rover at its current pose.
        if self.sand_state.body_q is not None:
            robot_state_init = NewtonManager._state_0
            wp.copy(self.sand_state.body_q,
                    robot_state_init.body_q,
                    count=min(n_col, robot_state_init.body_q.shape[0]))
            if self.sand_state.body_qd is not None:
                wp.copy(self.sand_state.body_qd,
                        robot_state_init.body_qd,
                        count=min(n_col, robot_state_init.body_qd.shape[0]))
        _p("[MPM] Running project_outside...")
        self.mpm_solver.project_outside(self.sand_state, self.sand_state, dt=mpm_dt)
        _p("[MPM] Initial project_outside complete.")

        max_nodes = 1 << 20
        self._col_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._col_imp_pos  = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._col_imp_ids  = wp.full(max_nodes, value=-1, dtype=int, device=device)
        self._body_sand_f  = wp.zeros(robot_model.body_count,
                                      dtype=wp.spatial_vector, device=device)
        self._col_body_id  = mpm_model.collider.collider_body_index
        self._n_col_bodies = n_col

        # Per-env settle counter: sand-force magnitude is clamped while > 0.
        # Set by _reset_idx, decremented once per physics step inside
        # _inject_sand_forces. Stored as int32 Warp array for kernel use.
        self._bodies_per_env  = int(robot_model.body_count // self.num_envs)
        self._settle_counter_wp = wp.zeros(self.num_envs, dtype=int, device=device)

        self._collect_mpm_impulses()
        self._mpm_ready = True
        self._init_sand_visual()

        # Patch NewtonManager.step to call MPM post-XPBD processing.
        # Use a guard to prevent re-patching if multiple env instances are created.
        if not hasattr(NewtonManager, '_mpm_patched'):
            original_step_fn = NewtonManager.step.__func__
            NewtonManager._mpm_envs = []  # Track all env instances
            NewtonManager._viewer_envs = []  # Track viewer instances (viewer-only mode)
            NewtonManager._mpm_patched = True

            @classmethod          # type: ignore[misc]
            def _patched_step(cls):
                # PRE-XPBD: inject sand forces from the most recent collected impulses
                # BEFORE the rigid solver step. Was previously called from
                # _pre_physics_step (once per policy step); with decimation=2 that
                # meant only the LAST physics substep's impulses were ever injected
                # — the first substep's impulses were silently overwritten.
                # Moved here so each physics substep gets its own freshly-collected
                # sand force applied → 2× contact resolution restored.
                for env_ref in NewtonManager._mpm_envs:
                    if env_ref._mpm_ready:
                        env_ref._inject_sand_forces()
                original_step_fn(cls)
                # MPM post-processing
                for env_ref in NewtonManager._mpm_envs:
                    if env_ref._mpm_ready:
                        env_ref._mpm_post_step()
                # Viewer rendering (for viewer-only mode without MPM)
                for env_ref in NewtonManager._viewer_envs:
                    if env_ref._viewer and env_ref._viewer.is_running():
                        env_ref._render_frame += 1
                        if env_ref._render_frame % 2 == 0:
                            try:
                                state_0 = NewtonManager._state_0
                                env_ref._viewer.begin_frame(env_ref._render_frame / 50.0)
                                env_ref._viewer.log_state(state_0)
                                env_ref._viewer.end_frame()
                            except Exception:
                                pass

            NewtonManager.step = _patched_step

        # ── Keep sand forces alive across substeps ──────────────────────────
        # NewtonManager's substep loop calls state.clear_forces() after every
        # substep (states ping-pong), so a once-per-frame injection is consumed
        # by substep 1 only — the ÷num_substeps dilution bug. Wrapping
        # clear_forces to re-add the sand wrench after each clear makes the
        # injected force CONTINUOUS over the frame: constant force ⇒ exactly
        # the collected impulse J, without the first-substep kN-class pulse
        # (an earlier ×num_substeps fix delivered J impulsively and ratcheted
        # buried rovers upward at ~4 cm/s with net force ≈ 0).
        if not hasattr(NewtonManager, "_sand_clear_forces_patched"):
            NewtonManager._sand_clear_forces_patched = True
            for _st in (NewtonManager._state_0, NewtonManager._state_1):
                _orig_clear = _st.clear_forces

                def _make_clear_wrapper(orig, st_ref):
                    def _clear_and_readd():
                        orig()
                        if st_ref.body_f is None:
                            return
                        for env_ref in NewtonManager._mpm_envs:
                            if env_ref._mpm_ready:
                                wp.launch(
                                    kernel=_add_spatial_forces,
                                    dim=st_ref.body_f.shape[0],
                                    inputs=[env_ref._body_sand_f, 1.0, st_ref.body_f],
                                    device=env_ref.device,
                                )
                    return _clear_and_readd

                _st.clear_forces = _make_clear_wrapper(_orig_clear, _st)

        # Register this env instance for MPM post-processing
        NewtonManager._mpm_envs.append(self)

        n = self.sand_model.particle_count
        print(f"[MPM] Sand: {n} particles | {n // num_envs} per env | "
              f"{num_envs} envs | voxel={VOXEL_SIZE}m")

    # ── MPM coupling helpers ─────────────────────────────────────────────────

    def _collect_mpm_impulses(self):
        result = self.mpm_solver.collect_collider_impulses(self.sand_state)
        if result is None:
            return
        impulses, pos, cids = result
        if impulses is None or len(impulses) == 0:
            return
        n = min(len(impulses), len(self._col_impulses))
        self._col_imp_ids.fill_(-1)
        self._col_impulses[:n].assign(impulses[:n])
        self._col_imp_pos[:n].assign(pos[:n])
        self._col_imp_ids[:n].assign(cids[:n])

    def _inject_sand_forces(self):
        if not self._mpm_ready:
            return
        robot_model = NewtonManager._model
        state_0     = NewtonManager._state_0
        mpm_dt      = NewtonManager._dt

        # Zero body_f before accumulating sand forces. Newton does NOT auto-clear
        # body_f between steps (reference example_mpm_twoway_coupling.py:202
        # calls state_0.clear_forces() at the start of each substep for this
        # reason). Without this zero, atomic_add into body_f compounds across
        # every step — sand forces grow unbounded over training.
        # We only clear body_f (not joints) because Isaac Lab applies joint
        # control through a separate actuation path; body_f carries only the
        # external forces we inject.
        state_0.body_f.zero_()

        self._body_sand_f.zero_()
        wp.launch(
            compute_body_forces,
            dim=len(self._col_imp_ids),
            inputs=[
                mpm_dt,
                self._col_imp_ids,
                self._col_impulses,
                self._col_imp_pos,
                self._col_body_id,
                state_0.body_q,
                robot_model.body_com,
                self._body_sand_f,
            ],
            device=self.device,
        )
        # Post-reset settle: clamp per-body sand force magnitude while
        # env_settle_counter > 0, then decrement it. Prevents the MPM
        # penetration-resolution impulse from launching the rover on the
        # first few physics steps after teleport-spawn.
        wp.launch(
            clamp_sand_forces_during_settle,
            dim=robot_model.body_count,
            inputs=[
                self._body_sand_f,
                self._settle_counter_wp,
                int(self._bodies_per_env),
                float(self.cfg.settle_force_cap),
                float(self.cfg.settle_torque_cap),
            ],
            device=self.device,
        )
        wp.launch(
            decrement_settle_counter,
            dim=self.num_envs,
            inputs=[self._settle_counter_wp],
            device=self.device,
        )

        # scale=1.0: the clear_forces wrapper (see _init_mpm) re-adds the sand
        # wrench after every substep clear, so this force acts CONTINUOUSLY
        # over the frame — total impulse = collected J, no pulse ratcheting.
        wp.launch(
            kernel=_add_spatial_forces,
            dim=robot_model.body_count,
            inputs=[self._body_sand_f, 1.0, state_0.body_f],
            device=self.device,
        )

    def _mpm_post_step(self):
        robot_model = NewtonManager._model
        state_0     = NewtonManager._state_0
        mpm_dt      = NewtonManager._dt

        if self.sand_state.body_q is not None and self._n_col_bodies > 0:
            n = min(self._n_col_bodies, state_0.body_q.shape[0])
            # subtract_body_force: copies body_q/qd into sand_state while removing
            # the previously applied sand force from the velocity — prevents
            # double-counting in the MPM collider boundary conditions.
            # This matches the reference pattern in example_mpm_twoway_coupling.py.
            wp.launch(
                subtract_body_force,
                dim=n,
                inputs=[
                    mpm_dt,
                    state_0.body_q,
                    state_0.body_qd,
                    self._body_sand_f,
                    robot_model.body_inv_inertia,
                    robot_model.body_inv_mass,
                    self.sand_state.body_q,
                    self.sand_state.body_qd,
                ],
                device=self.device,
            )

        self.mpm_solver.step(
            self.sand_state, self.sand_state,
            contacts=None, control=None, dt=mpm_dt,
        )

        # Evict particles that ended up inside collider geometry (wheels, chassis)
        # after this substep. Without this, buried particles have no SDF normal →
        # no contact impulse → rover floats with no real wheel-sand coupling.
        self.mpm_solver.project_outside(self.sand_state, self.sand_state, dt=mpm_dt)

        # Clamp any escaped particle back into its env's sand box.
        # Without this, even one particle drifting away causes the sparse VDB
        # bounding box to balloon each step → volume_builder OOM around step 5.
        wp.launch(
            clamp_escaped_particles,
            dim=int(self.sand_model.particle_count),
            inputs=[
                self.sand_state.particle_q,
                self._env_origins_wp,
                int(self._particles_per_env),
                float(SAND_HALF_X),
                float(SAND_HALF_Y),
                float(SAND_DEPTH),
                self._ejecta_headroom,
            ],
            device=self.device,
        )

        self._collect_mpm_impulses()
        self._render_sand_particles()

    def _init_viewer_only(self):
        """Lightweight viewer init — no MPM, just rigid-body visualization."""
        self._viewer = None
        self._render_frame = 0

        if os.environ.get("ENTRAPMENT_NO_VIEWER") == "1":
            print("[Viewer-Only] ENTRAPMENT_NO_VIEWER=1 — disabled (training).")
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("[Viewer-Only] No display — disabled.")
            return
        try:
            from newton.viewer import ViewerGL
            robot_model = NewtonManager._model
            n_bodies = getattr(robot_model, 'body_count', 0)
            n_shapes = getattr(robot_model, 'shape_count', 0)
            print(f"[Viewer-Only] Newton model: {n_bodies} bodies, {n_shapes} shapes")

            self._viewer = ViewerGL(width=1440, height=900, vsync=False)
            self._viewer.set_model(robot_model)
            self._viewer.show_collision = True
            self._viewer.show_joints = True
            self._viewer.set_camera(
                pos=wp.vec3(2.5, -2.5, 1.5),
                pitch=-20.0,
                yaw=135.0,
            )
            # Register this env for viewer rendering (same pattern as MPM patch)
            if not hasattr(NewtonManager, '_viewer_envs'):
                NewtonManager._viewer_envs = []
            NewtonManager._viewer_envs.append(self)
            print(f"[Viewer-Only] ViewerGL open — {n_bodies} bodies, no sand")
        except Exception as e:
            import traceback
            print(f"[Viewer-Only] Failed: {e}")
            traceback.print_exc()
            self._viewer = None

    def _init_sand_visual(self):
        self._viewer       = None
        self._vis_stride   = 1
        self._render_frame = 0
        self._grains       = None   # cosmetic grain rendering (set up below if viewer opens)

        # Training must NEVER open the live ViewerGL: rendering 60k+ sand points
        # (or 1.5M grains) every few frames at <1 FPS wastes huge GPU time over a
        # multi-hour run AND the partial 16-env render makes the rover look like
        # it floats (no sand drawn). train.py sets ENTRAPMENT_NO_VIEWER=1; eval /
        # probe scripts leave it unset so they still get the viewer. The
        # headless arg alone does NOT suppress ViewerGL (it only gates DISPLAY).
        if os.environ.get("ENTRAPMENT_NO_VIEWER") == "1":
            print("[Sand Visual] ENTRAPMENT_NO_VIEWER=1 — viewer disabled (training).")
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("[Sand Visual] No display — viewer disabled.")
            return
        try:
            from newton.viewer import ViewerGL
        except Exception as e:
            print(f"[Sand Visual] ViewerGL unavailable: {e}")
            return

        n = self.sand_model.particle_count
        self._vis_stride = max(1, n // 60_000)
        n_vis = n // self._vis_stride
        # Grain rendering (cosmetic only — physics unchanged). Each env-0
        # physics particle is split into SAND_GRAINS_PPP small render grains
        # advected by the solver's own update_render_grains (the look of
        # Newton's example_mpm_grain_rendering / what makes the demo sand
        # look like sand instead of voxel blobs). SAND_GRAINS=0 restores the
        # raw-particle debug view.
        self._grains = None
        self._grains_enabled = os.environ.get("SAND_GRAINS", "1") != "0"
        robot_model = NewtonManager._model
        try:
            self._viewer = ViewerGL(width=1440, height=900, vsync=False)
            self._viewer.set_model(robot_model)
            self._viewer.show_collision = True
            self._viewer.show_joints = True

            # Debug: check if Newton model has shapes
            n_shapes = getattr(robot_model, 'shape_count', 0)
            n_bodies = getattr(robot_model, 'body_count', 0)
            print(f"[Sand Visual] Newton model: {n_bodies} bodies, {n_shapes} shapes")

            # Camera positioned to see env_0 clearly from above-behind.
            # env_0 origin is at (0,0,0); rover spawns at z~0.2m.
            self._viewer.set_camera(
                pos=wp.vec3(2.5, -2.5, 1.5),
                pitch=-20.0,   # degrees — tilt down to see sand bed
                yaw=135.0,     # degrees — look from +X,-Y corner toward origin
            )
            print(f"[Sand Visual] ViewerGL open — robots:{self.num_envs}  sand pts:{n_vis}  "
                  f"shapes:{n_shapes}")
            if self._grains_enabled:
                self._setup_grain_rendering()
        except Exception as e:
            print(f"[Sand Visual] ViewerGL setup failed: {e}")
            self._viewer = None

    def _setup_grain_rendering(self):
        """Cosmetic sub-particle grains for env 0 (the camera target).

        Splits each physics particle into N small grains sampled inside the
        particle's cube, advected each render frame by the solver's
        update_render_grains (APIC-local + grid PIC + ellipsoid projection).
        Pure rendering — the MPM state is never modified.
        """
        try:
            ppp = int(os.environ.get("SAND_GRAINS_PPP", "3"))
            ppe = int(self._particles_per_env)
            # How many envs to draw sand under. Particles are laid out env-major
            # (env e owns rows [e*ppe, (e+1)*ppe)), so rendering the first
            # n_render_envs envs is just a longer slice. Default: ALL envs when
            # there are few (≤4 → nice multi-rover view), else env-0 only (16+
            # envs would be millions of grains → viewer chokes). SAND_GRAINS_ENVS
            # overrides. The hull-grain mask is env-0-specific so it's only
            # applied when rendering a single env.
            default_envs = self.num_envs if self.num_envs <= 4 else 1
            self._grain_render_envs = max(1, min(
                self.num_envs,
                int(os.environ.get("SAND_GRAINS_ENVS", str(default_envs)))))
            n_render = ppe * self._grain_render_envs
            grains_full = self.mpm_solver.sample_render_grains(self.sand_state, ppp)
            self._grains_full = grains_full
            self._grains = grains_full[:n_render]
            n_g = n_render * ppp
            grain_r = float(VOXEL_SIZE) / (3.0 * ppp)
            dev = self.sand_state.particle_q.device
            # ±30% size variation — uniform grain size reads as synthetic.
            # base_radii holds the true sizes; _grain_radii is rewritten every
            # render frame by the hull mask (radius 0 = hidden inside the hull).
            _rr = np.random.default_rng(1)
            self._grain_base_radii = wp.array(
                (grain_r * _rr.uniform(0.7, 1.3, size=n_g)).astype(np.float32),
                dtype=wp.float32, device=dev)
            self._grain_radii = wp.clone(self._grain_base_radii)
            # Mars-sand palette with per-grain jitter — a flat single colour is
            # what makes point clouds look like plastic. (One-time, host-side.)
            rng = np.random.default_rng(0)
            # Martian regolith (iron-oxide butterscotch-red) — matches the
            # raw-particle fallback palette and the paper's Mars-analogue claim.
            base = np.array([0.62, 0.33, 0.20], dtype=np.float32)
            cols = base[None, :] + rng.uniform(-0.05, 0.05, size=(n_g, 3)).astype(np.float32)
            cols += rng.uniform(-0.06, 0.06, size=(n_g, 1)).astype(np.float32)  # luminance jitter
            self._grain_colors = wp.array(np.clip(cols, 0.0, 1.0), dtype=wp.vec3, device=dev)
            # Previous-frame particle positions: our MPM steps in place, so we
            # keep our own snapshot for update_render_grains' state_prev.
            self._grain_prev_q = wp.clone(self.sand_state.particle_q)
            self._grain_err_once = False
            print(f"[Sand Visual] Grain rendering ON — {n_g} grains across "
                  f"{self._grain_render_envs}/{self.num_envs} env(s) "
                  f"(ppp={ppp}, r={grain_r*1000:.1f} mm). SAND_GRAINS_ENVS to change how "
                  f"many envs show sand, SAND_GRAINS=0 for raw debug view, SAND_GRAINS_PPP "
                  f"to trade density vs FPS.")
        except Exception as e:
            print(f"[Sand Visual] Grain rendering unavailable ({e}) — using raw particles.")
            self._grains = None

    def _render_sand_particles(self):
        if self._viewer is None or not self._viewer.is_running():
            return
        self._render_frame += 1
        if self._render_frame % 3 != 0:
            return
        try:
            robot_state = NewtonManager._state_0
            self._viewer.begin_frame(self._render_frame / 50.0)
            self._viewer.log_state(robot_state)

            if self._grains is not None:
                # Advect the cosmetic grains by the motion since the last
                # render frame (we render every 3rd physics step). state_prev
                # only needs particle_q — pass our snapshot through a shim.
                from types import SimpleNamespace
                prev = SimpleNamespace(particle_q=self._grain_prev_q)
                dt_render = 3.0 * (NewtonManager._dt if hasattr(NewtonManager, "_dt") else 0.02)
                try:
                    self.mpm_solver.update_render_grains(
                        prev, self.sand_state, self._grains, dt_render)
                    wp.copy(self._grain_prev_q, self.sand_state.particle_q)
                except Exception as ge:
                    if not self._grain_err_once:
                        self._grain_err_once = True
                        print(f"[Sand Visual] grain update failed ({ge}) — reverting to raw particles")
                    self._grains = None
            if self._grains is not None:
                # Hide grains inside the (physics-transparent) hull volume —
                # render-only mask, re-evaluated against the live rover pose.
                # Only valid for the single-env view (mask uses env-0's pose);
                # when drawing multiple envs, skip it and show raw grains.
                rp = self.root_pos[0] if (hasattr(self, "root_pos")
                                          and self._grain_render_envs == 1) else None
                if rp is not None:
                    rq = wp.to_torch(self.robot.data.root_link_quat_w)[0]
                    wp.launch(
                        _mask_grains_in_hull,
                        dim=self._grain_radii.shape[0],
                        inputs=[
                            self._grains.flatten(),
                            self._grain_base_radii,
                            self._grain_radii,
                            wp.vec3(float(rp[0]), float(rp[1]), float(rp[2])),
                            wp.quat(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])),
                        ],
                        device=self.device,
                    )
                self._viewer.log_points(
                    name="sand",
                    points=self._grains.flatten(),
                    radii=self._grain_radii,
                    colors=self._grain_colors,
                )
            else:
                sand_pts = self.sand_state.particle_q[::self._vis_stride]
                n_vis = len(sand_pts)
                # Reuse pre-allocated arrays to avoid VRAM fragmentation
                if not hasattr(self, '_vis_radii') or len(self._vis_radii) != n_vis:
                    self._vis_radii  = wp.full(n_vis, VOXEL_SIZE * 0.6,
                                               dtype=wp.float32, device=sand_pts.device)
                    self._vis_colors = wp.full(n_vis, wp.vec3(0.62, 0.30, 0.20),
                                               dtype=wp.vec3, device=sand_pts.device)

                self._viewer.log_points(
                    name="sand",
                    points=sand_pts,
                    radii=self._vis_radii,
                    colors=self._vis_colors,
                )
            self._viewer.end_frame()
        except Exception:
            pass

    # ── DirectRLEnv interface ────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # NOTE: sand-force injection moved into NewtonManager._patched_step so it
        # runs every physics substep (decimation=2 → 2 substeps per policy step).
        # See big comment in _patched_step. Calling it here would re-inject the
        # same impulse twice per step and double the per-substep coupling force.

    def _apply_action(self) -> None:
        # Zero actions for envs that have already crossed the escape threshold this
        # policy step but haven't been reset yet — eliminates the ~0.02 m post-escape
        # drift between done detection and _reset_idx.
        root_xy   = wp.to_torch(self.robot.data.root_link_pos_w)[:, :2]
        rel_pos   = root_xy - self._spawn_pos
        axial     = (rel_pos * self._escape_dir).sum(dim=-1)
        live_mask = (axial <= ESCAPE_DISTANCE).float().unsqueeze(-1)

        # Drive: velocity targets (with domain-randomized motor gain)
        drive_targets = self.actions[:, :6] * DRIVE_VEL_LIMIT * self._motor_gain * live_mask
        self.robot.set_joint_velocity_target(drive_targets, joint_ids=self._drive_ids)
        # Steer: position targets
        steer_targets = self.actions[:, 6:] * STEER_POS_LIMIT * live_mask
        self.robot.set_joint_position_target(steer_targets, joint_ids=self._steer_ids)

    def _get_observations(self) -> dict:
        self.joint_vel  = wp.to_torch(self.robot.data.joint_vel)
        self.joint_pos  = wp.to_torch(self.robot.data.joint_pos)
        self.root_vel_b = wp.to_torch(self.robot.data.root_com_lin_vel_b)
        self.ang_vel_b  = wp.to_torch(self.robot.data.root_com_ang_vel_b)
        self.grav_b     = wp.to_torch(self.robot.data.projected_gravity_b)
        self.lin_acc_w  = wp.to_torch(self.robot.data.body_com_lin_acc_w)[:, 0, :]

        # Drive wheel velocities normalised  (N, 6)
        drive_vel = self.joint_vel[:, self._drive_ids] / DRIVE_VEL_LIMIT

        # Per-wheel slip ratio  (N, 6)
        wheel_speed = drive_vel * DRIVE_VEL_LIMIT * WHEEL_RADIUS
        v_x   = self.root_vel_b[:, 0:1].expand(-1, 6)
        eps   = 0.01
        denom = torch.maximum(torch.abs(wheel_speed),
                              torch.abs(v_x).clamp(min=eps))
        slip  = ((wheel_speed - v_x) / denom).clamp(-1.0, 1.0)   # (N, 6)

        # Steering joint positions normalised  (N, 4)
        steer_pos = self.joint_pos[:, self._steer_ids] / STEER_POS_LIMIT

        # IMU specific force in BODY frame, normalised by local |g|  (N, 3).
        # A physical IMU measures f_b = R⁻¹(a_w − g_w): body frame, gravity
        # included. The old obs (world-frame coordinate acceleration / 9.81) was
        # not measurable by any onboard sensor and used Earth g under Mars
        # gravity. Normalising by the local g magnitude makes the at-rest
        # reading a unit vector on any planet — deployment maps the real
        # accelerometer output to this obs by dividing by local g.
        quat_w  = wp.to_torch(self.robot.data.root_link_quat_w)  # (x,y,z,w)
        spec_force_w = self.lin_acc_w - self._g_vec_w
        imu_acc = quat_apply_inverse(quat_w, spec_force_w) / self._g_mag

        # Tilt indicator  (N, 1)
        grav_z = self.grav_b[:, 2:3]

        # Per-wheel torque delta (N, 6) — step-wise change in normalised torque.
        # REPLACES absolute torque norm which was always ~1.0 (motors always saturated
        # vs sand resistance). Delta captures struggle dynamics: high delta = wheel
        # oscillating between stall and slip; low delta = steady state (buried or free).
        applied_torque = wp.to_torch(self.robot.data.applied_effort)
        drive_torque = applied_torque[:, self._drive_ids]
        effort_limits = wp.to_torch(self.robot.data.joint_effort_limits)
        drive_effort_lim = effort_limits[:, self._drive_ids].clamp(min=0.1)
        drive_torque_norm = (drive_torque / drive_effort_lim).clamp(-1.0, 1.0)
        drive_torque_delta_pw = (drive_torque_norm - self._prev_drive_torque_norm).abs().clamp(0.0, 2.0)

        # Entrapment detection flag  (N, 1)
        # Count consecutive steps with low FORWARD PROGRESS and high slip.
        # CRITICAL FIX: previously used |v_x_body|, which clears the flag during
        # rocking (|±0.5| > 0.15 → "not stuck") — the exact recovery behavior we
        # want to reward. Also failed on backward drift (|-0.32| > 0.15 → "not stuck")
        # which was the policy's failure mode at step 15k of the old run.
        # Now: project world-frame velocity onto per-episode escape heading and
        # check if forward progress (signed, not absolute) is below threshold.
        # This correctly fires during: stationary, rocking, backward drift.
        # Does not fire during: genuine forward escape.
        v_world_xy_obs = wp.to_torch(self.robot.data.root_com_lin_vel_w)[:, :2].nan_to_num(0.0)
        v_forward = (v_world_xy_obs * self._escape_dir).sum(dim=-1)
        v_x_scalar = self.root_vel_b[:, 0].abs()  # kept for slip-anomaly (body-frame)
        mean_slip  = torch.mean(torch.abs(slip), dim=-1)
        is_stuck   = (v_forward < self._ENTRAP_VX_THRESH) & (mean_slip > self._ENTRAP_SLIP_THRESH)
        self._entrap_counter = torch.where(is_stuck, self._entrap_counter + 1, torch.zeros_like(self._entrap_counter))
        stuck_flag = (self._entrap_counter >= self._ENTRAP_STEPS_THRESH).float()

        # Post-reset burial grace: tick counter down while rover is still near
        # spawn; zero it out once the rover has escaped the burial zone so the
        # natural is_stuck logic takes over for the rest of the episode.
        # Computed here so dist_norm below can reuse pos_xy — actually we need
        # the xy distance; defer the zero-out to after dist is computed.
        self.root_pos = wp.to_torch(self.robot.data.root_link_pos_w)
        dist_from_spawn = torch.norm(self.root_pos[:, :2] - self._spawn_pos, dim=-1)
        left_spawn = dist_from_spawn > self.cfg.burial_grace_dist
        self._burial_grace_counter = torch.where(
            left_spawn,
            torch.zeros_like(self._burial_grace_counter),
            (self._burial_grace_counter - 1.0).clamp(min=0.0),
        )
        grace_flag = (self._burial_grace_counter > 0.0).float()
        self._entrap_flag = torch.maximum(stuck_flag, grace_flag)

        # Slip-based anomaly detection: sustained high slip + low body velocity.
        # Absolute torque is always ~1.0 in training (motors always saturated vs sand),
        # so replaced with slip+velocity which is informative regardless of actuator tuning.
        #   Stuck:    mean_slip > 0.65 AND v_x < 0.20 m/s → fires correctly ✓
        #   Escaping: v_x > 0.20 m/s even though slip is still high → does NOT fire ✓
        self._prev_drive_torque_norm = drive_torque_norm.detach().clone()
        self._dbg_mean_torque_ratio = torch.mean(drive_torque_norm.abs(), dim=-1)
        self._dbg_torque_delta      = torch.mean(drive_torque_delta_pw, dim=-1)
        self._dbg_max_applied_eff   = drive_torque.abs().max()

        # Slip anomaly: same forward-projected v check as is_stuck, for consistency.
        # Using |v_x_body| would miss rocking and backward drift — the same bug as is_stuck.
        is_anomalous = (mean_slip > self._SLIP_ANOMALY_THRESH) & \
                       (v_forward < self._SLIP_ANOMALY_VX_THRESH)

        self._torque_anomaly_counter = torch.where(
            is_anomalous,
            self._torque_anomaly_counter + 1.0,
            (self._torque_anomaly_counter - 0.5).clamp(min=0.0),
        )
        self._torque_anomaly_flag = (self._torque_anomaly_counter >= self._SLIP_ANOMALY_STEPS_THRESH).float()

        # Progress along episode escape heading, normalised by escape threshold.
        # Uses per-episode escape direction so the signal is direction-agnostic.
        rel_pos_obs = self.root_pos[:, :2] - self._spawn_pos
        proj_dist_obs = (rel_pos_obs * self._escape_dir).sum(dim=-1).clamp(min=0.0)
        dist_norm = (proj_dist_obs / ESCAPE_DISTANCE).clamp(0.0, 2.0).unsqueeze(-1)

        # 6 + 6 + 4 + 3 + 1 + 6 + 1 + 1 + 1 = 29
        obs = torch.cat([
            drive_vel, slip, steer_pos, imu_acc, grav_z,
            drive_torque_delta_pw, self._entrap_flag.unsqueeze(-1),
            self._torque_anomaly_flag.unsqueeze(-1),
            dist_norm,
        ], dim=-1)
        obs = obs.nan_to_num(0.0).clamp(-5.0, 5.0)
        # Domain randomization: structured per-sensor additive observation noise.
        # Applied to policy obs only (first 29 dims) — critic sees clean privileged
        # signals, so noise is injected before the privileged block is concatenated.
        # Binary flags (entrap_flag at dim 26, slip_anomaly at dim 27) are NOT noised
        # — they are computed decisions, not raw sensor measurements.
        # Physical-unit sigmas are divided by the same normaliser as the obs
        # channel they perturb, so the datasheet values are applied verbatim.
        # (Pre-2026-06-13 bug: physical sigmas were added to NORMALISED obs —
        # IMU noise was ~10× datasheet, wheel-vel ~6×.)
        obs[:, 0:6]   += (self.cfg.dr_noise_wheel_vel / DRIVE_VEL_LIMIT) * torch.randn(self.num_envs, 6,  device=self.device)
        obs[:, 6:12]  += self.cfg.dr_noise_slip         * torch.randn(self.num_envs, 6,  device=self.device)
        obs[:, 12:16] += (self.cfg.dr_noise_steer_pos / STEER_POS_LIMIT) * torch.randn(self.num_envs, 4,  device=self.device)
        obs[:, 16:19] += (self.cfg.dr_noise_imu_acc / self._g_mag)       * torch.randn(self.num_envs, 3,  device=self.device)
        obs[:, 19:20] += self.cfg.dr_noise_grav_z       * torch.randn(self.num_envs, 1,  device=self.device)
        obs[:, 20:26] += self.cfg.dr_noise_drive_torque * torch.randn(self.num_envs, 6,  device=self.device)
        # obs[:, 26:27] — entrap_flag: binary, no noise
        # obs[:, 27:28] — slip_anomaly: binary, no noise
        obs[:, 28:29] += self.cfg.dr_noise_dist_norm    * torch.randn(self.num_envs, 1,  device=self.device)
        obs = obs.nan_to_num(0.0).clamp(-5.0, 5.0)

        if self.cfg.use_privileged_critic:
            # Oracle features for asymmetric critic. Not exposed to the policy at
            # train or deploy time — the critic slices [:, 29:] from the full obs.
            chassis_z    = self.root_pos[:, 2:3] - self.scene.env_origins[:, 2:3]
            wheel_center_z = chassis_z - CHASSIS_TO_WHEEL_Z
            sand_top_z   = getattr(self, "_sand_surface_z", SAND_DEPTH)
            wheel_burial = (sand_top_z - wheel_center_z).clamp(min=0.0)
            sand_force_proxy = drive_torque.abs().mean(dim=-1, keepdim=True)
            priv = torch.cat([
                self._sinkage.unsqueeze(-1),   # 1: episode-static true sinkage
                wheel_burial,                  # 1: live burial depth
                sand_force_proxy,              # 1: gross sand resistance
                self.root_vel_b,               # 3: full body linear velocity
                self.ang_vel_b[:, 2:3],        # 1: yaw rate
                chassis_z,                     # 1: true chassis height above env origin
            ], dim=-1).nan_to_num(0.0).clamp(-5.0, 5.0)   # (N, 8)
            obs = torch.cat([obs, priv], dim=-1)          # (N, 37)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.root_pos = wp.to_torch(self.robot.data.root_link_pos_w)
        v_x = wp.to_torch(self.robot.data.root_com_lin_vel_b)[:, 0]

        # NaN guard: clamp velocities to sane range
        v_x = v_x.nan_to_num(0.0).clamp(-10.0, 10.0)

        # World-frame velocity projected onto per-episode escape heading for r_progress.
        # Body-frame v_x is kept for slip, rocking, and anomaly terms (wheel-relative).
        v_world_full = wp.to_torch(self.robot.data.root_com_lin_vel_w).nan_to_num(0.0).clamp(-10.0, 10.0)
        v_world_xy = v_world_full[:, :2]
        v_z_world  = v_world_full[:, 2]
        v_x_world = (v_world_xy * self._escape_dir).sum(dim=-1)

        # All per-step reward terms use step_dt (policy step = sim.dt * decimation)
        # since _get_rewards fires once per policy step, not per physics step.
        dt = self.step_dt

        # Forward progress — ALWAYS on. Previously gated by (1 - entrap_flag) which created
        # a bistable reward landscape: zero progress signal during the entire burial-grace
        # window (3 s) and any sustained-slip episode, leaving rocking as the only positive
        # term. Policy collapsed to backward-thrash (mean_vx=-0.10, mean_dist=0.001).
        # Rocking stays as a small additive bonus (see p_rocking below).
        r_progress = self.cfg.rew_forward_progress * v_x_world * dt
        # Explicit reverse penalty so backward drift isn't a free local optimum.
        p_reverse  = self.cfg.pen_reverse * torch.clamp(-v_x_world, min=0.0) * dt

        # Escape bonus — distance projected onto per-episode escape heading from spawn.
        # Direction-agnostic: each episode has its own heading, covering all 360°.
        rel_pos = self.root_pos[:, :2] - self._spawn_pos
        dist    = (rel_pos * self._escape_dir).sum(dim=-1).clamp(min=0.0)
        time_scale = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length

        # Progressive milestone bonuses — restored variable-distance plot structure.
        # Thresholds rescaled to the new 3.0 m bed so the final tier coincides with
        # ESCAPE_DISTANCE and gives a real traverse signal.
        # ONE-TIME bonus per episode — fires only when the threshold is first crossed.
        milestones = [0.5, 1.0, 2.0, ESCAPE_DISTANCE]   # 4 bands: 0.5/1.0/2.0/3.0m
        milestone_weights = [0.1, 0.2, 0.4, 1.0]
        r_escape = torch.zeros(self.num_envs, device=self.device)
        for i, (thresh, w) in enumerate(zip(milestones, milestone_weights)):
            newly_reached = (dist > thresh) & (self._milestone_reached[:, i] == 0)
            r_escape += newly_reached.float() * self.cfg.rew_escape_bonus * w * time_scale
            self._milestone_reached[:, i] = torch.where(
                dist > thresh, torch.ones_like(self._milestone_reached[:, i]), self._milestone_reached[:, i]
            )
        # Dense progress shaping: reward only NEW +X progress (Δdist), not current
        # distance. Prevents reward-farming by lingering near the escape threshold:
        # an agent hovering at dist=1.4 m earns zero shaping, while an agent
        # progressing 0.1 m/step earns proportional credit. Only positive deltas
        # are credited (backward motion during rocking is handled by r_rocking).
        delta_dist = (dist - self._prev_dist).clamp(min=0.0)
        r_escape  += 5.0 * delta_dist
        self._prev_dist = dist.clone()

        # Slip penalty (drive wheels) - consistent with observation calculation
        # Suppressed when entrap_flag=1: rocking requires high slip, penalising it
        # actively prevents recovery.
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)
        drive_vel = self.joint_vel[:, self._drive_ids].nan_to_num(0.0) * WHEEL_RADIUS
        v_x_exp   = v_x.unsqueeze(1).expand(-1, 6)
        eps   = 0.01
        denom = torch.maximum(torch.abs(drive_vel),
                              torch.abs(v_x_exp).clamp(min=eps))
        slip      = ((drive_vel - v_x_exp) / denom).clamp(-1.0, 1.0)
        # Slip penalty — kept gated by entrap_flag so rocking-induced slip during recovery
        # isn't punished. Once entrap_flag clears, slip penalty resumes.
        p_slip    = self.cfg.pen_slip * torch.mean(torch.abs(slip), dim=-1) * (1.0 - self._entrap_flag) * dt

        # Tilt penalty
        ang_vel = wp.to_torch(self.robot.data.root_com_ang_vel_b).nan_to_num(0.0)
        p_tilt  = self.cfg.pen_tilt * torch.norm(ang_vel[:, :2], dim=-1) * dt

        # Vertical-hop penalty: directly punish |v_z| to discourage bouncing.
        # Locomotion-time bouncing was emergent from torque saturation (v6 audit:
        # raw_mean_torque_ratio = 0.9995 → every step max-thrust → impulsive lurches).
        # Gated by burial_grace: involuntary z-drift from gravity + sinkage spike
        # during the grace window must not be penalised — the policy has no control over it.
        grace_active = (self._burial_grace_counter > 0.0).float()
        p_hop = self.cfg.pen_hop * torch.abs(v_z_world) * (1.0 - grace_active) * dt

        # Action smoothness penalty
        p_smooth = self.cfg.pen_action_delta * torch.norm(
            self.actions - self.prev_action, dim=-1
        ) * dt
        self.prev_action = self.actions.clone()

        # Grind penalty (v9): penalize commanding high drive velocity while wheels
        # are slipping (i.e. torque is realized but produces no progress). Replaces
        # `pen_action_mag` which punished commands unconditionally — that term cannot
        # break torque saturation since saturation is set by v_error in sand, not by
        # the magnitude of the velocity command. This term fires only when the policy
        # is actually grinding (slip > thresh), giving the right pressure off the
        # saturator without taxing legitimate high-thrust escape attempts.
        mean_slip_abs = torch.mean(torch.abs(slip), dim=-1)
        high_slip     = (mean_slip_abs > self.cfg.slip_grind_thresh).float()
        drive_cmd_mag = torch.norm(self.actions[:, :6], dim=-1)
        p_grind       = self.cfg.pen_grind * high_slip * drive_cmd_mag * dt

        # Abnormal action penalty (sustained high torque with low progress).
        # Suppressed when entrap_flag=1: backward rocking is intentional recovery,
        # not an anomaly — firing this penalty during recovery teaches the wrong thing.
        # Per-env tensor (was incorrectly a scalar mean over all envs).
        progress_penalty = torch.clamp(-v_x, min=0.0)
        p_abnormal = (self.cfg.pen_abnormal
                      * progress_penalty
                      * self._torque_anomaly_flag
                      * (1.0 - self._entrap_flag)
                      * dt)

        # Rocking bonus when trapped — small additive bonus, NOT a primary signal.
        # World-frame projected on escape_dir so wheel-spin / yaw thrash doesn't farm reward;
        # only motion alternation along the escape heading counts as productive rocking.
        # Weight reduced 5.0 → 0.5 — Δ-distance shaping (5.0 × delta_dist in r_escape) is
        # the real escape signal; rocking is just a bootstrap nudge during deep burial.
        v_proj_change = torch.abs(v_x_world - self._prev_v_x_world)
        p_rocking  = self.cfg.rew_rocking * v_proj_change * self._entrap_flag * dt
        self.prev_v_x = v_x.clone()
        self._prev_v_x_world = v_x_world.clone()

        # Curriculum progress is updated in _reset_idx (once per episode reset),
        # NOT here in _get_rewards which runs every step. Updating here would
        # saturate the curriculum within seconds of training starting.

        self.extras.setdefault("log", {})
        self.extras["log"]["escape_rate"]        = self._escape_count / max(1, self._episode_count)
        # Projected distance along per-episode escape heading — 4 bands for paper plots
        self.extras["log"]["progress_0_5m"]      = (dist > 0.5).float().mean()
        self.extras["log"]["progress_1_0m"]      = (dist > 1.0).float().mean()
        self.extras["log"]["progress_2_0m"]      = (dist > 2.0).float().mean()
        self.extras["log"]["progress_3_0m"]      = (dist > 3.0).float().mean()   # == escape (rear wheels 0.25m past sand edge)
        self.extras["log"]["mean_vx"]            = v_x.mean()
        self.extras["log"]["mean_abs_slip"]      = torch.mean(torch.abs(slip), dim=-1).mean()
        self.extras["log"]["entrap_flag_rate"]   = self._entrap_flag.mean()
        self.extras["log"]["slip_anomaly_rate"] = self._torque_anomaly_flag.mean()
        # Torque-signal diagnostics: if raw_mean_torque_ratio stays ~0, applied_effort
        # isn't populated by the implicit actuator and we need a derived-torque fallback.
        if hasattr(self, "_dbg_mean_torque_ratio"):
            self.extras["log"]["raw_mean_torque_ratio"] = self._dbg_mean_torque_ratio.mean()
            self.extras["log"]["raw_torque_delta"]      = self._dbg_torque_delta.mean()
            self.extras["log"]["max_applied_effort"]    = self._dbg_max_applied_eff
        self.extras["log"]["mean_dist"]          = dist.mean()
        self.extras["log"]["curriculum_progress"] = self._curriculum_progress.mean()
        # Trivial-escape diagnostic: fraction of episodes that cleared 0.3 m within
        # the first 5 s (125 policy steps at 25 Hz). High values mean the prescribed
        # sinkage isn't actually trapping the rover → paper's central claim weakens.
        # NOTE: only episodes that DID eventually cross 0.3 m contribute; never-crossed
        # envs show first_progress_step = -1 and are excluded.
        crossed = self._first_progress_step > 0
        if crossed.any():
            trivial = crossed & (self._first_progress_step < float(self.cfg.trivial_escape_steps_thresh))
            self.extras["log"]["trivial_escape_frac"] = (
                trivial.float().sum() / crossed.float().sum()
            )
        # Competence-gate diagnostics
        if self._recent_escapes_full:
            self.extras["log"]["recent_escape_rate"] = self._recent_escapes.mean()

        # Per-component reward logging (signed; penalties stored negative to match
        # their contribution to the total). Enables a real "reward breakdown" plot
        # that shows which term is driving learning instead of just the aggregate.
        self.extras["log"]["rew_progress"] = r_progress.mean()
        self.extras["log"]["rew_escape"]   = r_escape.mean()
        self.extras["log"]["rew_rocking"]  = p_rocking.mean()
        self.extras["log"]["pen_slip"]     = (-p_slip).mean()
        self.extras["log"]["pen_tilt"]     = (-p_tilt).mean()
        self.extras["log"]["pen_smooth"]   = (-p_smooth).mean()
        self.extras["log"]["pen_abnormal"] = (-p_abnormal).mean()
        self.extras["log"]["pen_reverse"]  = (-p_reverse).mean()
        self.extras["log"]["pen_hop"]      = (-p_hop).mean()
        self.extras["log"]["pen_grind"]    = (-p_grind).mean()
        self.extras["log"]["grind_rate"]   = high_slip.mean()

        return r_progress + r_escape - p_slip - p_tilt - p_smooth - p_abnormal - p_reverse - p_hop - p_grind + p_rocking

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.root_pos  = wp.to_torch(self.robot.data.root_link_pos_w)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)
        grav_b         = wp.to_torch(self.robot.data.projected_gravity_b)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Omnidirectional escape: projected distance along per-episode heading from spawn.
        rel_pos_done = self.root_pos[:, :2] - self._spawn_pos
        axial_progress_done = (rel_pos_done * self._escape_dir).sum(dim=-1)
        escaped = axial_progress_done > ESCAPE_DISTANCE

        flipped = grav_b[:, 2] > -0.34   # >70° tilt
        # Sunk: wheel-bottom burial below the SETTLED sand surface exceeds the
        # threshold (v12). The old check (root z < origin − 0.20) required the
        # rover to fall through the entire 0.60 m bed before firing — dead code;
        # the 2026-06-13 probe showed a rover fully swallowed (−0.35 m) without
        # terminating. Threshold 0.45 m sits well past max prescribed sinkage
        # (0.28) so legitimate deep-burial episodes aren't clipped, but
        # catastrophic swallowing terminates instead of wasting the episode.
        if self.cfg.skip_mpm:
            # No sand → no burial. The rover legitimately drives on the ground
            # plane, ~0.6 m "below the surface" — the sunk check would fire
            # instantly and is meaningless here.
            sunk = torch.zeros_like(flipped)
        else:
            sand_top_done = getattr(self, "_sand_surface_z", SAND_DEPTH)
            wheel_bottom_z = (self.root_pos[:, 2] - self.scene.env_origins[:, 2]) \
                             - CHASSIS_TO_WHEEL_Z - WHEEL_RADIUS
            sunk = wheel_bottom_z < (sand_top_done - self.cfg.sunk_burial_thresh)

        # Lateral out-of-bounds: drift perpendicular to escape_dir > env_spacing/2 - margin.
        # Without this, a sideways-drifting rover never terminates and can collide with
        # neighboring envs (env_spacing=14 m → bound at 6 m lateral).
        axial_vec      = axial_progress_done.unsqueeze(-1) * self._escape_dir
        lateral_vec    = rel_pos_done - axial_vec
        lateral_dist   = torch.linalg.norm(lateral_vec, dim=-1)
        lateral_bound  = float(self.cfg.scene.env_spacing) * 0.5 - self.cfg.lateral_oob_margin
        lateral_oob    = lateral_dist > lateral_bound

        terminated = escaped | flipped | sunk | lateral_oob

        # Expose the per-env outcome flags computed HERE (before DirectRLEnv
        # auto-resets done envs within the same step()). External eval harnesses
        # must read these — reading root_pos/_spawn_pos after step() returns gives
        # the POST-reset pose for done envs (≈ spawn), so a distance-threshold
        # escape check done outside the env is always wrong. `escaped` already
        # isolates true escapes from flipped/sunk/lateral_oob terminations.
        self._last_escaped = escaped.clone()
        self._last_final_proj = axial_progress_done.clone()
        # True terminal position relative to spawn, world frame (pre-reset). Since
        # each episode faces a per-reset random escape heading, this encodes the
        # genuine directional outcome — used for the omnidirectional top-down plot.
        self._last_rel_xy = rel_pos_done.clone()

        # Tick per-env episode step counter and capture first-time-past-0.3m as a
        # trivial-escape diagnostic: if the rover clears a token progress threshold
        # within a few seconds, the prescribed sinkage didn't actually trap it.
        self._episode_step += 1.0
        axial_progress = axial_progress_done
        trivial_thresh = 0.3  # m
        newly_past = (axial_progress > trivial_thresh) & (self._first_progress_step < 0)
        self._first_progress_step = torch.where(
            newly_past, self._episode_step, self._first_progress_step
        )

        # Track escape rate before envs are reset
        done_mask = terminated | time_out
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            self._escape_count  += int(escaped[done_mask].sum().item())
            self._episode_count += n_done
            # Push outcomes into rolling window for competence-gated curriculum.
            done_env_ids = done_mask.nonzero(as_tuple=True)[0]
            outcomes     = escaped[done_env_ids].float()
            buf_len = self._recent_escapes.shape[0]
            for o in outcomes:
                self._recent_escapes[self._recent_escapes_idx] = o
                self._recent_escapes_idx = (self._recent_escapes_idx + 1) % buf_len
                if self._recent_escapes_idx == 0:
                    self._recent_escapes_full = True

            # Failure mode logging: append one row per terminated episode to CSV.
            if self.cfg.log_failure_modes and self._failure_csv_path is not None:
                curriculum_lvl = float(self._curriculum_progress.mean().item()) \
                    if hasattr(self, '_curriculum_progress') else float('nan')
                # Per-env friction is stored in MPM material array — approximate it
                # from the last DR sample if accessible, else NaN.
                rows = []
                for ei in done_env_ids.tolist():
                    esc_val   = int(escaped[ei].item())
                    sinkage   = float(self._sinkage[ei].item())
                    friction  = float(self._friction[ei].item())
                    rel_pos_ei = self.root_pos[ei, :2] - self._spawn_pos[ei]
                    final_dist = float((rel_pos_ei * self._escape_dir[ei]).sum().item())
                    ep_steps   = int(self._episode_step[ei].item())
                    rows.append([
                        self._episode_counter, esc_val, sinkage, friction,
                        curriculum_lvl, final_dist, ep_steps,
                    ])
                    self._episode_counter += 1
                try:
                    with open(self._failure_csv_path, "a", newline="") as f:
                        csv.writer(f).writerows(rows)
                except OSError:
                    pass  # never crash training due to CSV I/O failure

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Curriculum: one increment per env reset, normalised by total expected resets
        # ACROSS ALL ENVS (not per env). v9 fix: previously divided only by per-env
        # resets (~200 over a 200k-step run), but with 64 envs running in parallel
        # the agent sees 64× as many resets in the same wall-clock window. Curriculum
        # was saturating to 1.0 within ~5% of training, leaving 95% at max difficulty.
        # Correct denominator = total_timesteps × num_envs / steps_per_ep, since skrl
        # `timesteps` counts policy steps per env (not aggregate env-steps).
        if hasattr(self, '_curriculum_progress'):
            policy_hz        = 1.0 / (self.cfg.decimation * self.cfg.sim.dt)
            steps_per_ep     = int(self.cfg.episode_length_s * policy_hz)
            total_resets_est = max(1, (self._total_timesteps * self.num_envs) // steps_per_ep)
            # Competence-gated curriculum: advance only when the policy demonstrates
            # genuine recovery — not just trivial bouncing out.
            # Gate conditions (both must hold):
            #   1. recent_escape_rate >= 0.50 (was 0.40 — too easy when bounce escapes inflated it)
            #   2. trivial_escape_frac < 0.25 — less than 25% of escapes are trivial (<5s)
            # Without condition 2, the curriculum advanced on bounce escapes, pushing
            # sinkage deeper before the policy learned genuine recovery.
            curriculum_speed = 1.0 / 3.0
            # Fail-closed: gate stays at 0 until rolling window fills AND competence is proven.
            # Previous default of 1.0 advanced the curriculum unconditionally during the first
            # ~window_size episodes — observed pushing curriculum to 0.08 with escape_rate=0.
            competence_gate = 0.0
            backoff_gate    = 0.0
            if self._recent_escapes_full:
                recent_rate  = self._recent_escapes.mean().item()
                trivial_frac = float(self.extras.get("log", {}).get("trivial_escape_frac", 1.0))
                rate_ok    = recent_rate  >= 0.50
                trivial_ok = trivial_frac <  0.25
                if rate_ok and trivial_ok:
                    competence_gate = 1.0
                # Backoff: regression OR bounce-exploit collapse → decay curriculum.
                if (recent_rate  <  self.cfg.curriculum_backoff_rate_thresh
                    or trivial_frac >  self.cfg.curriculum_backoff_trivial_thresh):
                    backoff_gate = self.cfg.curriculum_backoff_scale
            step_norm = len(env_ids) / float(total_resets_est)
            prev_prog = self._curriculum_progress.clone()
            self._curriculum_progress = (
                self._curriculum_progress
                + (competence_gate - backoff_gate) * curriculum_speed * step_norm
            )
            # Backoff floor: once the curriculum has reached 0.2, backoff never
            # decays it below 0.2 (no reset to trivial difficulty / catastrophic
            # forgetting). The floor is min(prev, 0.2) so it can never LIFT
            # progress — the old clamp(min=0.2) jumped the very first episode
            # to 0.2 difficulty instead of starting at the shallow end.
            floor = torch.minimum(prev_prog, torch.full_like(prev_prog, 0.2))
            self._curriculum_progress = torch.clamp(
                torch.maximum(self._curriculum_progress, floor), min=0.0, max=1.0
            )

        # ── Robot pose ───────────────────────────────────────────────────────
        default_root_pose = wp.to_torch(
            self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        # Sample full 360° escape heading — direction-agnostic recovery primitive.
        # The rover always faces its escape direction; all 360° are covered across episodes.
        escape_angle = sample_uniform(0.0, 6.2832, (len(env_ids),), self.device)
        cos_a = torch.cos(escape_angle)
        sin_a = torch.sin(escape_angle)
        self._escape_dir[env_ids] = torch.stack([cos_a, sin_a], dim=-1)

        # Spawn 0.5 m behind center along -escape_dir so the rover has a full
        # ESCAPE_DISTANCE of runway ahead inside the sand bed in any direction.
        spawn_offset = 0.5  # m (matches old SPAWN_X_OFFSET magnitude)
        default_root_pose[:, 0] -= spawn_offset * cos_a
        default_root_pose[:, 1] -= spawn_offset * sin_a
        self._spawn_pos[env_ids] = default_root_pose[:, :2].clone()

        # Place rover so wheels are partially buried in sand.
        # Sand surface is at z = env_origin_z + SAND_DEPTH (0.60 m).
        # Wheel center (Drive joint) is CHASSIS_TO_WHEEL_Z (0.167 m) below chassis root.
        # Chassis z when wheel sits ON sand surface:
        #   z_chassis = env_z + SAND_DEPTH + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS
        # Sinkage lowers wheel center into sand:
        #   z_chassis = env_z + SAND_DEPTH + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS - sinkage
        # Curriculum: start with shallow sinkage, increase as training progresses.
        progress = min(1.0, self._curriculum_progress.mean().item()) if hasattr(self, '_curriculum_progress') else 0.0
        if self.cfg.sinkage_override is not None:
            sinkage_min = sinkage_max = float(self.cfg.sinkage_override)
        else:
            sinkage_min = self.cfg.dr_sinkage_range[0] + (self.cfg.dr_sinkage_range[1] - self.cfg.dr_sinkage_range[0]) * progress
            sinkage_max = self.cfg.dr_sinkage_range[0] + (self.cfg.dr_sinkage_range[1] - self.cfg.dr_sinkage_range[0]) * min(1.0, progress + 0.2)
        sinkage_depth = sample_uniform(
            sinkage_min, sinkage_max,
            (len(env_ids),), self.device,
        )
        self._sinkage[env_ids] = sinkage_depth
        # Surface = measured post-consolidation height (v12), not the as-built
        # SAND_DEPTH — the bed compacts during the init pre-settle.
        sand_top = getattr(self, "_sand_surface_z", SAND_DEPTH)
        default_root_pose[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + sand_top + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS - sinkage_depth
        )

        # Yaw = escape_angle (rover faces its per-episode escape direction exactly).
        # Small ±5° jitter prevents identical starting orientations across envs.
        yaw_jitter = sample_uniform(-0.0873, 0.0873, (len(env_ids),), self.device)
        yaw = escape_angle + yaw_jitter
        half_yaw = yaw * 0.5
        default_root_pose[:, 3] = 0.0
        default_root_pose[:, 4] = 0.0
        default_root_pose[:, 5] = torch.sin(half_yaw)
        default_root_pose[:, 6] = torch.cos(half_yaw)

        default_root_vel = wp.to_torch(
            self.robot.data.default_root_vel)[env_ids].clone() * 0.0
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone() * 0.0

        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(default_root_vel, env_ids)
        # Small random spin noise on drive joints only
        spin_noise = torch.zeros_like(joint_vel)
        spin_noise[:, self._drive_ids] = sample_uniform(
            -1.0, 1.0, (len(env_ids), len(self._drive_ids)), self.device
        )
        self.robot.write_joint_state_to_sim(joint_pos, spin_noise, None, env_ids)

        self.prev_action[env_ids] = 0.0
        self.prev_v_x[env_ids] = 0.0
        self._prev_v_x_world[env_ids] = 0.0
        # Rover spawns buried in sand → treat as already trapped from step 0.
        # This activates the rocking reward immediately without waiting 15 steps.
        self._entrap_counter[env_ids] = float(self._ENTRAP_STEPS_THRESH)
        self._entrap_flag[env_ids] = 1.0
        # Burial grace: force-hold flag=1 for burial_grace_steps policy steps
        # after reset (or until rover moves > burial_grace_dist from spawn).
        self._burial_grace_counter[env_ids] = float(self.cfg.burial_grace_steps)
        # Activate MPM post-reset settle window for these envs (clamps
        # per-body sand force for settle_steps physics steps — see
        # clamp_sand_forces_during_settle).
        if hasattr(self, "_settle_counter_wp"):
            settle_torch = wp.to_torch(self._settle_counter_wp)
            settle_torch[env_ids] = int(self.cfg.settle_steps)
        self._torque_anomaly_counter[env_ids] = 0.0
        self._torque_anomaly_flag[env_ids] = 0.0
        self._prev_drive_torque_norm[env_ids] = 0.0
        self._milestone_reached[env_ids] = 0.0
        # Reset per-episode tracking for Δdist shaping and trivial-escape diagnostics.
        self._prev_dist[env_ids]            = 0.0
        self._episode_step[env_ids]         = 0.0
        self._first_progress_step[env_ids]  = -1.0

        # Domain randomization: per-env motor gain
        self._motor_gain[env_ids] = sample_uniform(
            self.cfg.dr_motor_gain_range[0], self.cfg.dr_motor_gain_range[1],
            (len(env_ids), 1), self.device,
        )

        # ── Rock scatter ─────────────────────────────────────────────────────
        if self._rock_prim_paths:
            try:
                self._scatter_rocks(env_ids)
            except Exception:
                pass

        # ── Domain randomisation: sand friction (per-env) ───────────────────
        # PRE-FIX (silent bug): assigned a SCALAR to self.sand_model.particle_mu,
        # which (a) is global across all envs and (b) is a no-op after solver
        # construction (the solver copies particle_mu once and never re-reads it).
        # POST-FIX: write directly to mpm_solver.material_parameters.friction at
        # the slice corresponding to each env's particles. One sample per env per
        # reset → real per-episode friction DR with no cross-env contamination.
        if self._mpm_ready and hasattr(self.mpm_solver, "material_parameters"):
            if self.cfg.friction_override is not None:
                mu_lo = mu_hi = float(self.cfg.friction_override)
            else:
                mu_lo, mu_hi = self.cfg.dr_friction_range
            mu_per_env = sample_uniform(
                mu_lo, mu_hi, (len(env_ids),), self.device,
            ).cpu().numpy()
            friction_arr = self.mpm_solver.material_parameters.friction
            for k, ei in enumerate(env_ids):
                start = int(self._particle_env_starts[ei])
                count = int(self._particles_per_env)
                wp.launch(
                    set_env_friction,
                    dim=count,
                    inputs=[friction_arr, start, count, float(mu_per_env[k])],
                    device=self.device,
                )
                self._friction[ei] = float(mu_per_env[k])

        # ── Reset MPM sand patches ────────────────────────────────────────────
        if self._mpm_ready:
            for ei in env_ids:
                start = int(self._particle_env_starts[ei])
                count = int(self._particles_per_env)
                wp.launch(
                    reset_particle_range,
                    dim=count,
                    inputs=[
                        self._sand_q0,
                        self._sand_qd0,
                        self.sand_state.particle_q,
                        self.sand_state.particle_qd,
                        self.sand_state.particle_elastic_strain,
                        self.sand_state.particle_Jp,
                        self.sand_state.particle_qd_grad,
                        self.sand_state.particle_transform,
                        start,
                        count,
                    ],
                    device=self.device,
                )
            self._col_imp_ids.fill_(-1)

            # ── Spawn pocket-carving (audit fix A1, 2026-06-13) ─────────────
            # Teleport-spawn leaves sand particles overlapping the wheel/blade
            # volumes. At full coupling strength the per-step eviction of that
            # overlap squeezes the rover UPWARD out of its prescribed burial
            # (~120–180 N sustained — the "rover floats on the bed" symptom;
            # in the force-dilution era this was ÷8 and went unnoticed).
            # Resolve the overlap POSITIONALLY here, before any dynamics: sync
            # the just-teleported collider poses into the sand state and run
            # repeated project_outside passes. Particles get seated around the
            # wheels with zero velocity; the first dynamics step then sees a
            # carved pocket instead of interpenetration. project_outside is
            # global, which is harmless — it runs every step anyway.
            robot_state = NewtonManager._state_0
            if self.sand_state.body_q is not None and self._n_col_bodies > 0:
                n = min(self._n_col_bodies, robot_state.body_q.shape[0])
                wp.copy(self.sand_state.body_q, robot_state.body_q, count=n)
                if self.sand_state.body_qd is not None and robot_state.body_qd is not None:
                    wp.copy(self.sand_state.body_qd, robot_state.body_qd, count=n)
            _mpm_dt = NewtonManager._dt if hasattr(NewtonManager, "_dt") else 0.02
            for _ in range(int(os.environ.get("SPAWN_CARVE_ITERS", "8"))):
                self.mpm_solver.project_outside(self.sand_state, self.sand_state, dt=_mpm_dt)

            # Re-snap cosmetic render grains when any RENDERED env resets — its
            # particles just teleported back to the settled lattice. Pure
            # rendering; never touches the MPM state.
            if getattr(self, "_grains", None) is not None:
                try:
                    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
                    n_rendered = getattr(self, "_grain_render_envs", 1)
                    if any(i < n_rendered for i in ids):
                        ppe = int(self._particles_per_env)
                        ppp = int(self._grains_full.shape[1])
                        self._grains_full = self.mpm_solver.sample_render_grains(
                            self.sand_state, ppp)
                        self._grains = self._grains_full[:ppe * n_rendered]
                        wp.copy(self._grain_prev_q, self.sand_state.particle_q)
                except Exception:
                    pass

    def close(self):
        """Cleanup: unregister this env from MPM post-processing and viewer rendering."""
        if hasattr(NewtonManager, '_mpm_envs') and self in NewtonManager._mpm_envs:
            NewtonManager._mpm_envs.remove(self)
        if hasattr(NewtonManager, '_viewer_envs') and self in NewtonManager._viewer_envs:
            NewtonManager._viewer_envs.remove(self)
        super().close()


# ── Helpers ────────────────────────────────────────────────────────────────────

@wp.kernel
def _add_spatial_forces(
    src: wp.array(dtype=wp.spatial_vector),
    scale: float,
    dst: wp.array(dtype=wp.spatial_vector),
):
    """dst[i] += scale * src[i] — accumulate sand forces into Newton's force buffer.

    scale = num_substeps (v12 CRITICAL FIX): NewtonManager's substep loop calls
    `state_0.clear_forces()` after EVERY substep, so a force injected once per
    frame is consumed by substep 1 only — the delivered impulse was J/n. With
    n=8 substeps, every sand force in every run to date was diluted 8×: no
    wheel traction (thrust/8), phantom "fluidization" sinking (support/8 → the
    bed had to overshoot 8× through deep penetration to hold the rover at all).
    Scaling by n delivers the full collected impulse J in the first substep.
    `_body_sand_f` itself stays UNSCALED — subtract_body_force (the MPM
    anti-double-count) needs the frame-average force.
    """
    i = wp.tid()
    wp.atomic_add(dst, i, scale * src[i])


@wp.kernel
def _mask_grains_in_hull(
    grains:     wp.array(dtype=wp.vec3),
    base_radii: wp.array(dtype=wp.float32),
    radii:      wp.array(dtype=wp.float32),
    root_pos:   wp.vec3,
    root_quat:  wp.quat,   # (x, y, z, w)
):
    """RENDER-ONLY: zero the radius of grains inside the rover hull volume.

    Hull–soil contact is unmodelled (wheels-only sand colliders — see the
    SAND_HULL_COLLIDER note in _newton_init_cb), so at deep burial sand
    legitimately occupies the space where the hull is drawn. Hiding those
    grains fixes the "sand floating inside the rover" visual without touching
    physics. Bounds = USD Body bbox (0.665×0.490 m, z ∈ [0, 0.253]) + margin.
    """
    i = wp.tid()
    p = wp.quat_rotate_inv(root_quat, grains[i] - root_pos)
    if wp.abs(p[0]) < 0.35 and wp.abs(p[1]) < 0.26 and p[2] > -0.12 and p[2] < 0.27:
        radii[i] = 0.0
    else:
        radii[i] = base_radii[i]
