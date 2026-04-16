"""
Mars Rover — Wheel Entrapment Recovery — Newton DirectRLEnv.

Physics stack:
  • MuJoCo Warp (MJWarpSolverCfg) — rigid-body dynamics for the 6-wheel Mars rover
  • SolverImplicitMPM      — sparse-grid MPM for granular regolith
  • Two-way coupling        — sand impulses → robot bodies, robot SDF → sand grid

Mars environment:
  • Static terrain mesh  : RLRoverLab terrain_merged.usd (photogrammetry Mars analog)
  • Rock scatter         : 10 rock USDs placed randomly each episode reset
  • Regolith bed         : 1.2 m × 1.2 m × 0.15 m XPBD particle bed per env

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
  drive_torque  (6)  — normalised drive joint torques (inspired by Bi & Ding 2026)
  entrap_flag   (1)  — binary entrapment indicator
  torque_anomaly(1)  — sustained high-torque anomaly flag
  dist_norm     (1)  — distance from env origin / escape threshold (0=spawn, 1=escaped)

Action (10D):
  drive_cmd     (6)  — velocity targets [-1,1] → ±6 rad/s
  steer_cmd     (4)  — position targets [-1,1] → ±0.6 rad
"""

from __future__ import annotations

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
from isaaclab.utils.math import sample_uniform

import newton as nt
from newton.solvers import SolverImplicitMPM

from robots import MARS_ROVER_CFG, ROVER_WHEEL_RADIUS
from .mpm_kernels import compute_body_forces, subtract_body_force, reset_particle_range, clamp_escaped_particles
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
ESCAPE_DISTANCE = 1.5                       # m — escape the entrapment zone

# Regolith pit geometry (per env, centred at env origin)
# 2.0 m × 2.0 m patch — wide enough that all 6 wheels stay fully buried during
# rocking maneuvers and the rover can't trivially escape sideways off the edge.
# Particle count per env ≈ 64 000 (80×80×10 at PPC=2, voxel=0.05 m, depth=0.25m).
SAND_HALF_X  = 1.0
SAND_HALF_Y  = 1.0
SAND_DEPTH   = 0.25           # raised from 0.15 — deeper bed so axle-level burial is possible
VOXEL_SIZE   = 0.05
PPC          = 2.0

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
    # + 6 drive torque + 1 entrapment flag + 1 torque anomaly flag
    # + 1 dist_norm (distance from origin / escape threshold) = 29
    observation_space = 29
    # 6 drive cmd + 4 steer cmd = 10
    action_space      = 10
    state_space       = 0

    episode_length_s = 30.0  # raised from 20s: rocking recovery takes several back-forth cycles
    decimation       = 2     # policy at 25 Hz
    skip_mpm         = False  # set True for viewer-only mode (no sand physics)

    # Reward weights
    # r_progress kept moderate so escape bonus stays dominant; agent must escape,
    # not just drive laps on the sand surface.
    rew_forward_progress = 1.5   # reduced: forward velocity alone shouldn't beat escape bonus
    rew_escape_bonus     = 20.0  # raised: escaping must be the highest-value action
    pen_slip             = 0.5   # halved: slip is unavoidable in sand, not a primary signal
    pen_tilt             = 0.3
    pen_action_delta     = 0.05
    pen_abnormal         = 0.3   # reduced: anomaly flag is still imperfect
    rew_rocking          = 2.0   # raised: rocking while trapped must clearly beat r_progress

    # Domain randomization ranges (inspired by Bi & Ding 2026, Table 9)
    dr_motor_gain_range  = (0.8, 1.2)   # multiplicative gain on drive velocity targets
    dr_obs_noise_std     = 0.02         # additive Gaussian noise on observations
    dr_sinkage_range     = (0.12, 0.22) # m — bury wheel axle, not just rim (was 0.06–0.12)

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
        num_substeps=4,
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
        env_spacing=8.0,     # sand 2 m wide + 1.5 m escape distance + buffer
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

        self.prev_action = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.prev_v_x = torch.zeros(self.num_envs, device=self.device)  # For rocking bonus calculation

        # One-time milestone tracking (3 milestones: 0.5m, 1.0m, 1.5m)
        # Each flag flips 0→1 the first time that distance is crossed in the episode.
        self._milestone_reached = torch.zeros(self.num_envs, 3, device=self.device)

        # Entrapment detection state (inspired by Bi & Ding 2026)
        # Counts consecutive steps where v_x < threshold AND mean_slip > threshold
        self._entrap_counter = torch.zeros(self.num_envs, device=self.device)
        self._entrap_flag    = torch.zeros(self.num_envs, device=self.device)
        self._ENTRAP_VX_THRESH   = 0.15  # m/s — rover making no real progress (raised from 0.05)
        self._ENTRAP_SLIP_THRESH = 0.4   # high slip ratio (lowered slightly for sensitivity)
        self._ENTRAP_STEPS_THRESH = 15   # ~0.6s at 25Hz — avoid triggering on brief slowdowns

        # Torque-based anomaly detection (inspired by Bi & Ding 2026)
        # Detect when torque CHANGE (not absolute) indicates struggling — avoids always-on issue.
        # Absolute threshold must be > normal driving torque (motors near limits in sand = anomaly).
        self._torque_anomaly_counter = torch.zeros(self.num_envs, device=self.device)
        self._torque_anomaly_flag    = torch.zeros(self.num_envs, device=self.device)
        self._prev_drive_torque_norm = torch.zeros(self.num_envs, 6, device=self.device)
        self._TORQUE_ANOMALY_THRESH       = 0.85  # mean |torque|/limit — sustained near-limit
        self._TORQUE_ANOMALY_DELTA_THRESH = 0.10  # mean torque fluctuation — struggling signal
        self._TORQUE_ANOMALY_STEPS_THRESH = 20    # ~0.8s sustained before flagging

        # Domain randomization: per-env motor gain (randomized at reset)
        self._motor_gain = torch.ones(self.num_envs, 1, device=self.device)
        
        # Curriculum learning progress tracker (across episodes)
        self._curriculum_progress = torch.zeros(1, device=self.device)  # Tracks overall training progress

        # Per-env rock pose storage (num_envs, ROCKS_PER_ENV, 3)
        self._rock_positions = torch.zeros(
            self.num_envs, ROCKS_PER_ENV, 3, device=self.device
        )

    # ── Scene setup ─────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Override Isaac Lab's default USD parsing to skip convex hull computation
        # for all 368 mesh shapes in Mars_Rover.usd. Without this, add_usd(stage)
        # takes 30+ min computing mesh approximations we don't need (we use proxy
        # spheres for collision instead). With skip_mesh_approximation=True it's <30s.
        @classmethod
        def _fast_instantiate(cls):
            from pxr import UsdGeom
            from isaaclab.sim._impl.newton_manager import get_current_stage
            stage = get_current_stage()
            up_axis = UsdGeom.GetStageUpAxis(stage)
            builder = nt.ModelBuilder(up_axis=up_axis)
            builder.add_usd(
                stage,
                skip_mesh_approximation=True,
                collapse_fixed_joints=False,   # keep differential/rocker link bodies separate
                load_visual_shapes=True,        # needed for correct visual transforms
            )
            NewtonManager.set_builder(builder)
        NewtonManager.instantiate_builder_from_stage = _fast_instantiate

        # Ground plane + proxy collision shapes via Newton builder callback.
        # The USD mesh collision shapes (329 of them) flood XPBD's contact buffer
        # → NaN. Fix: disable mesh collision, add 6 invisible proxy spheres on
        # wheel bodies only (same fix as view_rover.py).
        def _newton_init_cb():
            builder = NewtonManager._builder
            n_shapes = getattr(builder, 'shape_count', 0)
            n_bodies = getattr(builder, 'body_count', 0)
            print(f"[Newton Init] bodies={n_bodies}  mesh_shapes={n_shapes}")

            # Disable collision on all USD mesh shapes — keep VISIBLE for rendering.
            for s in range(n_shapes):
                builder.shape_flags[s] = builder.shape_flags[s] & ~nt.ShapeFlags.COLLIDE_SHAPES

            # Add proxy sphere ONLY on wheel bodies — sole collision geometry.
            # density=0.0 (massless) auto-disables has_particle_collision, so we
            # must explicitly re-enable it so MPM setup_collider registers these
            # spheres as SDF colliders (COLLIDE_PARTICLES flag).
            cfg = nt.ModelBuilder.ShapeConfig(
                ke=2e3, kd=1e2, kf=1e3, mu=0.75, density=0.0,
                has_shape_collision=True,
                has_particle_collision=True,   # ← critical: enables COLLIDE_PARTICLES
                is_visible=False,
            )
            body_keys = builder.body_key if hasattr(builder, 'body_key') else []
            n_wheels = 0
            for b in range(n_bodies):
                name = body_keys[b] if b < len(body_keys) else ''
                if 'Drive' in name:
                    builder.add_shape_sphere(body=b, radius=WHEEL_RADIUS, cfg=cfg)
                    n_wheels += 1
            print(f"[Newton Init] Disabled mesh collision on {n_shapes} shapes, "
                  f"added {n_wheels} wheel proxy spheres.")

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
        
        # NOTE: Mars gravity (3.72 m/s²) is set inside _init_mpm / _init_viewer_only
        # because NewtonManager._model is None here (model built by Newton's start callbacks).

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

        lo  = np.array([-SAND_HALF_X, -SAND_HALF_Y, 0.0])
        hi  = np.array([ SAND_HALF_X,  SAND_HALF_Y, SAND_DEPTH])
        res = np.array(np.ceil(PPC * (hi - lo) / VOXEL_SIZE), dtype=int)
        cell_size = (hi - lo) / res
        radius    = float(np.max(cell_size) * 0.5)
        mass      = float(np.prod(cell_size) * 1700.0)
        _p(f"[MPM] Grid res={res}, cell_size={cell_size}, radius={radius:.4f}, mass={mass:.4f}")

        sand_builder = nt.ModelBuilder(up_axis=nt.Axis.Z)
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
                jitter=2.0 * radius,
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
        self.sand_model.particle_mu = 0.9     # raised from 0.7 — higher inter-particle friction
        self.sand_model.particle_ke = 2.0e5   # raised from 5e4 — stiffer (200 kPa); still below CFL blow-up
        # NOTE: ke > ~5e5 Pa causes particle explosion: CFL > 1, VDB grid balloons → OOM.
        # 2e5 is the safe upper bound verified empirically at dt=0.005s.
        _p("[MPM] Sand model finalized.")

        mpm_opt = SolverImplicitMPM.Options()
        mpm_opt.voxel_size        = VOXEL_SIZE
        mpm_opt.tolerance         = 1.0e-5
        mpm_opt.grid_type         = "sparse"
        mpm_opt.transfer_scheme   = "apic"   # angular-momentum-conserving PIC — better wheel↔sand momentum transfer
        mpm_opt.strain_basis      = "P0"
        mpm_opt.max_iterations    = 30
        mpm_opt.critical_fraction = 0.025
        mpm_opt.hardening         = 5.0
        mpm_opt.air_drag          = 1.0

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
        _p(f"[MPM] collider_body_count={n_col}  — if 0, wheel↔sand coupling is DISABLED")
        if n_col > 0 and self.sand_state.body_q is not None:
            robot_state_init = NewtonManager._state_0
            wp.copy(self.sand_state.body_q,
                    robot_state_init.body_q,
                    count=min(n_col, robot_state_init.body_q.shape[0]))
            if self.sand_state.body_qd is not None:
                wp.copy(self.sand_state.body_qd,
                        robot_state_init.body_qd,
                        count=min(n_col, robot_state_init.body_qd.shape[0]))

        mpm_dt = NewtonManager._dt if hasattr(NewtonManager, "_dt") else 1.0 / 50.0
        _p("[MPM] Running project_outside...")
        self.mpm_solver.project_outside(self.sand_state, self.sand_state, dt=mpm_dt)
        _p("[MPM] Initial project_outside complete.")

        self._sand_q0  = wp.clone(self.sand_state.particle_q)
        self._sand_qd0 = wp.clone(self.sand_state.particle_qd)

        max_nodes = 1 << 20
        self._col_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._col_imp_pos  = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._col_imp_ids  = wp.full(max_nodes, value=-1, dtype=int, device=device)
        self._body_sand_f  = wp.zeros(robot_model.body_count,
                                      dtype=wp.spatial_vector, device=device)
        self._col_body_id  = mpm_model.collider.collider_body_index
        self._n_col_bodies = n_col

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
                original_step_fn(cls)
                # MPM post-processing
                for env_ref in NewtonManager._mpm_envs:
                    if env_ref._mpm_ready:
                        env_ref._mpm_post_xpbd()
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
        wp.launch(
            kernel=_add_spatial_forces,
            dim=robot_model.body_count,
            inputs=[self._body_sand_f, state_0.body_f],
            device=self.device,
        )

    def _mpm_post_xpbd(self):
        robot_model = NewtonManager._model
        state_0     = NewtonManager._state_0
        mpm_dt      = NewtonManager._dt

        if self.sand_state.body_q is not None and self._n_col_bodies > 0:
            wp.copy(self.sand_state.body_q,
                    state_0.body_q,
                    count=min(self._n_col_bodies, state_0.body_q.shape[0]))
        if self.sand_state.body_qd is not None and self._n_col_bodies > 0:
            wp.copy(self.sand_state.body_qd,
                    state_0.body_qd,
                    count=min(self._n_col_bodies, state_0.body_qd.shape[0]))

        self.mpm_solver.step(
            self.sand_state, self.sand_state,
            contacts=None, control=None, dt=mpm_dt,
        )

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
            ],
            device=self.device,
        )

        self._collect_mpm_impulses()
        self._render_sand_particles()

    def _init_viewer_only(self):
        """Lightweight viewer init — no MPM, just rigid-body visualization."""
        self._viewer = None
        self._render_frame = 0

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
        except Exception as e:
            print(f"[Sand Visual] ViewerGL setup failed: {e}")
            self._viewer = None

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


            sand_pts = self.sand_state.particle_q[::self._vis_stride]
            n_vis = len(sand_pts)
            # Reuse pre-allocated arrays to avoid VRAM fragmentation
            if not hasattr(self, '_vis_radii') or len(self._vis_radii) != n_vis:
                self._vis_radii  = wp.full(n_vis, VOXEL_SIZE * 0.6,
                                           dtype=wp.float32, device=sand_pts.device)
                self._vis_colors = wp.full(n_vis, wp.vec3(0.76, 0.60, 0.42),
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
        self._inject_sand_forces()

    def _apply_action(self) -> None:
        # Drive: velocity targets (with domain-randomized motor gain)
        drive_targets = self.actions[:, :6] * DRIVE_VEL_LIMIT * self._motor_gain
        self.robot.set_joint_velocity_target(drive_targets, joint_ids=self._drive_ids)
        # Steer: position targets
        steer_targets = self.actions[:, 6:] * STEER_POS_LIMIT
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

        # IMU acceleration normalised  (N, 3)
        imu_acc = self.lin_acc_w / 9.81

        # Tilt indicator  (N, 1)
        grav_z = self.grav_b[:, 2:3]

        # Drive motor torques normalised by effort limits  (N, 6)
        # (Bi & Ding 2026: torque saturation is the strongest entrapment signal)
        applied_torque = wp.to_torch(self.robot.data.applied_effort)
        drive_torque = applied_torque[:, self._drive_ids]
        effort_limits = wp.to_torch(self.robot.data.joint_effort_limits)
        drive_effort_lim = effort_limits[:, self._drive_ids].clamp(min=0.1)
        drive_torque_norm = (drive_torque / drive_effort_lim).clamp(-1.0, 1.0)

        # Entrapment detection flag  (N, 1)
        # Count consecutive steps with low v_x and high slip
        v_x_scalar = self.root_vel_b[:, 0].abs()
        mean_slip  = torch.mean(torch.abs(slip), dim=-1)
        is_stuck   = (v_x_scalar < self._ENTRAP_VX_THRESH) & (mean_slip > self._ENTRAP_SLIP_THRESH)
        self._entrap_counter = torch.where(is_stuck, self._entrap_counter + 1, torch.zeros_like(self._entrap_counter))
        self._entrap_flag    = (self._entrap_counter >= self._ENTRAP_STEPS_THRESH).float()

        # Torque-based anomaly detection (inspired by Bi & Ding 2026)
        # Anomaly = sustained near-limit torque WITH high torque fluctuation (struggling).
        # Using both magnitude AND delta avoids the always-on problem: in sand the rover
        # needs high torque, but true entrapment also shows rapid torque oscillation as
        # wheels alternately grip and slip.
        torque_ratio = (drive_torque / drive_effort_lim).clamp(0.0, 2.0)
        mean_torque_ratio = torch.mean(torque_ratio.abs(), dim=-1)
        torque_delta = torch.mean((drive_torque_norm - self._prev_drive_torque_norm).abs(), dim=-1)
        self._prev_drive_torque_norm = drive_torque_norm.detach().clone()
        is_anomalous = (mean_torque_ratio > self._TORQUE_ANOMALY_THRESH) & \
                       (torque_delta > self._TORQUE_ANOMALY_DELTA_THRESH)
        self._torque_anomaly_counter = torch.where(
            is_anomalous, self._torque_anomaly_counter + 1, torch.zeros_like(self._torque_anomaly_counter)
        )
        self._torque_anomaly_flag = (self._torque_anomaly_counter >= self._TORQUE_ANOMALY_STEPS_THRESH).float()

        # Distance from env origin, normalised by escape threshold (0=spawn, 1=escaped)
        # Gives the agent explicit progress feedback it cannot infer from velocity alone.
        self.root_pos   = wp.to_torch(self.robot.data.root_link_pos_w)
        pos_xy    = self.root_pos[:, :2] - self.scene.env_origins[:, :2]
        dist_norm = (torch.norm(pos_xy, dim=-1) / ESCAPE_DISTANCE).clamp(0.0, 2.0).unsqueeze(-1)

        # 6 + 6 + 4 + 3 + 1 + 6 + 1 + 1 + 1 = 29
        obs = torch.cat([
            drive_vel, slip, steer_pos, imu_acc, grav_z,
            drive_torque_norm, self._entrap_flag.unsqueeze(-1),
            self._torque_anomaly_flag.unsqueeze(-1),
            dist_norm,
        ], dim=-1)
        obs = obs.nan_to_num(0.0).clamp(-5.0, 5.0)
        # Domain randomization: additive observation noise
        if self.cfg.dr_obs_noise_std > 0.0:
            obs = obs + self.cfg.dr_obs_noise_std * torch.randn_like(obs)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.root_pos = wp.to_torch(self.robot.data.root_link_pos_w)
        v_x = wp.to_torch(self.robot.data.root_com_lin_vel_b)[:, 0]

        # NaN guard: clamp velocities to sane range
        v_x = v_x.nan_to_num(0.0).clamp(-10.0, 10.0)

        # Forward progress — suppressed when trapped so backward rocking isn't penalised.
        # When entrap_flag=1 the rocking bonus takes over as the locomotion signal.
        r_progress = self.cfg.rew_forward_progress * v_x * self.cfg.sim.dt * (1.0 - self._entrap_flag)

        # Escape bonus
        pos_xy     = self.root_pos[:, :2] - self.scene.env_origins[:, :2]
        dist       = torch.norm(pos_xy, dim=-1)
        time_scale = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length
        
        # Progressive milestone bonuses at 0.5m, 1.0m, 1.5m (escape).
        # ONE-TIME bonus per episode — fires only when the threshold is first crossed.
        milestones = [0.5, 1.0, ESCAPE_DISTANCE]
        milestone_weights = [0.2, 0.4, 1.0]
        r_escape = torch.zeros(self.num_envs, device=self.device)
        for i, (thresh, w) in enumerate(zip(milestones, milestone_weights)):
            newly_reached = (dist > thresh) & (self._milestone_reached[:, i] == 0)
            r_escape += newly_reached.float() * self.cfg.rew_escape_bonus * w * time_scale
            self._milestone_reached[:, i] = torch.where(
                dist > thresh, torch.ones_like(self._milestone_reached[:, i]), self._milestone_reached[:, i]
            )
        # Dense distance shaping: small per-step reward proportional to current distance
        # This gives a smooth gradient toward the escape zone each step.
        r_escape += 0.5 * dist * self.cfg.sim.dt

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
        p_slip    = self.cfg.pen_slip * torch.mean(torch.abs(slip), dim=-1) * (1.0 - self._entrap_flag) * self.cfg.sim.dt

        # Tilt penalty
        ang_vel = wp.to_torch(self.robot.data.root_com_ang_vel_b).nan_to_num(0.0)
        p_tilt  = self.cfg.pen_tilt * torch.norm(ang_vel[:, :2], dim=-1) * self.cfg.sim.dt

        # Action smoothness penalty
        p_smooth = self.cfg.pen_action_delta * torch.norm(
            self.actions - self.prev_action, dim=-1
        ) * self.step_dt
        self.prev_action = self.actions.clone()

        # Abnormal action penalty (sustained high torque with low progress).
        # Suppressed when entrap_flag=1: backward rocking is intentional recovery,
        # not an anomaly — firing this penalty during recovery teaches the wrong thing.
        # Per-env tensor (was incorrectly a scalar mean over all envs).
        progress_penalty = torch.clamp(-v_x, min=0.0)
        p_abnormal = (self.cfg.pen_abnormal
                      * progress_penalty
                      * self._torque_anomaly_flag
                      * (1.0 - self._entrap_flag)
                      * self.cfg.sim.dt)

        # Rocking bonus when trapped — per-env tensor, weight raised 0.1 → 1.0.
        # Rewards any alternation in forward/backward velocity while entrap_flag=1,
        # now strong enough to outweigh the (suppressed) slip penalty.
        v_x_change = torch.abs(v_x - self.prev_v_x)
        p_rocking  = self.cfg.rew_rocking * v_x_change * self._entrap_flag * self.cfg.sim.dt
        self.prev_v_x = v_x.clone()

        # Curriculum progress is updated in _reset_idx (once per episode reset),
        # NOT here in _get_rewards which runs every step. Updating here would
        # saturate the curriculum within seconds of training starting.

        self.extras.setdefault("log", {})
        self.extras["log"]["escape_rate"]        = (dist > ESCAPE_DISTANCE).float().mean()
        self.extras["log"]["milestone_0_5m"]     = (dist > 0.5).float().mean()
        self.extras["log"]["milestone_1_0m"]     = (dist > 1.0).float().mean()
        self.extras["log"]["mean_vx"]            = v_x.mean()
        self.extras["log"]["mean_abs_slip"]      = torch.mean(torch.abs(slip), dim=-1).mean()
        self.extras["log"]["entrap_flag_rate"]   = self._entrap_flag.mean()
        self.extras["log"]["torque_anomaly_rate"] = self._torque_anomaly_flag.mean()
        self.extras["log"]["mean_dist"]          = dist.mean()
        self.extras["log"]["curriculum_progress"] = self._curriculum_progress.mean()

        return r_progress + r_escape - p_slip - p_tilt - p_smooth - p_abnormal + p_rocking

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.root_pos  = wp.to_torch(self.robot.data.root_link_pos_w)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)
        grav_b         = wp.to_torch(self.robot.data.projected_gravity_b)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos_xy  = self.root_pos[:, :2] - self.scene.env_origins[:, :2]
        escaped = torch.norm(pos_xy, dim=-1) > ESCAPE_DISTANCE  # terminate on successful escape

        flipped = grav_b[:, 2] > -0.34   # >70° tilt
        sunk    = self.root_pos[:, 2] < -0.20  # chassis root below ground plane by >20cm

        terminated = escaped | flipped | sunk
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Curriculum: one increment per env reset, normalised by total expected resets.
        # 200k steps / 750 steps-per-episode (30s×25Hz) × 64 envs ≈ 17066 total resets.
        # We want progress to reach ~1.0 by end of training.
        if hasattr(self, '_curriculum_progress'):
            self._curriculum_progress += len(env_ids) / 17066.0
            self._curriculum_progress = torch.clamp(self._curriculum_progress, max=1.0)

        # ── Robot pose ───────────────────────────────────────────────────────
        default_root_pose = wp.to_torch(
            self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        # Place rover so wheels are partially buried in sand.
        # Sand surface is at z = env_origin_z + SAND_DEPTH (0.15 m).
        # Wheel center (Drive joint) is CHASSIS_TO_WHEEL_Z (0.167 m) below chassis root.
        # Chassis z when wheel sits ON sand surface:
        #   z_chassis = env_z + SAND_DEPTH + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS
        # Sinkage lowers wheel center into sand:
        #   z_chassis = env_z + SAND_DEPTH + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS - sinkage
        # Curriculum: start with shallow sinkage, increase as training progresses.
        progress = min(1.0, self._curriculum_progress.mean().item()) if hasattr(self, '_curriculum_progress') else 0.0
        sinkage_min = self.cfg.dr_sinkage_range[0] + (self.cfg.dr_sinkage_range[1] - self.cfg.dr_sinkage_range[0]) * progress
        sinkage_max = self.cfg.dr_sinkage_range[1]
        sinkage_depth = sample_uniform(
            sinkage_min, sinkage_max,
            (len(env_ids),), self.device,
        )
        default_root_pose[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + SAND_DEPTH + CHASSIS_TO_WHEEL_Z + WHEEL_RADIUS - sinkage_depth
        )

        # Random yaw ±30° around +X axis so the rover always faces roughly toward
        # the escape target.  Full ±180° caused ~50% of episodes to start facing away,
        # making escape structurally impossible and confusing the reward signal.
        yaw      = sample_uniform(-0.5236, 0.5236, (len(env_ids),), self.device)  # ±30°
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
        # Rover spawns buried in sand → treat as already trapped from step 0.
        # This activates the rocking reward immediately without waiting 15 steps.
        self._entrap_counter[env_ids] = float(self._ENTRAP_STEPS_THRESH)
        self._entrap_flag[env_ids] = 1.0
        self._torque_anomaly_counter[env_ids] = 0.0
        self._torque_anomaly_flag[env_ids] = 0.0
        self._prev_drive_torque_norm[env_ids] = 0.0
        self._milestone_reached[env_ids] = 0.0

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

        # ── Domain randomisation: sand friction ─────────────────────────────
        if self._mpm_ready:
            self.sand_model.particle_mu = float(
                sample_uniform(0.4, 1.0, (1,), self.device).item()
            )

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
    dst: wp.array(dtype=wp.spatial_vector),
):
    """dst[i] += src[i]  — accumulate sand forces into Newton's force buffer."""
    i = wp.tid()
    wp.atomic_add(dst, i, src[i])
