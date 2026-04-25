"""
Sim2Sim Validation Env — spawn buried in sand, navigate to a goal sign.

Geometry (env-local frame, origin at sand-bed center):
  Sand patch:  3.5 m × 3.5 m × 0.30 m, centered at (0, 0).
  Ground:      MuJoCo ground plane at z=0 (inherited from training env).
  A (spawn):   (0, 0)  — buried in sand at the bed centre (stuck by default).
  B (goal):    (+3.0, 0)  — flat ground east of sand, marked by a sign post.

Why no elevated platform: MuJoCo (rigid-body solver for the rover) does NOT
read static shapes added via Newton's builder. An "elevated" platform would
be visual-only — the rover would fall through it. Flat ground matches the
exact physics the policy was trained on.

A/B markers are added as Newton shapes (sphere + capsule pole) so they show
up in ViewerGL. They use as_site=True so they're non-colliding (sand and
rover pass through cleanly).

The training env (`envs/entrapment_env.py`) is NOT touched.
"""

import math
import torch
import warp as wp
from typing import Sequence

from isaaclab.utils import configclass
from isaaclab.sim._impl.newton_manager import NewtonManager

from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg, SAND_DEPTH


# Mars-red palette (RGB 0..1) for ViewerGL recoloring.
MARS_SAND_COLOR    = wp.vec3(0.62, 0.30, 0.20)
MARKER_GREEN_COLOR = (0.10, 0.85, 0.10)
MARKER_RED_COLOR   = (0.90, 0.12, 0.12)
MARKER_POLE_COLOR  = (0.95, 0.95, 0.95)


@configclass
class ValidationEnvCfg(EntrapmentEnvCfg):
    """A→sand→B validation scenario."""

    val_spawn_x:        float = 0.0     # centre of sand bed (buried)
    val_spawn_y:        float = 0.0
    val_goal_x:         float = 3.0
    val_goal_y:         float = 0.0
    val_arrival_radius: float = 0.5
    val_lateral_oob:    float = 4.5
    val_max_x:          float = 6.0


class ValidationEnv(EntrapmentEnv):
    """A→sand→B env on flat ground with Newton-side visual markers."""

    cfg: ValidationEnvCfg

    # ── Scene setup ──────────────────────────────────────────────────────────
    def _setup_scene(self):
        super()._setup_scene()
        # Markers are visual-only via viewer.log_points + viewer.log_lines —
        # no physics shapes, no Newton init hook needed. They're emitted each
        # frame from `_render_sand_particles` once the viewer is up.
        print(f"[Validation] Marker positions: "
              f"A=({self.cfg.val_spawn_x},{self.cfg.val_spawn_y}) "
              f"B=({self.cfg.val_goal_x},{self.cfg.val_goal_y})")

    # ── Visual-only nav-pin markers (log_points head + log_lines spike) ──────
    def _build_marker_arrays(self):
        """Lazy: build wp arrays for head spheres + spike line segments
        per env. Cached on self so they're only built once."""
        if getattr(self, "_marker_arrays_built", False):
            return

        head_radius_A = 0.22
        head_radius_B = 0.22
        spike_h       = 0.65   # spike length from base to head centre
        head_drop     = 0.06   # how far head dips below the spike's top tip

        env_origins_np = self.scene.env_origins.cpu().numpy()
        n_envs = len(env_origins_np)

        head_pts   = []   # (2*N, 3)  — A and B head centres per env
        head_radii = []   # (2*N,)
        head_cols  = []   # (2*N, 3)
        spike_starts = []  # (2*N, 3) — base of spike (on surface)
        spike_ends   = []  # (2*N, 3) — top tip of spike (touches head)

        for env_origin in env_origins_np:
            ox, oy, oz = (float(env_origin[0]), float(env_origin[1]),
                          float(env_origin[2]))
            for tag, (lx, ly), col, hr in [
                ("A", (self.cfg.val_spawn_x, self.cfg.val_spawn_y),
                 MARKER_GREEN_COLOR, head_radius_A),
                ("B", (self.cfg.val_goal_x,  self.cfg.val_goal_y),
                 MARKER_RED_COLOR,   head_radius_B),
            ]:
                wx, wy = ox + lx, oy + ly
                base_z = oz + (SAND_DEPTH if tag == "A" else 0.0)
                tip_z  = base_z + spike_h
                head_z = tip_z - head_drop

                spike_starts.append(wp.vec3(wx, wy, base_z))
                spike_ends.append(  wp.vec3(wx, wy, tip_z))
                head_pts.append(    wp.vec3(wx, wy, head_z))
                head_radii.append(float(hr))
                head_cols.append(wp.vec3(*col))

        device = "cpu"
        self._marker_head_pts   = wp.array(head_pts,   dtype=wp.vec3, device=device)
        self._marker_head_radii = wp.array(head_radii, dtype=float,   device=device)
        self._marker_head_cols  = wp.array(head_cols,  dtype=wp.vec3, device=device)
        self._marker_spike_starts = wp.array(spike_starts, dtype=wp.vec3, device=device)
        self._marker_spike_ends   = wp.array(spike_ends,   dtype=wp.vec3, device=device)
        spike_col = wp.vec3(*MARKER_POLE_COLOR)
        self._marker_spike_cols   = wp.array(
            [spike_col] * len(spike_starts), dtype=wp.vec3, device=device,
        )
        self._marker_arrays_built = True
        print(f"[Validation] Built nav-pin arrays for {n_envs} env(s) "
              f"(A=green, B=red, log_points + log_lines).")

    # ── Mars sand recolor + per-frame nav-pin overlay ─────────────────────────
    def _render_sand_particles(self):
        super()._render_sand_particles()

        # Sand particles: parent lazily allocates _vis_colors with tan; overwrite.
        if hasattr(self, "_vis_colors") and not getattr(self, "_mars_recolored", False):
            try:
                self._vis_colors.fill_(MARS_SAND_COLOR)
                self._mars_recolored = True
                print(f"[Validation] Sand recolored Mars red {MARS_SAND_COLOR}")
            except Exception as e:
                print(f"[Validation] Mars recolor failed: {e}")

        # Nav pins: emit each frame via the viewer's visual-only logging API.
        # Heads = log_points (radius+color per pin), spikes = log_lines (white).
        viewer = getattr(self, "_viewer", None)
        if viewer is not None:
            try:
                self._build_marker_arrays()
                viewer.log_points(
                    "/nav_pins",
                    points=self._marker_head_pts,
                    radii=self._marker_head_radii,
                    colors=self._marker_head_cols,
                )
                viewer.log_lines(
                    "/nav_pin_spikes",
                    starts=self._marker_spike_starts,
                    ends=self._marker_spike_ends,
                    colors=self._marker_spike_cols,
                    width=0.04,
                )
            except Exception as e:
                if not getattr(self, "_pin_log_warned", False):
                    print(f"[Validation] Pin logging failed: {e}")
                    self._pin_log_warned = True

    # ── Reset: rover spawns buried in sand at (val_spawn_x, val_spawn_y),
    # facing the goal sign B. We let the parent reset do the heavy lifting
    # (sinkage curriculum, settle window, entrap_flag=1, spawn pose with
    # correct burial z) and only override:
    #   * planar (x,y) — parent samples around the env origin; we pin to A.
    #   * yaw — parent uses random escape_angle; we point at B.
    #   * joint spin noise — parent adds ±1 rad/s spin; we zero it for a
    #     clean start.
    #   * _spawn_pos / _escape_dir — used by validation metrics.
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        device = self.device

        env_origins = self.scene.env_origins[env_ids]   # (n, 3)

        dx = self.cfg.val_goal_x - self.cfg.val_spawn_x
        dy = self.cfg.val_goal_y - self.cfg.val_spawn_y
        yaw = math.atan2(dy, dx)
        half_yaw = yaw * 0.5

        # Read the pose parent just wrote (keeps parent's z = sand surface
        # minus sinkage), and overwrite x, y, and quaternion.
        cur_pose = wp.to_torch(self.robot.data.root_link_pose_w)[env_ids].clone()
        cur_pose[:, 0] = env_origins[:, 0] + self.cfg.val_spawn_x
        cur_pose[:, 1] = env_origins[:, 1] + self.cfg.val_spawn_y
        cur_pose[:, 3] = 0.0
        cur_pose[:, 4] = 0.0
        cur_pose[:, 5] = math.sin(half_yaw)
        cur_pose[:, 6] = math.cos(half_yaw)

        zero_root_vel = torch.zeros(n, 6, device=device)
        self.robot.write_root_pose_to_sim(cur_pose, env_ids)
        self.robot.write_root_velocity_to_sim(zero_root_vel, env_ids)

        # Zero parent's random drive-spin noise so the rover starts truly still.
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Validation-specific state for metrics.
        self._spawn_pos[env_ids] = cur_pose[:, :2].clone()

        goal_local  = torch.tensor([self.cfg.val_goal_x,  self.cfg.val_goal_y],
                                   device=device, dtype=torch.float32)
        spawn_local = torch.tensor([self.cfg.val_spawn_x, self.cfg.val_spawn_y],
                                   device=device, dtype=torch.float32)
        rover_to_goal = goal_local - spawn_local
        rover_to_goal = rover_to_goal / (torch.norm(rover_to_goal) + 1e-6)
        self._escape_dir[env_ids] = rover_to_goal.unsqueeze(0).expand(n, -1).clone()

        # NOTE: do NOT zero _entrap_flag / _sinkage / _settle_counter — parent
        # already set them for a buried-in-sand start, which is exactly what
        # the validation scenario wants ("stuck by default, then go to B").

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.root_pos  = wp.to_torch(self.robot.data.root_link_pos_w)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_world = self.scene.env_origins[:, :2] + torch.tensor(
            [self.cfg.val_goal_x, self.cfg.val_goal_y],
            device=self.device, dtype=torch.float32,
        )
        dist_to_goal = torch.norm(self.root_pos[:, :2] - goal_world, dim=-1)
        reached_goal = dist_to_goal < self.cfg.val_arrival_radius

        rel_xy = self.root_pos[:, :2] - self._spawn_pos
        oob = (rel_xy[:, 1].abs() > self.cfg.val_lateral_oob) | \
              (rel_xy[:, 0].abs() > self.cfg.val_max_x)

        # NOTE: training-env-style `flipped` (tilt>70°) and `sunk` (z<-0.20)
        # checks were dropped from validation. The PPO escape primitive
        # routinely tilts the chassis past 70° as it lurches free of the
        # sand — those terminations were killing trials at the exact moment
        # of escape. For nav validation we only care about reaching B,
        # bouncing off the env boundary, or timing out.
        terminated = reached_goal | oob

        self._episode_step += 1.0
        return terminated, time_out
