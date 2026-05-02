import math
import torch
import warp as wp
from typing import Sequence

from isaaclab.utils import configclass
from isaaclab.sim._impl.newton_manager import NewtonManager

from envs.entrapment_env import EntrapmentEnv, EntrapmentEnvCfg, SAND_DEPTH


MARS_SAND_COLOR    = wp.vec3(0.62, 0.30, 0.20)
MARKER_GREEN_COLOR = (0.10, 0.85, 0.10)
MARKER_RED_COLOR   = (0.90, 0.12, 0.12)
MARKER_POLE_COLOR  = (0.95, 0.95, 0.95)


@configclass
class ValidationEnvCfg(EntrapmentEnvCfg):

    val_spawn_x:        float = 0.0
    val_spawn_y:        float = 0.0
    val_goal_x:         float = 3.0
    val_goal_y:         float = 0.0
    val_arrival_radius: float = 0.5
    val_lateral_oob:    float = 4.5
    val_max_x:          float = 6.0


class ValidationEnv(EntrapmentEnv):

    cfg: ValidationEnvCfg


    def _setup_scene(self):
        super()._setup_scene()


        print(f"[Validation] Marker positions: "
              f"A=({self.cfg.val_spawn_x},{self.cfg.val_spawn_y}) "
              f"B=({self.cfg.val_goal_x},{self.cfg.val_goal_y})")


    def _build_marker_arrays(self):
        if getattr(self, "_marker_arrays_built", False):
            return

        head_radius_A = 0.22
        head_radius_B = 0.22
        spike_h       = 0.65
        head_drop     = 0.06

        env_origins_np = self.scene.env_origins.cpu().numpy()
        n_envs = len(env_origins_np)

        head_pts   = []
        head_radii = []
        head_cols  = []
        spike_starts = []
        spike_ends   = []

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


    def _render_sand_particles(self):
        super()._render_sand_particles()


        if hasattr(self, "_vis_colors") and not getattr(self, "_mars_recolored", False):
            try:
                self._vis_colors.fill_(MARS_SAND_COLOR)
                self._mars_recolored = True
                print(f"[Validation] Sand recolored Mars red {MARS_SAND_COLOR}")
            except Exception as e:
                print(f"[Validation] Mars recolor failed: {e}")


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


    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        device = self.device

        env_origins = self.scene.env_origins[env_ids]

        dx = self.cfg.val_goal_x - self.cfg.val_spawn_x
        dy = self.cfg.val_goal_y - self.cfg.val_spawn_y
        yaw = math.atan2(dy, dx)
        half_yaw = yaw * 0.5


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


        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        self._spawn_pos[env_ids] = cur_pose[:, :2].clone()

        goal_local  = torch.tensor([self.cfg.val_goal_x,  self.cfg.val_goal_y],
                                   device=device, dtype=torch.float32)
        spawn_local = torch.tensor([self.cfg.val_spawn_x, self.cfg.val_spawn_y],
                                   device=device, dtype=torch.float32)
        rover_to_goal = goal_local - spawn_local
        rover_to_goal = rover_to_goal / (torch.norm(rover_to_goal) + 1e-6)
        self._escape_dir[env_ids] = rover_to_goal.unsqueeze(0).expand(n, -1).clone()


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


        terminated = reached_goal | oob

        self._episode_step += 1.0
        return terminated, time_out
