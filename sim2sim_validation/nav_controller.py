"""
Simple PD waypoint-following navigation controller.

Drives the rover toward a goal point in world frame using:
  - Proportional heading correction via steering joints
  - Constant forward drive velocity (reduced near goal)

This is intentionally simple — the paper contribution is the recovery
primitive, not the navigation. A deterministic PD controller isolates
the recovery contribution cleanly for reviewers.
"""

import math
import torch


class PDNavController:
    """
    Proportional-derivative waypoint follower for the AAU Mars rover.

    Inputs:  current world-frame pose (x, y, yaw), goal (x, y)
    Outputs: action tensor [drive_cmd(6), steer_cmd(4)] in [-1, 1]
    """

    def __init__(
        self,
        num_envs:        int,
        device:          str,
        drive_speed:     float = 0.6,   # fraction of max drive vel (1.0 = 6 rad/s)
        heading_gain:    float = 1.2,   # P-gain on heading error → steer angle
        arrival_radius:  float = 0.5,   # m — stop when within this distance of goal
    ):
        self.num_envs       = num_envs
        self.device         = device
        self.drive_speed    = drive_speed
        self.heading_gain   = heading_gain
        self.arrival_radius = arrival_radius

        # Goal position per env (world frame XY)
        self.goal = torch.zeros(num_envs, 2, device=device)

    def set_goal(self, goal_xy: torch.Tensor):
        """Set goal position for all envs. goal_xy: (num_envs, 2) world frame."""
        self.goal = goal_xy.to(self.device)

    def set_goal_single(self, goal_xy: torch.Tensor):
        """Broadcast a single (2,) goal to all envs."""
        self.goal = goal_xy.to(self.device).unsqueeze(0).expand(self.num_envs, -1).clone()

    def step(
        self,
        pos_xy:  torch.Tensor,   # (num_envs, 2) world-frame XY
        yaw:     torch.Tensor,   # (num_envs,)   world-frame yaw in radians
    ) -> torch.Tensor:
        """
        Compute action tensor from current pose toward goal.
        Returns: (num_envs, 10) action in [-1, 1].
        """
        rel = self.goal - pos_xy                            # (N, 2)
        dist = torch.norm(rel, dim=-1, keepdim=True)        # (N, 1)

        # Heading to goal in world frame
        goal_heading = torch.atan2(rel[:, 1], rel[:, 0])   # (N,)

        # Heading error (wrapped to [-π, π])
        heading_err = goal_heading - yaw
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi  # wrap

        # Steer correction — proportional, clamped to [-1, 1]
        steer_cmd = (self.heading_gain * heading_err / math.pi).clamp(-1.0, 1.0)  # (N,)

        # Drive speed — slow down near goal, stop at arrival
        arrived   = (dist.squeeze(-1) < self.arrival_radius)
        speed     = torch.where(arrived, torch.zeros_like(dist.squeeze(-1)),
                                torch.full_like(dist.squeeze(-1), self.drive_speed))

        # Build action tensor: 6 drive wheels all same speed, 4 steers all same angle
        action = torch.zeros(self.num_envs, 10, device=self.device)
        action[:, :6] = speed.unsqueeze(-1).expand(-1, 6)           # drive
        action[:, 6:] = steer_cmd.unsqueeze(-1).expand(-1, 4)       # steer

        return action, arrived

    def extract_yaw(self, root_pose: torch.Tensor) -> torch.Tensor:
        """
        Extract yaw from Newton root pose tensor [x,y,z, qx,qy,qz,qw].
        Returns: (num_envs,) yaw in radians.
        """
        qx = root_pose[:, 3]
        qy = root_pose[:, 4]
        qz = root_pose[:, 5]
        qw = root_pose[:, 6]
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return yaw
