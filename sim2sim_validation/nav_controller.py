import math
import torch


class PDNavController:

    def __init__(
        self,
        num_envs:        int,
        device:          str,
        drive_speed:     float = 0.6,
        heading_gain:    float = 1.2,
        arrival_radius:  float = 0.5,
        heading_deadband: float = 0.035,
    ):
        self.num_envs        = num_envs
        self.device          = device
        self.drive_speed     = drive_speed
        self.heading_gain    = heading_gain
        self.arrival_radius  = arrival_radius
        self.heading_deadband = heading_deadband


        self.goal = torch.zeros(num_envs, 2, device=device)

    def set_goal(self, goal_xy: torch.Tensor):
        self.goal = goal_xy.to(self.device)

    def set_goal_single(self, goal_xy: torch.Tensor):
        self.goal = goal_xy.to(self.device).unsqueeze(0).expand(self.num_envs, -1).clone()

    def step(
        self,
        pos_xy:  torch.Tensor,
        yaw:     torch.Tensor,
    ) -> torch.Tensor:
        rel = self.goal - pos_xy
        dist = torch.norm(rel, dim=-1, keepdim=True)


        goal_heading = torch.atan2(rel[:, 1], rel[:, 0])


        heading_err = goal_heading - yaw
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi


        steer_raw = self.heading_gain * heading_err / math.pi
        in_deadband = heading_err.abs() < self.heading_deadband
        steer_cmd = torch.where(in_deadband, torch.zeros_like(steer_raw), steer_raw).clamp(-1.0, 1.0)


        arrived   = (dist.squeeze(-1) < self.arrival_radius)
        speed     = torch.where(arrived, torch.zeros_like(dist.squeeze(-1)),
                                torch.full_like(dist.squeeze(-1), self.drive_speed))


        action = torch.zeros(self.num_envs, 10, device=self.device)
        action[:, :6] = speed.unsqueeze(-1).expand(-1, 6)
        action[:, 6:] = steer_cmd.unsqueeze(-1).expand(-1, 4)

        return action, arrived

    def extract_yaw(self, root_pose: torch.Tensor) -> torch.Tensor:
        qx = root_pose[:, 3]
        qy = root_pose[:, 4]
        qz = root_pose[:, 5]
        qw = root_pose[:, 6]

        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return yaw
