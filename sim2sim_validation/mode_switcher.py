import torch
from enum import IntEnum


class Mode(IntEnum):
    NAVIGATE  = 0
    ESCAPE    = 1


class ModeSwitcher:

    def __init__(
        self,
        num_envs:           int,
        device:             str,
        escape_distance:    float = 3.0,
        trigger_steps:      int   = 15,
        freed_steps:        int   = 25,
        freed_min_dist:     float = 1.9,


    ):
        self.num_envs        = num_envs
        self.device          = device
        self.escape_distance = escape_distance
        self.trigger_steps   = trigger_steps
        self.freed_steps     = freed_steps
        self.freed_min_dist  = freed_min_dist
        self.freed_counter   = torch.zeros(num_envs, device=device)

        self.mode            = torch.full((num_envs,), Mode.NAVIGATE,
                                          dtype=torch.long, device=device)
        self.trigger_counter = torch.zeros(num_envs, device=device)


        self.escape_spawn    = torch.zeros(num_envs, 2, device=device)
        self.escape_dir      = torch.zeros(num_envs, 2, device=device)


        self.steps_in_escape = torch.zeros(num_envs, device=device)
        self.escape_count    = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor):
        self.mode[env_ids]            = Mode.NAVIGATE
        self.trigger_counter[env_ids] = 0.0
        self.steps_in_escape[env_ids] = 0.0
        self.freed_counter[env_ids]   = 0.0

    def update(
        self,
        pos_xy:       torch.Tensor,
        entrap_flag:  torch.Tensor,
        goal_xy:      torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        newly_triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        newly_escaped   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_goal = goal_xy.clone()

        navigating = (self.mode == Mode.NAVIGATE)
        escaping   = (self.mode == Mode.ESCAPE)


        self.trigger_counter = torch.where(
            navigating & (entrap_flag > 0.5),
            self.trigger_counter + 1,
            torch.zeros_like(self.trigger_counter),
        )
        trigger_now = navigating & (self.trigger_counter >= self.trigger_steps)

        if trigger_now.any():
            env_ids = trigger_now.nonzero(as_tuple=True)[0]
            self.mode[env_ids]         = Mode.ESCAPE
            self.escape_spawn[env_ids] = pos_xy[env_ids].clone()


            rel = goal_xy[env_ids] - pos_xy[env_ids]
            dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)
            self.escape_dir[env_ids]   = rel / dist

            self.trigger_counter[env_ids] = 0.0
            self.steps_in_escape[env_ids] = 0.0
            self.escape_count[env_ids]   += 1
            newly_triggered[env_ids]      = True


        self.steps_in_escape = torch.where(
            escaping,
            self.steps_in_escape + 1,
            self.steps_in_escape,
        )


        rel_from_spawn = pos_xy - self.escape_spawn
        proj_dist = (rel_from_spawn * self.escape_dir).sum(dim=-1)


        self.freed_counter = torch.where(
            escaping & (entrap_flag < 0.5),
            self.freed_counter + 1,
            torch.zeros_like(self.freed_counter),
        )
        escape_done = escaping & (
            (proj_dist >= self.escape_distance)
            | ((self.freed_counter >= self.freed_steps)
               & (proj_dist >= self.freed_min_dist))
        )
        if escape_done.any():
            env_ids = escape_done.nonzero(as_tuple=True)[0]
            self.mode[env_ids] = Mode.NAVIGATE
            self.trigger_counter[env_ids] = 0.0
            self.freed_counter[env_ids]   = 0.0
            newly_escaped[env_ids] = True

        return self.mode.clone(), newly_triggered, newly_escaped, new_goal

    def get_escape_dir_for_env(self, env_id: int) -> torch.Tensor:
        return self.escape_dir[env_id]

    def summary(self) -> dict:
        return {
            "total_escapes":  int(self.escape_count.sum().item()),
            "mean_escape_steps": float(self.steps_in_escape.mean().item()),
        }
