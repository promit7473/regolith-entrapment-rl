"""
3-mode finite state machine for sim2sim validation.

Modes:
  NAVIGATE  — PD controller drives rover toward goal B
  ESCAPE    — trained PPO escape primitive takes over
  REPLANNED — escape done, resume navigation from new position

Transition logic:
  NAVIGATE  → ESCAPE    : entrap_flag fires for N consecutive steps
  ESCAPE    → REPLANNED : projected distance from spawn > ESCAPE_DISTANCE
  REPLANNED → NAVIGATE  : immediately (goal updated to B from new position)

This is the "dual-brain" architecture for the paper:
  navigation brain hands off to recovery primitive, then reclaims control.
"""

import math
import torch
from enum import IntEnum


class Mode(IntEnum):
    NAVIGATE  = 0
    ESCAPE    = 1
    REPLANNED = 2


class ModeSwitcher:
    """
    Per-env mode tracking and transition logic.

    The switcher watches the entrap_flag from the environment and triggers
    the escape primitive when the rover is genuinely stuck. After escape,
    it updates the goal and returns control to the navigation layer.
    """

    def __init__(
        self,
        num_envs:           int,
        device:             str,
        escape_distance:    float = 3.0,    # m — must match entrapment_env.ESCAPE_DISTANCE
        trigger_steps:      int   = 15,     # entrap_flag consecutive steps before switching
    ):
        self.num_envs        = num_envs
        self.device          = device
        self.escape_distance = escape_distance
        self.trigger_steps   = trigger_steps

        self.mode            = torch.full((num_envs,), Mode.NAVIGATE,
                                          dtype=torch.long, device=device)
        self.trigger_counter = torch.zeros(num_envs, device=device)

        # Recorded at the moment ESCAPE mode starts (used for projected dist)
        self.escape_spawn    = torch.zeros(num_envs, 2, device=device)
        self.escape_dir      = torch.zeros(num_envs, 2, device=device)

        # Per-env step counters for metrics
        self.steps_in_escape = torch.zeros(num_envs, device=device)
        self.escape_count    = torch.zeros(num_envs, device=device)
        self.failed_count    = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor):
        """Reset state for the given env indices."""
        self.mode[env_ids]            = Mode.NAVIGATE
        self.trigger_counter[env_ids] = 0.0
        self.steps_in_escape[env_ids] = 0.0

    def update(
        self,
        pos_xy:       torch.Tensor,   # (N, 2) current world-frame XY
        entrap_flag:  torch.Tensor,   # (N,)   from env obs
        goal_xy:      torch.Tensor,   # (N, 2) current navigation goal B
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update mode FSM.

        Returns:
          mode       (N,)    current mode per env
          newly_escaped (N,) bool — just transitioned ESCAPE→REPLANNED this step
          new_goal   (N, 2) updated goal (unchanged for NAVIGATE, = goal_xy for others)
        """
        newly_escaped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_goal = goal_xy.clone()

        navigating = (self.mode == Mode.NAVIGATE)
        escaping   = (self.mode == Mode.ESCAPE)

        # ── NAVIGATE → ESCAPE ─────────────────────────────────────────────────
        # Count consecutive steps with entrap_flag=1 while navigating
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

            # Escape heading = direction to goal B from current position
            rel = goal_xy[env_ids] - pos_xy[env_ids]
            dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)
            self.escape_dir[env_ids]   = rel / dist

            self.trigger_counter[env_ids] = 0.0
            self.steps_in_escape[env_ids] = 0.0
            self.escape_count[env_ids]   += 1

        # ── ESCAPE progress tracking ──────────────────────────────────────────
        self.steps_in_escape = torch.where(
            escaping,
            self.steps_in_escape + 1,
            self.steps_in_escape,
        )

        # Projected distance along escape heading from spawn
        rel_from_spawn = pos_xy - self.escape_spawn         # (N, 2)
        proj_dist = (rel_from_spawn * self.escape_dir).sum(dim=-1)  # (N,)

        # ── ESCAPE → REPLANNED ────────────────────────────────────────────────
        escape_done = escaping & (proj_dist >= self.escape_distance)
        if escape_done.any():
            env_ids = escape_done.nonzero(as_tuple=True)[0]
            self.mode[env_ids] = Mode.REPLANNED
            newly_escaped[env_ids] = True

        # ── REPLANNED → NAVIGATE ──────────────────────────────────────────────
        # Immediately: goal stays as B (unchanged), just switch mode back
        replanned = (self.mode == Mode.REPLANNED)
        if replanned.any():
            env_ids = replanned.nonzero(as_tuple=True)[0]
            self.mode[env_ids] = Mode.NAVIGATE
            self.trigger_counter[env_ids] = 0.0

        return self.mode.clone(), newly_escaped, new_goal

    def get_escape_dir_for_env(self, env_id: int) -> torch.Tensor:
        """Return escape direction for a single env (for overriding env._escape_dir)."""
        return self.escape_dir[env_id]

    def summary(self) -> dict:
        return {
            "total_escapes":  int(self.escape_count.sum().item()),
            "total_failures": int(self.failed_count.sum().item()),
            "mean_escape_steps": float(self.steps_in_escape.mean().item()),
        }
