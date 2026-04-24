"""
Trial metrics for sim2sim A→B validation.

Tracked per trial:
  - reached_goal      : bool   — rover arrived within arrival_radius of B
  - escaped           : bool   — escape primitive fired AND succeeded
  - time_to_escape    : int    — policy steps from entrap_flag trigger → escaped
  - time_to_goal      : int    — total steps from episode start → reached B
  - path_length       : float  — total distance travelled (m)
  - straight_line_dist: float  — straight-line A→B distance (m)
  - path_efficiency   : float  — straight_line / path_length (1.0 = perfect)
  - false_trigger_rate: float  — entrap_flag fired but rover wasn't actually stuck

Paper table: report mean ± std over N trials for each metric.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrialResult:
    trial_id:          int
    reached_goal:      bool
    escaped:           bool
    time_to_escape:    int     # steps; -1 if never escaped
    time_to_goal:      int     # steps; -1 if never reached
    path_length:       float
    straight_line_dist: float
    path_efficiency:   float   # straight_line / path_length
    escape_heading_error: float  # degrees — angle between escape dir and true dir-to-B


class MetricsTracker:
    """Accumulates per-step data and computes trial-level metrics."""

    def __init__(self, num_envs: int, device: str, goal_xy: torch.Tensor):
        self.num_envs  = num_envs
        self.device    = device
        self.goal_xy   = goal_xy.to(device)   # (N, 2) or (2,) broadcast

        self.results: List[TrialResult] = []

        # Per-env step accumulation
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._step          = torch.zeros(self.num_envs, device=self.device)
        self._path_length   = torch.zeros(self.num_envs, device=self.device)
        self._prev_pos      = torch.zeros(self.num_envs, 2, device=self.device)
        self._escape_step   = torch.full((self.num_envs,), -1.0, device=self.device)
        self._goal_step     = torch.full((self.num_envs,), -1.0, device=self.device)
        self._start_pos     = torch.zeros(self.num_envs, 2, device=self.device)
        self._escaped_flag  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reached_flag  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._escape_dir    = torch.zeros(self.num_envs, 2, device=self.device)
        self._trial_id      = 0

    def begin_trial(self, env_ids: torch.Tensor, start_pos: torch.Tensor):
        """Call at the start of each trial (episode reset)."""
        self._step[env_ids]         = 0
        self._path_length[env_ids]  = 0.0
        self._prev_pos[env_ids]     = start_pos[env_ids].clone()
        self._start_pos[env_ids]    = start_pos[env_ids].clone()
        self._escape_step[env_ids]  = -1.0
        self._goal_step[env_ids]    = -1.0
        self._escaped_flag[env_ids] = False
        self._reached_flag[env_ids] = False

    def step(
        self,
        pos_xy:       torch.Tensor,   # (N, 2)
        newly_escaped: torch.Tensor,  # (N,) bool
        arrived:      torch.Tensor,   # (N,) bool — nav controller reached goal
        escape_dir:   torch.Tensor,   # (N, 2) escape direction used
    ):
        """Update accumulators each policy step."""
        self._step += 1

        # Accumulate path length
        delta = torch.norm(pos_xy - self._prev_pos, dim=-1)
        self._path_length += delta
        self._prev_pos = pos_xy.clone()

        # Record escape step
        new_esc = newly_escaped & ~self._escaped_flag
        self._escape_step = torch.where(new_esc, self._step, self._escape_step)
        self._escaped_flag |= newly_escaped
        self._escape_dir[new_esc] = escape_dir[new_esc].clone()

        # Record goal step
        new_arr = arrived & ~self._reached_flag
        self._goal_step = torch.where(new_arr, self._step, self._goal_step)
        self._reached_flag |= arrived

    def end_trial(self, env_ids: torch.Tensor):
        """Call when trial ends. Returns list of TrialResult."""
        results = []
        for i in env_ids.tolist():
            sl = torch.norm(self.goal_xy[i] - self._start_pos[i]).item() \
                 if self.goal_xy.dim() == 2 else \
                 torch.norm(self.goal_xy - self._start_pos[i]).item()

            pl = max(self._path_length[i].item(), 1e-6)
            eff = sl / pl

            # Escape heading error vs true direction to B
            goal = self.goal_xy[i] if self.goal_xy.dim() == 2 else self.goal_xy
            true_dir = goal - self._start_pos[i]
            true_dir = true_dir / (torch.norm(true_dir) + 1e-6)
            dot = (self._escape_dir[i] * true_dir).sum().clamp(-1.0, 1.0)
            heading_err_deg = math.degrees(math.acos(dot.item()))

            results.append(TrialResult(
                trial_id           = self._trial_id,
                reached_goal       = bool(self._reached_flag[i].item()),
                escaped            = bool(self._escaped_flag[i].item()),
                time_to_escape     = int(self._escape_step[i].item()),
                time_to_goal       = int(self._goal_step[i].item()),
                path_length        = pl,
                straight_line_dist = sl,
                path_efficiency    = eff,
                escape_heading_error = heading_err_deg,
            ))
            self._trial_id += 1

        self.results.extend(results)
        return results

    def summary(self) -> dict:
        """Compute aggregate statistics across all completed trials."""
        if not self.results:
            return {}

        reached  = [r for r in self.results if r.reached_goal]
        escaped  = [r for r in self.results if r.escaped]
        esc_times = [r.time_to_escape for r in escaped if r.time_to_escape > 0]
        goal_times = [r.time_to_goal  for r in reached if r.time_to_goal > 0]

        def _stats(vals):
            if not vals: return {"mean": float("nan"), "std": float("nan")}
            a = np.array(vals, dtype=float)
            return {"mean": float(np.mean(a)), "std": float(np.std(a))}

        return {
            "n_trials":            len(self.results),
            "recovery_rate":       len(escaped)  / max(1, len(self.results)),
            "goal_reach_rate":     len(reached)  / max(1, len(self.results)),
            "time_to_escape":      _stats(esc_times),
            "time_to_goal":        _stats(goal_times),
            "path_efficiency":     _stats([r.path_efficiency for r in reached]),
            "heading_error_deg":   _stats([r.escape_heading_error for r in escaped]),
        }

    def print_summary(self):
        s = self.summary()
        if not s:
            print("No trials completed yet.")
            return
        print("\n" + "="*55)
        print("  Sim2Sim Validation Summary")
        print("="*55)
        print(f"  Trials:          {s['n_trials']}")
        print(f"  Recovery rate:   {s['recovery_rate']*100:.1f}%")
        print(f"  Goal reach rate: {s['goal_reach_rate']*100:.1f}%")
        t = s['time_to_escape']
        print(f"  Time to escape:  {t['mean']:.1f} ± {t['std']:.1f} steps")
        g = s['time_to_goal']
        print(f"  Time to goal:    {g['mean']:.1f} ± {g['std']:.1f} steps")
        e = s['path_efficiency']
        print(f"  Path efficiency: {e['mean']:.3f} ± {e['std']:.3f}")
        h = s['heading_error_deg']
        print(f"  Heading error:   {h['mean']:.1f} ± {h['std']:.1f} deg")
        print("="*55 + "\n")
