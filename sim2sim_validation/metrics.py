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
    time_to_escape:    int
    time_to_goal:      int
    path_length:       float
    straight_line_dist: float
    path_efficiency:   float
    escape_heading_error: float


class MetricsTracker:

    def __init__(self, num_envs: int, device: str, goal_xy: torch.Tensor,
                 attribute_escape: bool = True):
        self.num_envs        = num_envs
        self.device          = device
        self.goal_xy         = goal_xy.to(device)
        self.attribute_escape = attribute_escape

        self.results: List[TrialResult] = []


        self._reset_accumulators()

    def _reset_accumulators(self):
        self._step          = torch.zeros(self.num_envs, device=self.device)
        self._path_length   = torch.zeros(self.num_envs, device=self.device)
        self._prev_pos      = torch.zeros(self.num_envs, 2, device=self.device)
        self._escape_step   = torch.full((self.num_envs,), -1.0, device=self.device)
        self._trigger_step  = torch.full((self.num_envs,), -1.0, device=self.device)
        self._trigger_pos   = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_step     = torch.full((self.num_envs,), -1.0, device=self.device)
        self._start_pos     = torch.zeros(self.num_envs, 2, device=self.device)
        self._escaped_flag  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reached_flag  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._escape_dir    = torch.zeros(self.num_envs, 2, device=self.device)
        self._trial_id      = 0

    def begin_trial(self, env_ids: torch.Tensor, start_pos: torch.Tensor):
        self._step[env_ids]         = 0
        self._path_length[env_ids]  = 0.0
        self._prev_pos[env_ids]     = start_pos[env_ids].clone()
        self._start_pos[env_ids]    = start_pos[env_ids].clone()
        self._escape_step[env_ids]  = -1.0
        self._trigger_step[env_ids] = -1.0
        self._trigger_pos[env_ids]  = start_pos[env_ids].clone()
        self._goal_step[env_ids]    = -1.0
        self._escaped_flag[env_ids] = False
        self._reached_flag[env_ids] = False

    def step(
        self,
        pos_xy:           torch.Tensor,
        newly_triggered:  torch.Tensor,
        newly_escaped:    torch.Tensor,
        arrived:          torch.Tensor,
        escape_dir:       torch.Tensor,
    ):
        self._step += 1


        delta = torch.norm(pos_xy - self._prev_pos, dim=-1)
        self._path_length += delta
        self._prev_pos = pos_xy.clone()


        if self.attribute_escape:
            first_trig = newly_triggered & (self._trigger_step < 0)
            self._trigger_step = torch.where(first_trig, self._step, self._trigger_step)
            self._trigger_pos[first_trig] = pos_xy[first_trig].clone()

            new_esc = newly_escaped & ~self._escaped_flag
            self._escape_step = torch.where(new_esc, self._step, self._escape_step)
            self._escaped_flag |= newly_escaped
            self._escape_dir[new_esc] = escape_dir[new_esc].clone()


        new_arr = arrived & ~self._reached_flag
        self._goal_step = torch.where(new_arr, self._step, self._goal_step)
        self._reached_flag |= arrived

    def end_trial(self, env_ids: torch.Tensor):
        results = []
        for i in env_ids.tolist():
            sl = torch.norm(self.goal_xy[i] - self._start_pos[i]).item() \
                 if self.goal_xy.dim() == 2 else \
                 torch.norm(self.goal_xy - self._start_pos[i]).item()

            pl = max(self._path_length[i].item(), 1e-6)
            eff = sl / pl


            goal = self.goal_xy[i] if self.goal_xy.dim() == 2 else self.goal_xy
            true_dir = goal - self._trigger_pos[i]
            true_dir = true_dir / (torch.norm(true_dir) + 1e-6)
            dot = (self._escape_dir[i] * true_dir).sum().clamp(-1.0, 1.0)
            heading_err_deg = math.degrees(math.acos(dot.item()))


            esc_step = int(self._escape_step[i].item())
            trig_step = int(self._trigger_step[i].item())
            t_to_esc = (esc_step - trig_step) if (esc_step > 0 and trig_step >= 0) else -1

            results.append(TrialResult(
                trial_id           = self._trial_id,
                reached_goal       = bool(self._reached_flag[i].item()),
                escaped            = bool(self._escaped_flag[i].item()),
                time_to_escape     = t_to_esc,
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
            "trials": [
                {
                    "trial_id":             r.trial_id,
                    "reached_goal":         r.reached_goal,
                    "escaped":              r.escaped,
                    "time_to_escape":       r.time_to_escape,
                    "time_to_goal":         r.time_to_goal,
                    "path_length":          r.path_length,
                    "straight_line_dist":   r.straight_line_dist,
                    "path_efficiency":      r.path_efficiency,
                    "escape_heading_error": r.escape_heading_error,
                }
                for r in self.results
            ],
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
