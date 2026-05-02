import argparse
import os
import sys
import json
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Sim Validation: A→B with entrapment recovery")
parser.add_argument("--checkpoint",   type=str,   required=True,
                    help="Path to trained PPO checkpoint (.pt). "
                         "Note: skrl's `best_agent.pt` is selected by total reward; "
                         "the highest-escape-rate checkpoint is usually the final one "
                         "(e.g. agent_200000.pt). Prefer the final checkpoint for eval "
                         "unless you specifically want the best-reward snapshot.")
parser.add_argument("--num_envs",     type=int,   default=8,
                    help="Parallel environments (must be ≤ training num_envs for PPO_RNN states)")
parser.add_argument("--num_trials",   type=int,   default=20,
                    help="Total trials to run (spread across envs)")
parser.add_argument("--goal_x",       type=float, default=3.0,
                    help="Goal B position X in env-local frame (m). "
                         "Default 3.0 places B on platform east of sand pit.")
parser.add_argument("--goal_y",       type=float, default=0.0,
                    help="Goal B position Y in env-local frame (m).")
parser.add_argument("--max_steps",    type=int,   default=2000,
                    help="Max policy steps per trial before timeout")
parser.add_argument("--seed",         type=int,   default=123)
parser.add_argument("--experiments",  type=str,   default="recovery_gps",
                    help="Comma-separated list of experiment modes to run sequentially "
                         "in a single Newton init. Options: recovery_gps, "
                         "recovery_random, no_recovery. Example: "
                         "--experiments recovery_gps,recovery_random,no_recovery")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import warp as wp

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO_RNN
from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs
from sim2sim_validation.validation_env import ValidationEnv, ValidationEnvCfg

from sim2sim_validation.nav_controller import PDNavController
from sim2sim_validation.mode_switcher import ModeSwitcher, Mode
from sim2sim_validation.metrics import MetricsTracker


GRU_HIDDEN     = 256
GRU_LAYERS     = 1
SEQ_LEN        = 32
ROLLOUTS       = 64
POLICY_OBS_DIM = 29


class GRUPolicyNet(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)
        self._num_envs = num_envs
        self._hidden   = GRU_HIDDEN
        self._layers   = GRU_LAYERS
        self._seq_len  = SEQ_LEN

        self.encoder = nn.Sequential(nn.Linear(POLICY_OBS_DIM, 128), nn.ELU())
        self.gru     = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head    = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states = inputs["states"][:, :POLICY_OBS_DIM]
        rnn_list = inputs.get("rnn", [None])
        hidden   = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states).view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        output = self.head(x)
        return output, self.log_std.expand_as(output), {"rnn": [h_n]}


class GRUValueNet(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._num_envs = num_envs
        self._hidden   = GRU_HIDDEN
        self._layers   = GRU_LAYERS
        self._seq_len  = SEQ_LEN

        self.encoder = nn.Sequential(nn.Linear(self.num_observations, 128), nn.ELU())
        self.gru     = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head    = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states   = inputs["states"]
        rnn_list = inputs.get("rnn", [None])
        hidden   = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None

        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1

        x = self.encoder(states).view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        return self.head(x), {"rnn": [h_n]}


def build_agent(env, device, num_envs, checkpoint_path):


    num_obs = env.unwrapped.cfg.observation_space
    num_act = env.unwrapped.cfg.action_space
    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))

    policy = GRUPolicyNet(obs_space, act_space, device, num_envs)
    value  = GRUValueNet(obs_space, act_space, device, num_envs)

    models = {"policy": policy, "value": value}

    cfg = PPO_RNN_DEFAULT_CONFIG.copy()
    cfg["state_preprocessor"]       = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": num_obs, "device": device}
    cfg["value_preprocessor"]       = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=num_envs, device=device)

    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )


    agent.init()
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent


def reset_agent_rnn_state(agent):
    for attr in ("_rnn_initial_states", "_rnn_final_states"):
        states = getattr(agent, attr, None)
        if not states:
            continue
        for role_states in states.values():
            for h in role_states:
                if hasattr(h, "zero_"):
                    h.zero_()


def get_pos_yaw(env_unwrapped):
    pose = wp.to_torch(env_unwrapped.robot.data.root_link_pose_w)
    pos_xy = pose[:, :2].clone()
    qx, qy, qz, qw = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    yaw = torch.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return pos_xy, yaw


def override_escape_dir(env_unwrapped, goal_xy_tensor):
    pose = wp.to_torch(env_unwrapped.robot.data.root_link_pose_w)
    pos_xy = pose[:, :2]
    rel = goal_xy_tensor - pos_xy
    dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)
    env_unwrapped._escape_dir = rel / dist


def run_experiment(
    mode_name: str,
    env, unwrapped: ValidationEnv, agent,
    goal_xy: torch.Tensor,
    num_envs: int, num_trials: int, max_steps: int, device: str,
):
    assert mode_name in ("recovery_gps", "recovery_random", "no_recovery"), mode_name

    nav = PDNavController(
        num_envs=num_envs, device=device,
        drive_speed=0.6, heading_gain=1.2, arrival_radius=0.5,
    )
    nav.set_goal(goal_xy)

    switcher = ModeSwitcher(
        num_envs=num_envs, device=device,
        escape_distance=3.0, trigger_steps=15,
    )

    metrics = MetricsTracker(
        num_envs=num_envs, device=device, goal_xy=goal_xy,


        attribute_escape=(mode_name != "no_recovery"),
    )

    trials_done = 0
    trial_step  = torch.zeros(num_envs, dtype=torch.long, device=device)
    all_env_ids = torch.arange(num_envs, device=device)


    trace = {
        "t":           [],
        "pos_xy":      [],
        "action":      [],
        "mode":        [],
        "entrap_flag": [],
        "spawn_xy":    None,
        "goal_xy":     goal_xy[0].detach().cpu().tolist(),
    }
    POLICY_DT = 0.04

    obs, _ = env.reset()
    pos_xy, _ = get_pos_yaw(unwrapped)
    metrics.begin_trial(all_env_ids, pos_xy)
    switcher.reset(all_env_ids)
    trace["spawn_xy"] = pos_xy[0].detach().cpu().tolist()

    if mode_name == "recovery_gps":
        override_escape_dir(unwrapped, goal_xy)


    states = obs
    capturing_trace = True

    print(f"\n[exp:{mode_name}] {num_trials} trials  "
          f"(goal B in env-local frame = [{args_cli.goal_x:.1f}, "
          f"{args_cli.goal_y:.1f}] m)\n")

    while trials_done < num_trials:
        pos_xy, yaw = get_pos_yaw(unwrapped)

        entrap_flag = states[:, 26].clamp(0.0, 1.0)
        mode_fsm, newly_triggered, newly_escaped, _ = switcher.update(pos_xy, entrap_flag, goal_xy)


        if mode_name == "recovery_gps":
            just_triggered = (mode_fsm == Mode.ESCAPE) & (switcher.steps_in_escape == 1)
            if just_triggered.any():
                for i in just_triggered.nonzero(as_tuple=True)[0].tolist():
                    unwrapped._escape_dir[i] = switcher.escape_dir[i]

        nav_action, arrived = nav.step(pos_xy, yaw)

        with torch.no_grad():


            ppo_action, _, _ = agent.act(states, timestep=0, timesteps=1)

        if mode_name == "no_recovery":

            action = nav_action
        else:
            navigating_mask = (mode_fsm != Mode.ESCAPE).unsqueeze(-1).float()
            action = nav_action * navigating_mask + ppo_action * (1.0 - navigating_mask)

        obs_next, _, terminated, truncated, _ = env.step(action)
        states = obs_next

        if capturing_trace:
            trace["t"].append(float(trial_step[0].item()) * POLICY_DT)
            trace["pos_xy"].append(pos_xy[0].detach().cpu().tolist())
            trace["action"].append(action[0].detach().cpu().tolist())
            trace["mode"].append(int(mode_fsm[0].item()))
            trace["entrap_flag"].append(float(entrap_flag[0].item()))

        metrics.step(
            pos_xy=pos_xy,
            newly_triggered=newly_triggered,
            newly_escaped=newly_escaped,
            arrived=arrived,
            escape_dir=switcher.escape_dir,
        )

        trial_step += 1
        done_mask = terminated | truncated | (trial_step >= max_steps) | arrived

        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=True)[0]

            if capturing_trace and (0 in done_ids.tolist()):
                capturing_trace = False
            finished = metrics.end_trial(done_ids)
            for r in finished:
                trials_done += 1
                status = ("GOAL" if r.reached_goal else
                          "ESCAPED" if r.escaped else "TIMEOUT")
                print(f"  Trial {r.trial_id:3d} [{status:7s}] "
                      f"esc={r.time_to_escape:4d}s  "
                      f"goal={r.time_to_goal:4d}s  "
                      f"eff={r.path_efficiency:.3f}  "
                      f"hdg_err={r.escape_heading_error:.1f}°")
                if trials_done >= num_trials:
                    break

            if trials_done < num_trials:
                fresh_pos_xy, _ = get_pos_yaw(unwrapped)
                trial_step[done_ids] = 0
                metrics.begin_trial(done_ids, fresh_pos_xy)
                switcher.reset(done_ids)
                if mode_name == "recovery_gps":
                    pose_all = wp.to_torch(unwrapped.robot.data.root_link_pose_w)
                    pos_done = pose_all[done_ids, :2]
                    rel = goal_xy[done_ids] - pos_done
                    dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)
                    unwrapped._escape_dir[done_ids] = rel / dist

    metrics.print_summary()
    summary = metrics.summary()
    summary["trace"] = trace
    return summary


def main():
    set_seed(args_cli.seed)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = args_cli.num_envs

    cfg = ValidationEnvCfg()
    cfg.scene.num_envs = num_envs


    cfg.val_goal_x = args_cli.goal_x
    cfg.val_goal_y = args_cli.goal_y

    print(f"[sim2sim] Calling gym.make (Newton init starts here — first cold launch can take 10-15 min)…", flush=True)
    raw_env = gym.make("MarsRover-Sim2SimValidation-v0", cfg=cfg)
    print(f"[sim2sim] gym.make returned. Wrapping env…", flush=True)
    env     = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    unwrapped: ValidationEnv = raw_env.unwrapped
    unwrapped._total_timesteps = 200_000
    print(f"[sim2sim] Env ready (num_envs={num_envs}). Building agent…", flush=True)

    print(f"[sim2sim] Loading checkpoint: {args_cli.checkpoint}")
    agent = build_agent(env, device, num_envs, args_cli.checkpoint)
    print(f"[sim2sim] Agent initialized. Starting experiments.", flush=True)

    goal_world = torch.tensor([args_cli.goal_x, args_cli.goal_y],
                              dtype=torch.float32, device=device)
    env_origins_xy = unwrapped.scene.env_origins[:, :2]
    goal_xy = env_origins_xy + goal_world.unsqueeze(0)

    experiment_modes = [m.strip() for m in args_cli.experiments.split(",") if m.strip()]
    print(f"[sim2sim] Experiments to run: {experiment_modes}")

    all_summaries = {}
    for mode_name in experiment_modes:


        reset_agent_rnn_state(agent)
        summary = run_experiment(
            mode_name=mode_name,
            env=env, unwrapped=unwrapped, agent=agent,
            goal_xy=goal_xy,
            num_envs=num_envs,
            num_trials=args_cli.num_trials,
            max_steps=args_cli.max_steps,
            device=device,
        )
        all_summaries[mode_name] = summary

    out_dir = os.path.join(REPO_ROOT, "experiments", "sim2sim")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"summary_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args_cli), "experiments": all_summaries}, f, indent=2)
    print(f"\n[sim2sim] All experiments complete. Results → {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
