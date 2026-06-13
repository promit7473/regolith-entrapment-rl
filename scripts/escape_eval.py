"""Unified in-engine (Newton/MPM) escape evaluation — policy vs naive baselines.

Runs the SAME sinkage sweep under three controllers so escape rates are directly
comparable in the engine the policy was actually trained in:

  --control constant_drive : full forward throttle, zero steer  (naive baseline,
                             identical to the Chrono `constant_drive` baseline)
  --control rocking        : scripted alternating forward/backward drive, no steer
                             (hand-crafted field-robotics recovery primitive)
  --control policy         : learned GRU recovery policy (requires --checkpoint)

Purpose: establish that the naive baselines CANNOT escape the calibrated
entrapment (~0%), then show the learned policy can. This is the experiment the
Chrono validation failed to deliver (there policy 35% ≈ constant-drive 34%
because SCM sinkage was ~2 cm and the synthetic bulldoze term never engaged).

Examples:
    ./launch.sh scripts/escape_eval.py --headless --control constant_drive \
        --num_envs 64 --out_json experiments/escape_eval/constdrv.json
    ./launch.sh scripts/escape_eval.py --headless --control rocking \
        --num_envs 64 --out_json experiments/escape_eval/rocking.json
    ./launch.sh scripts/escape_eval.py --headless --control policy \
        --num_envs 64 --checkpoint <CKPT.pt> \
        --out_json experiments/escape_eval/policy.json

All three share --seed, --episodes_per_level and --sinkage_levels so the trial
distributions match. Output JSON has per-level + overall escape rate with a
bootstrap 95% CI and (for policy) the recorded action trace for behaviour
analysis (answers "did it just learn continuous drive?").
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="In-engine escape eval (policy vs baselines)")
parser.add_argument("--control",
                    choices=["policy", "constant_drive", "rocking",
                             "inching", "steer_paddle", "rock_paddle",
                             "spiral"],
                    required=True, help="Controller under test. "
                    "inching = Creager et al. 2015 push-pull extrication "
                    "(J. Terramechanics 57) approximated as alternating "
                    "locked/driving wheel groups; steer_paddle = Shrivastava "
                    "et al. 2020 (Sci. Robotics, RP15) cyclic-sweep paddling "
                    "approximated as full-lock sinusoidal steering + drive.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Policy checkpoint (.pt) — required for --control policy")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes_per_level", type=int, default=50)
parser.add_argument("--sinkage_levels", type=str, default="0.15,0.20,0.25,0.28",
                    help="Comma-separated initial sinkage depths (m) to sweep")
parser.add_argument("--friction_override", type=float, default=None,
                    help="Pin MPM sand friction (else DR range used)")
parser.add_argument("--half_period", type=float, default=2.0,
                    help="Half-cycle seconds for the cyclic baselines "
                         "(rocking drive flips, inching group swaps, "
                         "steer_paddle sweep half-period)")
parser.add_argument("--drive_mag", type=float, default=1.0,
                    help="Drive command magnitude for baselines [-1,1]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_json", type=str, default=None)
parser.add_argument("--out_csv", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.control == "policy" and not args_cli.checkpoint:
    parser.error("--control policy requires --checkpoint")

os.environ["LAUNCH_OV_APP"] = "1"
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import json

import numpy as np
import torch
import gymnasium as gym

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import envs  # noqa: F401 — registers MarsRover-RegolithEscape-v0
from envs.entrapment_env import EntrapmentEnvCfg, ESCAPE_DISTANCE


def _bootstrap_ci(x, n_boot=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for a binary array's mean."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (0.0, 0.0)
    rng = rng or np.random.default_rng(0)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _load_agent(device, num_obs, num_act, num_envs, checkpoint_path):
    """Load the trained GRU PPO agent (mirrors scripts/eval.py)."""
    from skrl.agents.torch.ppo import PPO_RNN
    from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # NOTE: import from gru_models, NOT train — train.py launches its own
    # SimulationApp at import time and deadlocks inside a running app.
    from gru_models import GRUPolicyNet, GRUValueNet, ROLLOUTS

    obs_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,))
    models = {
        "policy": GRUPolicyNet(obs_space, act_space, device, num_envs=num_envs),
        "value":  GRUValueNet(obs_space, act_space, device, num_envs=num_envs),
    }
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=num_envs, device=device)
    agent = PPO_RNN(models=models, memory=memory, cfg=PPO_RNN_DEFAULT_CONFIG.copy(),
                    observation_space=obs_space, action_space=act_space, device=device)
    # init() builds the agent's recurrent-state machinery (_rnn, _rnn_initial_states)
    # that act() relies on; without it act() raises AttributeError: no attribute '_rnn'.
    agent.init()
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    return agent


_DBG_OBS_PRINTED = False


def _as_states(obs):
    """Drill through any dict nesting (DirectRLEnv + gym wrappers) to the policy
    observation tensor. Env returns {"policy": tensor}; gym wrappers may nest it."""
    global _DBG_OBS_PRINTED
    if not _DBG_OBS_PRINTED:
        def _shape(o):
            if isinstance(o, dict):
                return {k: _shape(v) for k, v in o.items()}
            return type(o).__name__ + (str(tuple(o.shape)) if hasattr(o, "shape") else "")
        print(f"[escape_eval] obs structure: {_shape(obs)}", flush=True)
        _DBG_OBS_PRINTED = True
    x = obs
    while isinstance(x, dict):
        x = x["policy"] if "policy" in x else next(iter(x.values()))
    return x


def run_level(env, sinkage, n_eps, device, num_act, control,
              agent, half_period_steps, drive_mag, record_actions):
    """Run n_eps episodes at fixed sinkage under `control`; return outcome dicts."""
    unwrapped = env.unwrapped
    unwrapped.cfg.dr_sinkage_range = (sinkage, sinkage)

    results = []
    completed = 0
    step_counter = torch.zeros(unwrapped.num_envs, device=device)
    global_step = 0

    # Wheel-group masks for the inching (push-pull) baseline, built from the
    # actual drive-joint name order so index assumptions can't silently rot.
    drive_names = [unwrapped.robot.joint_names[i] for i in unwrapped._drive_ids]
    front_mask = torch.tensor(
        [1.0 if ("FL" in n or "FR" in n) else 0.0 for n in drive_names],
        device=device)
    rear_mid_mask = 1.0 - front_mask
    # Steer-corner sign mask for the spiral baseline (front +, rear −  =
    # double-Ackermann arc), from live steer-joint names.
    steer_names = [unwrapped.robot.joint_names[i] for i in unwrapped._steer_ids]
    steer_arc_sign = torch.tensor(
        [1.0 if ("FL" in n or "FR" in n) else -1.0 for n in steer_names],
        device=device)
    # SELF-VERIFICATION (this project's signature bug class is the silent
    # name-match no-op — chassis box, entropy floor, …). Fail loudly instead.
    print(f"[escape_eval] drive joints: {drive_names}")
    print(f"[escape_eval] steer joints: {steer_names}")
    print(f"[escape_eval] inching front mask = {front_mask.tolist()}, "
          f"spiral arc signs = {steer_arc_sign.tolist()}")
    assert front_mask.sum().item() == 2.0, \
        f"inching front mask must catch exactly FL+FR, got {front_mask.tolist()} for {drive_names}"
    assert steer_arc_sign.sum().item() == 0.0 and len(steer_names) == 4, \
        f"spiral arc mask must be +1,+1,-1,-1 over 4 corners, got {steer_arc_sign.tolist()} for {steer_names}"

    obs, _ = env.reset()

    while completed < n_eps:
        if control == "policy":
            # skrl PPO_RNN.act(states, ...) expects the raw states TENSOR, not a
            # {"states": ...} dict (it wraps internally). Pass the tensor directly.
            with torch.no_grad():
                actions, _, _ = agent.act(_as_states(obs),
                                          timestep=global_step, timesteps=0)
            action = actions
        elif control == "constant_drive":
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = drive_mag
        elif control == "rocking":
            phase = (step_counter // half_period_steps) % 2
            drive_sign = torch.where(phase == 0, torch.ones_like(phase),
                                     -torch.ones_like(phase))
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = (drive_sign * drive_mag).unsqueeze(-1).expand(-1, 6)
        elif control == "inching":
            # Creager et al. 2015 push-pull extrication (J. Terramechanics 57),
            # wheeled approximation for a passive rocker: alternate thrusting
            # wheel groups against held (0-velocity-target = braked) anchors.
            # Phase 0: rear+mid push, front anchors; phase 1: front pulls.
            phase = ((step_counter // half_period_steps) % 2).unsqueeze(-1)
            group = torch.where(
                phase == 0,
                rear_mid_mask.unsqueeze(0).expand(unwrapped.num_envs, -1),
                front_mask.unsqueeze(0).expand(unwrapped.num_envs, -1))
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = group * drive_mag
        elif control == "rock_paddle":
            # Combined operator strategy for drive+steer-only platforms: rocking
            # (alternating drive sign) superposed with steering sweeps — the
            # compatible-actuation "skilled operator" baseline. With parameter
            # tuning (scripts/tune_scripted_baseline.sh) this is the fairest
            # non-learning bar: the best scripted strategy this action space
            # admits.
            phase = (step_counter // half_period_steps) % 2
            drive_sign = torch.where(phase == 0, torch.ones_like(phase),
                                     -torch.ones_like(phase))
            sweep = torch.sin(
                2.0 * torch.pi * step_counter / float(2 * half_period_steps))
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = (drive_sign * drive_mag).unsqueeze(-1).expand(-1, 6)
            action[:, 6:10] = sweep.unsqueeze(-1).expand(-1, 4)
        elif control == "spiral":
            # Spiral egress heuristic (OURS — no published spiral extraction
            # method exists; searched 2026-06-13). Rationale: arcing shears
            # wheels sideways into fresh, uncompacted sand instead of the
            # self-dug rut; the lock angle relaxes linearly over the episode,
            # so a tight circle widens into translation — an Archimedean-ish
            # egress spiral. Steer: front +, rear − (double-Ackermann arc);
            # half_period sets the relaxation timescale (lock → straight over
            # 10 × half_period).
            relax_steps = float(10 * half_period_steps)
            lock = (1.0 - step_counter / relax_steps).clamp(min=0.15)
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = drive_mag
            action[:, 6:10] = lock.unsqueeze(-1) * steer_arc_sign.unsqueeze(0)
        else:  # steer_paddle
            # Shrivastava et al. 2020 (Sci. Robotics, RP15) cyclic-sweep
            # "paddling" gait, wheeled reduction: constant forward drive with
            # full-lock sinusoidal steering sweeps on all four corners
            # (sweep period = 2 × half_period).
            sweep = torch.sin(
                2.0 * torch.pi * step_counter / float(2 * half_period_steps))
            action = torch.zeros(unwrapped.num_envs, num_act, device=device)
            action[:, :6] = drive_mag
            action[:, 6:10] = sweep.unsqueeze(-1).expand(-1, 4)

        # Steering realization telemetry: for steering controllers, print
        # commanded vs realized corner angles a few times per run so "the
        # steering silently never moved" can never ship unnoticed again.
        if control in ("steer_paddle", "rock_paddle", "spiral") and \
                global_step in (37, 113, 411):  # prime-ish offsets — never
                # aligned with sweep zero-crossings (step-25-multiple periods
                # made commanded values print as 0.0 and confuse verification)
            try:
                import warp as _wp
                jp = _wp.to_torch(unwrapped.robot.data.joint_pos)[0, unwrapped._steer_ids]
                cmd = (action[0, 6:10] * 0.6).tolist()
                print(f"[escape_eval] {control} step {global_step}: steer cmd(rad)="
                      f"{[round(c,3) for c in cmd]} realized="
                      f"{[round(float(v),3) for v in jp]}")
            except Exception as e:
                print(f"[escape_eval] steer telemetry failed: {e}")

        if record_actions is not None:
            # record env 0's action + mean abs slip for behaviour analysis
            ob0 = _as_states(obs)[0].detach().cpu().numpy()
            record_actions.append({
                "sinkage": sinkage,
                "step": int(step_counter[0].item()),
                "action": action[0].detach().cpu().numpy().tolist(),
                "mean_abs_slip": float(np.mean(np.abs(ob0[6:12]))),
            })

        obs, _, terminated, truncated, _ = env.step(action)
        step_counter += 1.0
        global_step += 1
        done_mask = terminated | truncated

        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=True)[0]
            # Read the env's PRE-reset outcome flags (DirectRLEnv auto-resets done
            # envs inside step(), so root_pos/_spawn_pos here are already the new
            # episode). _last_escaped isolates true escapes from flip/sunk/oob.
            last_escaped    = unwrapped._last_escaped
            last_final_proj = unwrapped._last_final_proj
            last_rel_xy     = getattr(unwrapped, "_last_rel_xy", None)
            for ei in done_ids.tolist():
                if completed >= n_eps:
                    break
                proj = float(last_final_proj[ei].item())
                rel_x = float(last_rel_xy[ei, 0].item()) if last_rel_xy is not None else 0.0
                rel_y = float(last_rel_xy[ei, 1].item()) if last_rel_xy is not None else 0.0
                # Blowup guard: a legitimate escape lands just past ESCAPE_DISTANCE
                # (~3.0–3.1 m; per-step travel is <0.03 m). |proj| beyond BLOWUP_BOUND
                # means the MuJoCo-Warp solver exploded (seen under full throttle at
                # deep sinkage) — do NOT count that as an escape; flag it invalid.
                BLOWUP_BOUND = ESCAPE_DISTANCE + 1.0    # 4.0 m
                blowup  = abs(proj) > BLOWUP_BOUND
                escaped = bool(last_escaped[ei].item()) and not blowup
                results.append({
                    "sinkage":    sinkage,
                    "escaped":    int(escaped),
                    "blowup":     int(blowup),
                    "steps":      int(unwrapped._episode_step[ei].item()),
                    "final_dist": proj,
                    "rel_x":      rel_x,
                    "rel_y":      rel_y,
                })
                completed += 1
            step_counter[done_mask] = 0.0

    return results


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sinkage_levels = [float(s) for s in args_cli.sinkage_levels.split(",")]

    cfg = EntrapmentEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.log_failure_modes = False
    if args_cli.friction_override is not None:
        cfg.friction_override = args_cli.friction_override

    env = gym.make("MarsRover-RegolithEscape-v0", cfg=cfg)
    unwrapped = env.unwrapped
    device    = unwrapped.device
    num_act   = cfg.action_space

    agent = None
    if args_cli.control == "policy":
        agent = _load_agent(device, cfg.observation_space, num_act,
                            unwrapped.num_envs, args_cli.checkpoint)
        print(f"[escape_eval] loaded policy: {args_cli.checkpoint}")

    dt = float(cfg.sim.dt) * float(cfg.decimation)
    half_period_steps = max(1, int(args_cli.half_period / dt))
    record_actions = [] if args_cli.control == "policy" else None

    print(f"[escape_eval] control={args_cli.control}  seed={args_cli.seed}  "
          f"envs={args_cli.num_envs}  eps/level={args_cli.episodes_per_level}")
    print(f"[escape_eval] sinkage_levels={sinkage_levels}  "
          f"friction_override={args_cli.friction_override}")
    print()

    all_results = []
    per_level = {}
    for sinkage in sinkage_levels:
        print(f"  sinkage={sinkage:.2f}m  ({args_cli.episodes_per_level} eps)...", flush=True)
        lvl = run_level(env, sinkage, args_cli.episodes_per_level, device, num_act,
                        args_cli.control, agent, half_period_steps,
                        args_cli.drive_mag, record_actions)
        all_results.extend(lvl)
        esc = [r["escaped"] for r in lvl]
        rate = sum(esc) / max(1, len(esc))
        lo, hi = _bootstrap_ci(esc)
        n_blowup = sum(r.get("blowup", 0) for r in lvl)
        sane_proj = [r["final_dist"] for r in lvl if not r.get("blowup", 0)]
        per_level[f"{sinkage:.2f}"] = {
            "escape_rate": rate, "ci": [lo, hi], "n": len(esc),
            "n_blowup": n_blowup,
            "mean_final_dist": float(np.mean(sane_proj)) if sane_proj else float("nan"),
        }
        print(f"    escape_rate={rate:.3f}  CI=[{lo:.3f},{hi:.3f}]  "
              f"blowups={n_blowup}/{len(lvl)}  "
              f"avg_final_dist={per_level[f'{sinkage:.2f}']['mean_final_dist']:.2f}m")

    esc_all = [r["escaped"] for r in all_results]
    overall = sum(esc_all) / max(1, len(esc_all))
    o_lo, o_hi = _bootstrap_ci(esc_all)

    print()
    print("=" * 60)
    print(f"  IN-ENGINE (Newton/MPM) ESCAPE EVAL — control={args_cli.control}")
    print("=" * 60)
    print(f"  Overall escape rate : {overall:.3f}  95% CI=[{o_lo:.3f},{o_hi:.3f}]  "
          f"(N={len(esc_all)})")
    print("=" * 60)

    if args_cli.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_csv)) or ".", exist_ok=True)
        with open(args_cli.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sinkage", "escaped", "blowup", "steps",
                                              "final_dist", "rel_x", "rel_y"])
            w.writeheader()
            w.writerows(all_results)
        print(f"[escape_eval] CSV → {args_cli.out_csv}")

    if args_cli.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_json)) or ".", exist_ok=True)
        payload = {
            "control": args_cli.control,
            "checkpoint": args_cli.checkpoint,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "episodes_per_level": args_cli.episodes_per_level,
            "sinkage_levels": sinkage_levels,
            "friction_override": args_cli.friction_override,
            "overall_escape_rate": overall,
            "overall_ci": [o_lo, o_hi],
            "n_total": len(esc_all),
            "per_level": per_level,
        }
        if record_actions is not None:
            payload["action_trace"] = record_actions
        with open(args_cli.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[escape_eval] JSON → {args_cli.out_json}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
