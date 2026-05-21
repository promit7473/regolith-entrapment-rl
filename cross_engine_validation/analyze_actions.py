"""Log per-step actions in AAU Chrono trials to understand sim gap.

Saves action traces to .npz and prints comparative stats.
"""
import sys
import os
import time
import math
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from chrono_validation import (
    run_trial, build_scene_aau, CuriosityObsBuilder, BekkerWheelTerrain,
    _read_omega, _damp_chassis, _torque_from_bekker, _probe_motor_func,
    POLICY_OBS_DIM, ACT_DIM, GRU_HIDDEN, GRU_LAYERS,
    DRIVE_VEL_LIMIT, STEER_LIMIT, WHEEL_RADIUS_AAU, WHEEL_RADIUS_CHRONO,
    POLICY_DT, PHYSICS_DT, SUBSTEPS, MAX_POLICY_STEPS,
    ENTRAP_VX_THRESH, ENTRAP_SLIP_THRESH, ENTRAP_HOLD_STEPS,
    ESCAPE_DISTANCE, AAU_WHEEL_POS, AAU_STEER_INDICES,
    BW_KPHI, BW_KC, BW_N, BW_COHESION, BW_K, BW_ELASTIC, BW_DAMP,
    TERRAIN_Z, G_MARS, BULLDOZE_DAMP,
)

ONNX = "/media/rmedu/18C6E68BC6E66888/regolith-entrapment-rl/sim2real/onnx_export/output/recovery_policy.onnx"
N = 10  # trials per condition

import onnxruntime as ort

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])


def run_and_trace(control_mode: str, sess=None, seed: int = 42,
                  n_trials: int = N) -> list[dict]:
    """Run N AAU trials with action logging. Returns list of trace dicts."""
    all_traces = []
    rng = np.random.default_rng(seed)
    wheel_idx = range(6)
    steer_idx = AAU_STEER_INDICES
    w_radius = WHEEL_RADIUS_AAU

    for tid in range(n_trials):
        t0 = time.time()
        friction_deg = float(rng.uniform(10, 30))
        heading = float(rng.uniform(0, 2 * math.pi))
        friction_rad = math.radians(friction_deg)

        # Create terrain torque observers
        torque_wts = [BekkerWheelTerrain(friction_rad, wheel_radius=w_radius)
                      for _ in range(6)]

        sys, rover, driver, spawn, terrain = build_scene_aau(
            "granular", friction_rad, friction_deg)

        escape_dir = np.array([math.cos(heading), math.sin(heading)])
        spawn_xy = np.array([spawn.x, spawn.y])

        obs_builder = CuriosityObsBuilder(rover, wheel_radius=w_radius)
        h_state = np.zeros((GRU_LAYERS, 1, GRU_HIDDEN), dtype=np.float32)

        _DRIVE_NAMES = ("GetDriveMotorFunc", "GetWheelMotorFunc",
                        "GetDriveMotor", "GetWheelMotor")
        _STEER_NAMES = ("GetRockerSteerMotorFunc", "GetSteerMotorFunc",
                        "GetRockerSteerMotor", "GetSteerMotor")
        drive_funcs = [_probe_motor_func(rover, wid, _DRIVE_NAMES)
                       for wid in wheel_idx]
        steer_funcs = [_probe_motor_func(rover, wid, _STEER_NAMES)
                       for wid in steer_idx]
        wheel_bodies = [rover.GetWheel(wid).GetBody() for wid in wheel_idx]

        # Settle
        for gs in range(200):
            terrain.Synchronize(sys.GetChTime())
            rover.Update()
            sys.DoStepDynamics(PHYSICS_DT)

        # Per-step action trace
        actions: list[np.ndarray] = []  # [step, 10] (6 drive + 4 steer)
        states: list[np.ndarray] = []   # [step, 3]  (v_x, proj, slip)
        sinkages: list[np.ndarray] = []
        omega_axis = "y"
        applied_torques = np.zeros(6, dtype=np.float32)

        escaped = False
        t_to_escape = -1
        final_proj = 0.0
        step = 0

        for step in range(MAX_POLICY_STEPS):
            wheel_omegas = np.array(
                [_read_omega(wb, omega_axis) for wb in wheel_bodies])
            obs, info = obs_builder.build(
                escape_dir, spawn_xy, wheel_omegas, applied_torques)
            rover.Update()

            if control_mode == "policy":
                action_mean, h_state = sess.run(
                    ["action", "h_out"],
                    {"obs": obs.reshape(1, -1),
                     "h_in": h_state},
                )
                action = action_mean[0]
            else:
                action = np.array(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0, 0.0])

            actions.append(action.copy())
            states.append(np.array([
                info["v_x_body"], info["proj_dist"], info["mean_slip"]]))

            sim_t = sys.GetChTime()
            for k, f in enumerate(drive_funcs):
                if f is not None:
                    f.SetSetpoint(float(action[k]) * DRIVE_VEL_LIMIT, sim_t)

            steer_cmds = np.clip(action[6:10], -1.0, 1.0) * STEER_LIMIT
            for j, f in enumerate(steer_funcs):
                if f is not None:
                    f.SetSetpoint(float(steer_cmds[j]), sim_t)

            for ss in range(SUBSTEPS):
                v_x_step = float(rover.GetChassisVel().x)
                tq_step = np.zeros(6)
                for k, (wid, wt) in enumerate(zip(wheel_idx, torque_wts)):
                    wb = rover.GetWheel(wid).GetBody()
                    wpos = wb.GetPos()
                    tq_step[k] = _torque_from_bekker(
                        float(wheel_omegas[k]), v_x_step,
                        float(wpos.z), wt, PHYSICS_DT)
                bd_torques = _damp_chassis(rover, wheel_bodies,
                                            w_radius, friction_deg)
                applied_torques = tq_step + bd_torques
                terrain.Synchronize(sys.GetChTime())
                sys.DoStepDynamics(PHYSICS_DT)
                sk = np.array([
                    max(0.0, w_radius - float(
                        rover.GetWheel(wid).GetBody().GetPos().z))
                    for wid in wheel_idx
                ])
                sinkages.append(sk)

            final_proj = info["proj_dist"]
            if info["proj_dist"] >= ESCAPE_DISTANCE:
                escaped = True
                t_to_escape = step
                break

            cz = info["chassis_pos"][2]
            if cz < -2.0 or cz > 5.0:
                break

        dt = time.time() - t0
        actions_arr = np.array(actions)
        states_arr = np.array(states)
        sinkages_arr = np.array(sinkages)

        trace = {
            "trial_id":     tid,
            "friction":     friction_deg,
            "heading":      heading,
            "escaped":      escaped,
            "time_to_escape": t_to_escape,
            "final_proj":   final_proj,
            "mean_sinkage": float(np.mean(sinkages_arr)) if len(sinkages_arr) else 0.0,
            "actions":      actions_arr,
            "states":       states_arr,
            "sinkages":     sinkages_arr,
            "control":      control_mode,
        }
        all_traces.append(trace)

        tag = "ESC" if escaped else "FAIL"
        print(f"  [{control_mode:>14s}] trial {tid:2d}  φ={friction_deg:.1f}°  "
              f"[{tag:>10s}]  proj={final_proj:5.2f}m  "
              f"sink={trace['mean_sinkage']:.3f}m  actions={actions_arr.shape[0]}  ({dt:.1f}s)")

    return all_traces


def analyze(traces_policy: list[dict], traces_drive: list[dict]):
    """Compare action profiles between policy and constant_drive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"policy": "Policy (seed 1)", "constant_drive": "Full Throttle"}

    # Aggregate stats
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<35} {'Policy':>15} {'ConstDrive':>15}")
    print(f"{'-' * 35} {'-' * 15} {'-' * 15}")
    for name, traces in [("policy", traces_policy), ("constant_drive", traces_drive)]:
        esc = sum(1 for t in traces if t["escaped"])
        proj = np.mean([t["final_proj"] for t in traces])
        sink = np.mean([t["mean_sinkage"] for t in traces])
        acts = [t["actions"] for t in traces]
        mean_throttle = np.mean([np.mean(a[:, :6]) for a in acts])
        mean_steer    = np.mean([np.mean(np.abs(a[:, 6:])) for a in acts])
        throttle_std  = np.mean([np.std(a[:, :6]) for a in acts])
        steer_std     = np.mean([np.std(a[:, 6:]) for a in acts])
        vx = np.mean([np.mean(t["states"][:, 0]) for t in traces])

        label = labels[name]
        if label.startswith("Policy"):
            p_esc, p_proj, p_sink, p_thr, p_steer, p_thrstd, p_ststd, p_vx = \
                esc, proj, sink, mean_throttle, mean_steer, throttle_std, steer_std, vx
        else:
            d_esc, d_proj, d_sink, d_thr, d_steer, d_thrstd, d_ststd, d_vx = \
                esc, proj, sink, mean_throttle, mean_steer, throttle_std, steer_std, vx

    print(f"{'Escape count':<35} {p_esc:>4}/{N} {d_esc:>4}/{N}")
    print(f"{'Escape rate':<35} {p_esc/N*100:>14.1f}% {d_esc/N*100:>14.1f}%")
    print(f"{'Final proj (m)':<35} {p_proj:>15.3f} {d_proj:>15.3f}")
    print(f"{'Mean sinkage (m)':<35} {p_sink:>15.3f} {d_sink:>15.3f}")
    print(f"{'Mean throttle':<35} {p_thr:>15.3f} {d_thr:>15.3f}")
    print(f"{'Mean |steer|':<35} {p_steer:>15.4f} {d_steer:>15.4f}")
    print(f"{'Throttle std (temporal)':<35} {p_thrstd:>15.3f} {d_thrstd:>15.3f}")
    print(f"{'Steer std (temporal)':<35} {p_ststd:>15.4f} {d_ststd:>15.4f}")
    print(f"{'Mean v_x (m/s)':<35} {p_vx:>15.3f} {d_vx:>15.3f}")

    # Plot action profiles for first trial
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (name, traces, color) in enumerate([
        ("policy", traces_policy, "C0"),
        ("constant_drive", traces_drive, "C1"),
    ]):
        t = traces[0]
        ax_drive = axes[row, 0]
        ax_steer = axes[row, 1]
        steps = np.arange(t["actions"].shape[0])

        for wi in range(6):
            ax_drive.plot(steps, t["actions"][:, wi], lw=0.8,
                          label=f"W{wi}" if row == 0 else None,
                          color=f"C{wi}")
        ax_drive.set_ylabel(f"{name}: drive (rad/s)")
        ax_drive.set_ylim(-3, 9)

        for si in range(4):
            ax_steer.plot(steps, t["actions"][:, 6 + si], lw=0.8,
                          label=f"S{si}" if row == 0 else None,
                          color=f"C{si}")
        ax_steer.set_ylabel(f"{name}: steer (rad)")
        ax_steer.set_ylim(-1.2, 1.2)

        if row == 0:
            ax_drive.legend(fontsize=6, ncol=3)
            ax_steer.legend(fontsize=6, ncol=2)

    axes[0, 0].set_title("Drive speeds")
    axes[0, 1].set_title("Steer angles")
    for ax in axes[1, :]:
        ax.set_xlabel("Policy step")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "results", "action_profiles.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    # Compare per-trial action statistics
    print(f"\n{'=' * 70}")
    print("Per-trial mean throttle & steer variance")
    print(f"{'trial':>5} {'pol_thr':>8} {'drive_thr':>10} "
          f"{'pol_|steer|':>10} {'drive_|steer|':>11} "
          f"{'pol_vx':>7} {'drive_vx':>9}")
    for i in range(N):
        pa = traces_policy[i]["actions"]
        da = traces_drive[i]["actions"]
        pt = float(np.mean(pa[:, :6]))
        dt = float(np.mean(da[:, :6]))
        ps = float(np.mean(np.abs(pa[:, 6:])))
        ds_ = float(np.mean(np.abs(da[:, 6:])))
        pv = float(np.mean(traces_policy[i]["states"][:, 0]))
        dv = float(np.mean(traces_drive[i]["states"][:, 0]))
        print(f"{i:>5} {pt:>8.3f} {dt:>10.3f} {ps:>10.4f} {ds_:>11.4f} {pv:>7.3f} {dv:>9.3f}")


print("Running AAU policy trials with action tracing...")
traces_policy = run_and_trace("policy", sess)
savename = os.path.join(os.path.dirname(__file__), "results", "aau_policy_action_traces.npz")
np.savez(savename,
         actions=np.array([t["actions"] for t in traces_policy], dtype=object),
         states=np.array([t["states"] for t in traces_policy], dtype=object),
         meta=np.array([[t["escaped"], t["friction"], t["heading"], t["final_proj"]]
                        for t in traces_policy]))

print(f"\nRunning AAU constant_drive trials with action tracing...")
traces_drive = run_and_trace("constant_drive")

analyze(traces_policy, traces_drive)
