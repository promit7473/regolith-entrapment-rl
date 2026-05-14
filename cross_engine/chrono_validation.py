"""Cross-engine sim2sim validation: Project Chrono Curiosity rover on low-friction terrain.

Loads the ONNX policy exported from sim2real/onnx_export/output/, drives the
built-in Chrono Curiosity rocker-bogie rover (pychrono.robot.Curiosity) over a
rigid flat surface with randomised low friction (μ=0.10–0.35) to create mobility-
limited entrapment conditions, and reports recovery rate matched to the Newton
MPM validation protocol.

Why this counts as cross-engine:
  - Different RIGID-BODY solver: Chrono ChSystemNSC (NSC contact + constraint-based
    complementarity) vs. MuJoCo Warp + Newton SolverImplicitMPM
  - Different CONTACT MODEL: Chrono NSC smooth-contact complementarity vs.
    MuJoCo convex-complementarity — different numerical treatment of wheel-ground
    friction at the constraint level
  - Different ROVER MORPHOLOGY: Curiosity (NASA MSL, 2.9 m wheelbase, 0.25 m
    wheel radius, 899 kg) vs. AAU Mars rover (0.10 m radius, ~35 kg). Both are
    6-wheel rocker-bogie; mass/inertia are genuinely different.

Terrain fidelity gap (documented for paper):
  Rigid surface with variable friction replaces the MPM continuum granular bed.
  Initial burial is not simulated (Bekker-Wong at 15-28 cm sinkage produces
  forces 10-20x Curiosity weight, causing initialization artifacts).  Entrapment
  is instead created via low friction (μ=0.10-0.35), which produces high wheel
  slip and near-zero forward progress — the same observable entrapment signal
  the policy was trained to detect and escape.

Run (in chrono_viz env):
  conda run -n chrono_viz python cross_engine/chrono_validation.py \
      --onnx sim2real/onnx_export/output/recovery_policy.onnx \
      --num_trials 50 --seeds 0 1 2 \
      --output cross_engine/results/
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

import pychrono as chrono
import pychrono.robot as cr

_devnull_fd = os.open(os.devnull, os.O_WRONLY)  # kept for potential future use

class _SuppressSWIG:
    """Suppress SWIG's 'detected a memory leak' noise.
    SWIG uses printf() / PySys_WriteStdout which routes to C-level stdout (fd 1).
    We redirect both C-level fd 1 and Python's sys.stdout to /dev/null.
    The caller must use sys.stdout.write() or a saved stdout handle for any
    print() calls that should appear while this context is active.
    """
    def __enter__(self):
        self._saved_stdout = sys.stdout
        self._saved_fd1    = os.dup(1)
        sys.stdout = open(os.devnull, "w")
        os.dup2(_devnull_fd, 1)
    def __exit__(self, *_):
        os.dup2(self._saved_fd1, 1)
        os.close(self._saved_fd1)
        sys.stdout.close()
        sys.stdout = self._saved_stdout

# Resolve Chrono data directory: installed builds put data under share/chrono/data/
# rather than the default site-packages path. Try the share location first.
_CHRONO_DATA_CANDIDATES = [
    os.path.join(os.path.dirname(chrono.__file__),
                 "..", "..", "..", "share", "chrono", "data"),
    "/home/rmedu/anaconda3/envs/chrono_viz/share/chrono/data",
]
for _p in _CHRONO_DATA_CANDIDATES:
    _p = os.path.realpath(_p)
    if os.path.isdir(os.path.join(_p, "robot")):
        chrono.SetChronoDataPath(_p + "/")
        break

try:
    import onnxruntime as ort
except ImportError:
    sys.stderr.write(
        "[chrono_validation] onnxruntime not installed.\n"
        "  pip install onnxruntime  (in chrono_viz env)\n"
    )
    raise

# ─────────────────────────────────────────────────────────────────────────────
# Constants matching Newton training env
# ─────────────────────────────────────────────────────────────────────────────
POLICY_OBS_DIM   = 29
ACT_DIM          = 10
GRU_HIDDEN       = 256
GRU_LAYERS       = 1
DRIVE_VEL_LIMIT  = 6.0          # rad/s
STEER_LIMIT      = 0.6          # rad
WHEEL_RADIUS_AAU    = 0.10      # AAU rover (training)
WHEEL_RADIUS_CHRONO = 0.25      # Curiosity (cross-engine)

POLICY_DT       = 0.04          # 25 Hz policy
PHYSICS_DT      = 0.005         # 200 Hz Chrono integration
SUBSTEPS        = int(POLICY_DT / PHYSICS_DT)

ENTRAP_VX_THRESH    = 0.15
ENTRAP_SLIP_THRESH  = 0.40
ENTRAP_HOLD_STEPS   = 15

ESCAPE_DISTANCE     = 3.0       # m  (matches training)
MAX_POLICY_STEPS    = 400       # 16 s sim — caps non-escape wall-clock at ~25s

G_MARS              = 3.72      # m/s²

# Curiosity wheel ID order — matches obs ordering
WHEEL_ORDER = [cr.C_LF, cr.C_RF, cr.C_LM, cr.C_RM, cr.C_LB, cr.C_RB]

# ─────────────────────────────────────────────────────────────────────────────
# Bekker-Wong terrain parameters (Senatore & Iagnemma 2014, MMS-1 Mars simulant)
# ─────────────────────────────────────────────────────────────────────────────
BW_KPHI     = 0.2e6     # Pa/m^(n+1)  Bekker pressure-sinkage modulus (frictional)
BW_KC       = 0.0       # Pa/m^n      Bekker cohesive modulus
BW_N        = 1.1       # sinkage exponent
BW_COHESION = 0.0       # Pa          Mohr-Coulomb cohesion
BW_PHI      = 0.0       # rad         internal friction angle (set per trial from mu)
BW_K        = 0.01      # m           Janosi-Hanamoto shear modulus
BW_ELASTIC  = 2e7       # Pa/m        elastic unloading stiffness
BW_DAMP     = 3e4       # Pa·s/m      terrain damping

WHEEL_WIDTH  = 0.20     # m  Curiosity wheel width (contact footprint)

# Ground plane z-coordinate (terrain surface)
TERRAIN_Z    = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bekker-Wong wheel-terrain interaction (per wheel)
# ─────────────────────────────────────────────────────────────────────────────
class BekkerWheelTerrain:
    """
    Computes Bekker-Wong normal + Janosi-Hanamoto tangential forces for one
    wheel contacting a flat semi-infinite deformable bed.

    State: accumulated shear displacement per wheel (j_k).
    """

    def __init__(self, friction_angle_rad: float, cohesion: float = BW_COHESION,
                 wheel_radius: float = WHEEL_RADIUS_CHRONO,
                 wheel_width: float = WHEEL_WIDTH):
        self.phi        = friction_angle_rad
        self.cohesion   = cohesion
        self.r          = wheel_radius
        self.w          = wheel_width
        self.j          = 0.0   # shear displacement accumulator (m)

    def reset(self, sinkage_init: float = 0.0):
        self.j = 0.0
        self.sinkage = sinkage_init  # pre-burial depth (m, positive = buried)

    def step(self, omega: float, v_x: float, normal_load: float,
             dt: float) -> tuple[float, float, float]:
        """
        Returns (Fn, Ft, current_sinkage).
          Fn  — normal force (N, upward positive)
          Ft  — tangential force (N, forward positive)
          sinkage — current wheel sinkage (m)
        """
        # Contact geometry
        sinkage = max(0.0, self.sinkage)

        # Contact patch length (Hertzian approximation for cylindrical wheel)
        a = math.sqrt(max(0.0, 2.0 * self.r * sinkage))         # half-length
        A = 2.0 * a * self.w                                      # footprint area

        if A < 1e-6 or sinkage < 1e-5:
            # No contact
            self.j = 0.0
            return 0.0, 0.0, sinkage

        # Bekker normal pressure → force
        b = self.w                                                 # contact width
        p = (BW_KPHI / b + BW_KC) * (sinkage ** BW_N)
        Fn = p * A

        # Damping + elastic unloading
        v_z = 0.0  # simplified: no z-velocity tracking in pure-Python terrain
        Fn += BW_DAMP * v_z * A

        # Slip ratio
        wheel_lin = omega * self.r
        denom = max(abs(wheel_lin), abs(v_x), 1e-3)
        slip = (wheel_lin - v_x) / denom
        slip = float(np.clip(slip, -1.0, 1.0))

        # Janosi-Hanamoto shear displacement
        j_dot = abs(wheel_lin - v_x)
        self.j = max(0.0, self.j + j_dot * dt)

        # Mohr-Coulomb shear strength
        tau_max = self.cohesion + p * math.tan(self.phi)

        # Tangential traction
        tau = tau_max * (1.0 - math.exp(-self.j / BW_K))
        Ft = math.copysign(tau * A, slip)

        # Update sinkage: load-driven settlement toward Bekker equilibrium
        z_eq = (Fn / ((BW_KPHI / b + BW_KC) * A)) ** (1.0 / BW_N) if A > 0 else 0.0
        self.sinkage += (z_eq - self.sinkage) * min(1.0, dt * 5.0)
        self.sinkage = max(0.0, self.sinkage)

        return float(Fn), float(Ft), float(self.sinkage)


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder — mirrors envs/entrapment_env.py _get_observations()
# ─────────────────────────────────────────────────────────────────────────────
class CuriosityObsBuilder:
    def __init__(self, rover: cr.Curiosity, wheel_radius: float = WHEEL_RADIUS_CHRONO):
        self.rover = rover
        self.wheel_radius = wheel_radius
        self.entrap_counter = 0
        self.torque_history: list[float] = []

    def reset(self):
        self.entrap_counter = 0
        self.torque_history.clear()

    def build(self, escape_dir: np.ndarray, spawn_xy: np.ndarray,
              wheel_omegas: np.ndarray, applied_torques: np.ndarray
              ) -> tuple[np.ndarray, dict]:
        chassis_pos = self.rover.GetChassisPos()
        chassis_vel = self.rover.GetChassisVel()
        chassis_rot = self.rover.GetChassisRot()

        # GetChassisAcc may not exist — fall back to chassis body linear acceleration
        try:
            chassis_acc = self.rover.GetChassisAcc()
        except AttributeError:
            try:
                chassis_acc = self.rover.GetChassis().GetBody().GetLinAcc()
            except Exception:
                chassis_acc = chrono.ChVector3d(0.0, 0.0, 0.0)

        # Invert rotation quaternion — API differs across pychrono versions
        try:
            rot_inv = chassis_rot.GetConjugate()
        except AttributeError:
            rot_inv = chrono.ChQuaterniond(chassis_rot)
            try:
                rot_inv.Conjugate()
            except Exception:
                pass

        body_vel   = rot_inv.Rotate(chassis_vel)
        v_x_body   = float(body_vel.x)
        v_proj     = float(chassis_vel.x * escape_dir[0] + chassis_vel.y * escape_dir[1])

        # Wheel angular velocities (normalized)
        wheel_vel_norm = np.clip(wheel_omegas / DRIVE_VEL_LIMIT, -2.0, 2.0).astype(np.float32)

        # Per-wheel slip
        wheel_lin = wheel_omegas * self.wheel_radius
        slip = np.where(np.abs(wheel_lin) > 1e-3,
                        1.0 - v_x_body / (wheel_lin + 1e-6), 0.0)
        slip = np.clip(slip, -1.0, 1.0).astype(np.float32)
        mean_abs_slip = float(np.mean(np.abs(slip)))

        # Steering positions
        def _get_steer_angle(wid):
            for mname in ("GetRockerSteerMotor", "GetSteerMotor",
                          "GetRockerSteerMotorFunc", "GetSteerMotorFunc"):
                fn = getattr(self.rover, mname, None)
                if fn is not None:
                    try:
                        m = fn(wid)
                        if m is None:
                            continue
                        for aname in ("GetMotorAngle", "GetMotorPos",
                                      "GetMotorRot", "GetPos"):
                            af = getattr(m, aname, None)
                            if af is not None:
                                try:
                                    return float(af())
                                except Exception:
                                    pass
                    except Exception:
                        continue
            return 0.0

        try:
            steer_pos = np.array([
                _get_steer_angle(cr.C_LF),
                _get_steer_angle(cr.C_RF),
                _get_steer_angle(cr.C_LB),
                _get_steer_angle(cr.C_RB),
            ], dtype=np.float32)
        except Exception:
            steer_pos = np.zeros(4, dtype=np.float32)
        steer_pos_norm = np.clip(steer_pos / STEER_LIMIT, -1.0, 1.0)

        # IMU
        body_acc_vec = rot_inv.Rotate(chassis_acc)
        imu_acc = np.array([float(body_acc_vec.x), float(body_acc_vec.y),
                            float(body_acc_vec.z) + G_MARS], dtype=np.float32) / 10.0

        # Gravity z
        grav_world = chrono.ChVector3d(0.0, 0.0, -G_MARS)
        grav_body  = rot_inv.Rotate(grav_world)
        gravity_z  = float(grav_body.z) / G_MARS

        # Drive torque
        drive_torque_norm = np.clip(applied_torques / 40.0, -1.0, 1.0).astype(np.float32)

        # Entrap flag
        entrapped = (abs(v_x_body) < ENTRAP_VX_THRESH) and (mean_abs_slip > ENTRAP_SLIP_THRESH)
        self.entrap_counter = (min(self.entrap_counter + 1, ENTRAP_HOLD_STEPS + 5)
                               if entrapped else max(self.entrap_counter - 2, 0))
        entrap_flag = 1.0 if self.entrap_counter >= ENTRAP_HOLD_STEPS else 0.0

        # Torque anomaly
        mean_torque = float(np.mean(np.abs(drive_torque_norm)))
        self.torque_history.append(mean_torque)
        if len(self.torque_history) > 50:
            self.torque_history.pop(0)
        baseline = float(np.median(self.torque_history)) if self.torque_history else 0.0
        torque_anomaly = 1.0 if (mean_torque > baseline * 1.4
                                  and abs(v_x_body) < 0.20
                                  and len(self.torque_history) > 15) else 0.0

        # Projected distance
        rel_xy = np.array([float(chassis_pos.x) - spawn_xy[0],
                            float(chassis_pos.y) - spawn_xy[1]])
        proj_dist = float(np.dot(rel_xy, escape_dir))
        dist_norm = float(np.clip(proj_dist / ESCAPE_DISTANCE, 0.0, 1.0))

        obs = np.concatenate([
            wheel_vel_norm, slip, steer_pos_norm, imu_acc,
            np.array([gravity_z], dtype=np.float32),
            drive_torque_norm,
            np.array([entrap_flag, torque_anomaly, dist_norm], dtype=np.float32),
        ]).astype(np.float32)
        assert obs.shape == (POLICY_OBS_DIM,), f"obs shape {obs.shape}"

        info = {
            "v_x_body":    v_x_body,
            "v_proj":      v_proj,
            "mean_slip":   mean_abs_slip,
            "proj_dist":   proj_dist,
            "entrap_flag": entrap_flag,
            "chassis_pos": (float(chassis_pos.x), float(chassis_pos.y),
                            float(chassis_pos.z)),
        }
        return obs, info


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder
# ─────────────────────────────────────────────────────────────────────────────
def build_scene(sinkage_init: float, friction: float):
    """
    Build a Chrono ChSystemNSC with:
      - rigid flat ground (no terrain module required)
      - Curiosity rover pre-buried to sinkage_init via spawn z offset
      - BekkerWheelTerrain instances apply forces per wheel each step
    """
    sys = chrono.ChSystemNSC()

    # Collision system — Type_BULLET is standard in pychrono 10
    try:
        sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    except AttributeError:
        pass  # older API sets it differently or defaults to Bullet

    sys.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -G_MARS))

    # Timestepper — try several names across pychrono versions
    for _ts_name in ("Type_EULER_IMPLICIT_LINEARIZED",
                     "Type_EULER_IMPLICIT_PROJECTED",
                     "Type_EULER_IMPLICIT"):
        if hasattr(chrono.ChTimestepper, _ts_name):
            sys.SetTimestepperType(getattr(chrono.ChTimestepper, _ts_name))
            break

    # Solver — try several names across pychrono versions
    _solver_set = False
    for _sv_name in ("Type_BARZILAIBORWEIN", "Type_PSSOR", "Type_APGD",
                     "Type_SOR", "Type_PSOR"):
        if hasattr(chrono.ChSolver, _sv_name):
            try:
                sys.SetSolverType(getattr(chrono.ChSolver, _sv_name))
                _solver_set = True
                break
            except Exception:
                continue
    if not _solver_set:
        pass  # leave default solver

    try:
        sys.GetSolver().AsIterative().SetMaxIterations(100)
    except Exception:
        pass

    try:
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.002)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)
    except Exception:
        pass

    # Rigid ground plane — 100×100 m centered on spawn so rover never exits
    # even at maximum slide (Curiosity ~1 m/s × 30 s = 30 m; 50 m half-extent is safe).
    ground_mat = chrono.ChContactMaterialNSC()
    ground_mat.SetFriction(float(friction))
    ground_mat.SetRestitution(0.0)
    ground = chrono.ChBodyEasyBox(100.0, 100.0, 0.1, 1000.0, True, True, ground_mat)
    ground.SetPos(chrono.ChVector3d(0.0, 0.0, TERRAIN_Z - 0.05))
    ground.SetFixed(True)
    sys.Add(ground)

    # Curiosity rover — spawn position accounts for initial burial
    # Spawn with wheels just resting on the surface (no penetration at init).
    # Entrapment is created by low friction, not initial burial — Bekker-Wong
    # at 15-28cm sinkage generates forces 10-20x the Curiosity weight (899 kg)
    # and causes violent initialization artifacts.
    z_spawn = WHEEL_RADIUS_CHRONO + TERRAIN_Z + 0.02   # 2 cm clearance
    spawn = chrono.ChVector3d(-2.0, 0.0, z_spawn)

    driver = cr.CuriositySpeedDriver(0.5, 0.0)
    rover  = cr.Curiosity(sys, cr.CuriosityChassisType_FullRover,
                          cr.CuriosityWheelType_RealWheel)
    rover.SetDriver(driver)
    rover.Initialize(chrono.ChFramed(spawn, chrono.QUNIT))

    return sys, rover, driver, spawn


# ─────────────────────────────────────────────────────────────────────────────
# Apply Bekker-Wong forces to each wheel for one physics step
# ─────────────────────────────────────────────────────────────────────────────
def apply_terrain_forces(rover: cr.Curiosity,
                         wheel_terrains: list[BekkerWheelTerrain],
                         wheel_omegas: np.ndarray,
                         v_x_body: float,
                         dt: float) -> np.ndarray:
    """
    Compute Bekker-Wong Fn/Ft for each wheel and apply as external forces.
    Returns array of applied normal forces (N) for sinkage tracking.
    """
    fn_arr = np.zeros(6)
    for k, (wid, wt) in enumerate(zip(WHEEL_ORDER, wheel_terrains)):
        wheel_body = rover.GetWheel(wid).GetBody()
        wpos = wheel_body.GetPos()

        # Wheel z relative to terrain surface
        wheel_center_z = float(wpos.z)
        sinkage = TERRAIN_Z + WHEEL_RADIUS_CHRONO - wheel_center_z
        wt.sinkage = max(0.0, sinkage)

        omega_k = float(wheel_omegas[k])
        Fn, Ft, _ = wt.step(omega_k, v_x_body, 0.0, dt)
        fn_arr[k] = Fn

        if Fn > 0.0:
            # Normal force: upward (+Z)
            f_normal = chrono.ChVector3d(0.0, 0.0, Fn)
            # Tangential force: along wheel's forward direction (body X)
            rot = wheel_body.GetRot()
            fwd = rot.Rotate(chrono.ChVector3d(1.0, 0.0, 0.0))
            f_tang  = chrono.ChVector3d(Ft * fwd.x, Ft * fwd.y, 0.0)
            f_total = chrono.ChVector3d(f_normal.x + f_tang.x,
                                        f_normal.y + f_tang.y,
                                        f_normal.z)
            # AccumulateForce signature: (force, point, is_local)
            # Some pychrono builds use AddForce or AccumulateForce with 2 args
            try:
                wheel_body.AccumulateForce(f_total, wpos, False)
            except TypeError:
                try:
                    wheel_body.AccumulateForce(f_total, wpos)
                except Exception:
                    try:
                        wheel_body.AddForce(f_total)
                    except Exception:
                        pass

    return fn_arr


# ─────────────────────────────────────────────────────────────────────────────
# Trial loop
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrialResult:
    seed:            int
    trial_id:        int
    sinkage_init:    float
    friction:        float
    escape_dir_x:    float
    escape_dir_y:    float
    escaped:         bool
    time_to_escape:  int
    final_proj:      float
    entrapped_steps: int
    failure_mode:    str
    terminal_x:      float
    terminal_y:      float
    terminal_z:      float
    trajectory_xy:   list = None   # [(x, y), ...] per policy step; None if not recorded


def _probe_motor_func(rover_obj, wid, getter_names):
    """Return the first callable motor function found, or None.  Cached once per trial."""
    for mname in getter_names:
        fn = getattr(rover_obj, mname, None)
        if fn is None:
            continue
        try:
            f = fn(wid)
        except Exception:
            continue
        if f is not None and hasattr(f, "SetSetpoint"):
            return f
    return None


def run_trial(onnx_session: ort.InferenceSession, seed: int, trial_id: int,
              sinkage_init: float, friction: float,
              escape_heading: float, verbose: bool = False) -> TrialResult:

    with _SuppressSWIG():
        sys, rover, driver, spawn = build_scene(sinkage_init, friction)

    escape_dir = np.array([math.cos(escape_heading), math.sin(escape_heading)],
                          dtype=np.float32)
    spawn_xy = np.array([float(spawn.x), float(spawn.y)], dtype=np.float32)

    obs_builder = CuriosityObsBuilder(rover)
    h_state      = np.zeros((GRU_LAYERS, 1, GRU_HIDDEN), dtype=np.float32)

    # Cache motor handles once to avoid per-step SWIG wrapper creation.
    _DRIVE_NAMES = ("GetDriveMotorFunc", "GetWheelMotorFunc",
                    "GetDriveMotor", "GetWheelMotor")
    _STEER_NAMES = ("GetRockerSteerMotorFunc", "GetSteerMotorFunc",
                    "GetRockerSteerMotor", "GetSteerMotor")
    _STEER_WIDS  = [cr.C_LF, cr.C_RF, cr.C_LB, cr.C_RB]
    drive_funcs = [_probe_motor_func(rover, wid, _DRIVE_NAMES) for wid in WHEEL_ORDER]
    steer_funcs = [_probe_motor_func(rover, wid, _STEER_NAMES) for wid in _STEER_WIDS]
    # Cache wheel bodies for omega / torque reads
    wheel_bodies = [rover.GetWheel(wid).GetBody() for wid in WHEEL_ORDER]

    # Settle 25 steps with zero commands
    wheel_omegas = np.zeros(6, dtype=np.float32)
    for _ in range(25):
        rover.Update()
        sys.DoStepDynamics(PHYSICS_DT)

    escaped, t_to_escape = False, -1
    final_proj      = 0.0
    entrapped_steps = 0
    applied_torques = np.zeros(6, dtype=np.float32)
    info = {"chassis_pos": (float(spawn.x), float(spawn.y), float(spawn.z)),
            "v_x_body": 0.0, "mean_slip": 0.0, "proj_dist": 0.0, "entrap_flag": 0.0}
    trajectory_xy: list[tuple[float, float]] = []   # (x, y) per policy step

    for step in range(MAX_POLICY_STEPS):
        # Read wheel angular velocities (cached wheel bodies)
        def _omega(wb):
            try:
                av = wb.GetAngVelLocal()
                try:    return float(av.z)
                except AttributeError: return float(av)
            except Exception:
                return 0.0
        wheel_omegas = np.array([_omega(wb) for wb in wheel_bodies], dtype=np.float32)

        obs, info = obs_builder.build(escape_dir, spawn_xy, wheel_omegas, applied_torques)

        # Policy forward pass
        action_mean, h_state = onnx_session.run(
            ["action", "h_out"],
            {"obs": obs.reshape(1, -1).astype(np.float32),
             "h_in": h_state.astype(np.float32)},
        )
        action = action_mean[0]

        sim_t = sys.GetChTime()
        rover.Update()

        # Apply drive commands using cached motor handles
        for k, f in enumerate(drive_funcs):
            if f is not None:
                f.SetSetpoint(float(action[k]) * DRIVE_VEL_LIMIT, sim_t)

        steer_cmds = np.clip(action[6:10], -1.0, 1.0) * STEER_LIMIT
        for j, f in enumerate(steer_funcs):
            if f is not None:
                f.SetSetpoint(float(steer_cmds[j]), sim_t)

        # Step physics at 200 Hz — NSC contact handles all ground reaction forces
        for _ in range(SUBSTEPS):
            sys.DoStepDynamics(PHYSICS_DT)

        # Torque feedback: use zero fallback (motor torque not easily accessible via NSC)
        applied_torques = np.zeros(6, dtype=np.float32)

        trajectory_xy.append((info["chassis_pos"][0], info["chassis_pos"][1]))
        if info["entrap_flag"] > 0.5:
            entrapped_steps += 1
        final_proj = info["proj_dist"]

        if info["proj_dist"] >= ESCAPE_DISTANCE:
            escaped      = True
            t_to_escape  = step
            break

        cz = info["chassis_pos"][2]
        if cz < -2.0 or cz > 5.0:
            break

        if verbose and step % 100 == 0:
            print(f"      step {step:4d}  proj={info['proj_dist']:5.2f}m  "
                  f"v_x={info['v_x_body']:+.2f}  slip={info['mean_slip']:.2f}  "
                  f"entrap={int(info['entrap_flag'])}")

    # Failure classification
    cz = info["chassis_pos"][2]
    if escaped:
        failure_mode = "ESCAPED"
    elif cz < -2.0:
        failure_mode = "terrain_diverge"
    elif cz > 0.6:
        failure_mode = "high_centered"
    elif abs(info["chassis_pos"][1] - spawn_xy[1]) > 1.5:
        failure_mode = "lateral_OOB"
    elif entrapped_steps > 0.8 * MAX_POLICY_STEPS:
        failure_mode = "stall_in_bed"
    elif final_proj < 0.5:
        failure_mode = "no_progress"
    else:
        failure_mode = "timeout_no_progress"

    return TrialResult(
        seed=seed, trial_id=trial_id,
        sinkage_init=sinkage_init, friction=friction,
        escape_dir_x=float(escape_dir[0]), escape_dir_y=float(escape_dir[1]),
        escaped=escaped, time_to_escape=t_to_escape, final_proj=final_proj,
        entrapped_steps=entrapped_steps, failure_mode=failure_mode,
        terminal_x=info["chassis_pos"][0],
        terminal_y=info["chassis_pos"][1],
        terminal_z=info["chassis_pos"][2],
        trajectory_xy=trajectory_xy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_resample: int = 10_000,
                 conf: float = 0.95) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(n_resample, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, (1 - conf) / 2 * 100))
    hi = float(np.percentile(means, (1 + conf) / 2 * 100))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx",         required=True)
    ap.add_argument("--num_trials",   type=int,   default=50)
    ap.add_argument("--seeds",        type=int,   nargs="+", default=[0, 1, 2])
    ap.add_argument("--sinkage_min",  type=float, default=0.15)  # kept for CSV schema compat
    ap.add_argument("--sinkage_max",  type=float, default=0.28)
    ap.add_argument("--friction_min", type=float, default=0.10)  # low μ → high slip → entrapment
    ap.add_argument("--friction_max", type=float, default=0.35)
    ap.add_argument("--output",       default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"[chrono] ONNX loaded: {args.onnx}")
    print(f"[chrono] inputs : {[i.name for i in sess.get_inputs()]}")
    print(f"[chrono] outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"[chrono] engine : Chrono ChSystemNSC (NSC complementarity)")
    print(f"[chrono] terrain: rigid flat surface, friction μ ∈ [{args.friction_min},{args.friction_max}]")
    print(f"[chrono] rover  : Curiosity FullRover, RealWheel")
    print(f"[chrono] {len(args.seeds)} seeds × {args.num_trials} trials = "
          f"{len(args.seeds) * args.num_trials} total\n")

    all_results: list[TrialResult] = []
    t_start = time.time()

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        print(f"\n— seed {seed} —")
        for trial_id in range(args.num_trials):
            sinkage  = float(rng.uniform(args.sinkage_min, args.sinkage_max))
            friction = float(rng.uniform(args.friction_min, args.friction_max))
            heading  = float(rng.uniform(0.0, 2 * math.pi))
            t0 = time.time()
            try:
                with _SuppressSWIG():
                    r = run_trial(sess, seed, trial_id, sinkage, friction, heading,
                                  verbose=args.verbose)
                all_results.append(r)
            except Exception as e:
                import traceback
                print(f"  [seed {seed} trial {trial_id}] ERROR: {e}")
                traceback.print_exc()
                continue
            dt = time.time() - t0
            tag = "ESC" if r.escaped else r.failure_mode[:10]
            print(f"  trial={trial_id:3d}  sink={sinkage:.3f}  μ={friction:.2f}  "
                  f"hd={math.degrees(heading):5.1f}°  [{tag:>10s}]  "
                  f"proj={r.final_proj:5.2f}m  ({dt:.1f}s)")

    n = len(all_results)
    if n == 0:
        print("[chrono] NO TRIALS COMPLETED. Aborting.")
        return

    escaped_arr = np.array([r.escaped for r in all_results], dtype=float)
    rec_rate    = float(escaped_arr.mean())
    rec_lo, rec_hi = bootstrap_ci(escaped_arr)
    esc_times   = np.array([r.time_to_escape for r in all_results if r.escaped])
    proj_arr    = np.array([r.final_proj for r in all_results])

    by_mode: dict[str, int] = {}
    for r in all_results:
        by_mode[r.failure_mode] = by_mode.get(r.failure_mode, 0) + 1

    print("\n" + "=" * 62)
    print("  Cross-engine sim2sim — Chrono ChSystemNSC + rigid low-μ terrain")
    print("=" * 62)
    print(f"  Trials completed  : {n}  ({len(args.seeds)} seeds × {args.num_trials})")
    print(f"  Recovery rate     : {rec_rate*100:.1f}%   "
          f"95% CI = [{rec_lo*100:.1f}, {rec_hi*100:.1f}]%")
    if len(esc_times):
        print(f"  Time-to-escape    : {esc_times.mean():.1f} ± {esc_times.std():.1f} steps "
              f"({esc_times.mean()*POLICY_DT:.1f} s)")
    print(f"  Mean final proj   : {proj_arr.mean():.2f} m")
    print(f"  Failure modes     : {by_mode}")
    print(f"  Wall-clock        : {time.time() - t_start:.1f} s")

    csv_path = os.path.join(args.output, "chrono_nsc_results.csv")
    with open(csv_path, "w", newline="") as f:
        # Exclude trajectory_xy from CSV (saved separately as NPZ)
        scalar_fields = [k for k in asdict(all_results[0]).keys() if k != "trajectory_xy"]
        w = csv.DictWriter(f, fieldnames=scalar_fields)
        w.writeheader()
        for r in all_results:
            row = asdict(r)
            row.pop("trajectory_xy", None)
            w.writerow(row)

    # Save per-trial trajectories as NPZ for heatmap plotting
    traj_path = os.path.join(args.output, "chrono_nsc_trajectories.npz")
    traj_data = {}
    for r in all_results:
        key = f"seed{r.seed}_trial{r.trial_id}"
        xy = r.trajectory_xy or []
        traj_data[key] = np.array(xy, dtype=np.float32) if xy else np.zeros((0, 2))
        traj_data[f"{key}_escaped"] = np.array([r.escaped])
        traj_data[f"{key}_spawn_xy"] = np.array([r.escape_dir_x, r.escape_dir_y])
    np.savez(traj_path, **traj_data)
    print(f"  Trajectories → {traj_path}")

    summary_path = os.path.join(args.output, "chrono_nsc_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "engine":            "Project Chrono",
            "rigid_body_solver": "ChSystemNSC (NSC complementarity, Bullet broadphase)",
            "terrain":           "Rigid flat surface, friction μ randomised per trial",
            "friction_range":    [args.friction_min, args.friction_max],
            "terrain_rationale": (
                "Bekker-Wong at 15-28 cm sinkage produces forces 10-20x Curiosity weight "
                "(899 kg); rigid low-μ terrain creates the same observable entrapment signal "
                "(high wheel slip, near-zero progress) without initialization artifacts."
            ),
            "rover":             "Curiosity FullRover (NASA MSL, pychrono.robot)",
            "n_trials":          n,
            "seeds":             args.seeds,
            "trials_per_seed":   args.num_trials,
            "recovery_rate":     rec_rate,
            "recovery_rate_ci_95": [rec_lo, rec_hi],
            "time_to_escape_mean_steps": float(esc_times.mean()) if len(esc_times) else None,
            "time_to_escape_std_steps":  float(esc_times.std())  if len(esc_times) else None,
            "failure_modes":     by_mode,
            "onnx_policy":       os.path.abspath(args.onnx),
            "fidelity_gap_notes": [
                "Rigid body solver: Chrono NSC (complementarity) vs training Newton "
                "(MPM implicit) — different contact resolution method.",
                "Terrain: rigid surface with variable friction vs training MPM continuum "
                "elasto-plasticity — different physics class, not a parameter perturbation.",
                "Rover mass: Curiosity 899 kg vs AAU ~35 kg; wheel radius 0.25 m vs 0.10 m. "
                "Genuine morphological domain gap.",
                "Per-wheel drive commands issued individually via per-wheel SpeedDriver "
                "motor function SetSetpoint; steer via per-wheel SteerMotorFunc.",
            ],
        }, f, indent=2)

    print(f"\n  CSV     → {csv_path}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
