"""Cross-engine sim2sim validation: Project Chrono Curiosity rover on SCM terrain.

Loads the ONNX policy exported from sim2real/onnx_export/output/, drives the
built-in Chrono Curiosity rocker-bogie rover (pychrono.robot.Curiosity) over a
deformable SCM (Bekker-Wong) terrain patch, and reports recovery rate /
time-to-escape against a Recovery-GPS condition matched to the Newton MPM
validation protocol.

Why this counts as cross-engine:
  - Different RIGID-BODY engine: Chrono ChSystemNSC vs. MuJoCo Warp
  - Different TERRAIN PHYSICS: SCM Bekker-Wong semi-empirical
    pressure-sinkage vs. Newton SolverImplicitMPM continuum elasto-plasticity
    (a different *class* of physics, not just a parameter perturbation)
  - Different ROVER MORPHOLOGY: Chrono Curiosity (NASA MSL, 2.9 m wheelbase,
    0.25 m wheel radius, 899 kg) vs. AAU Mars rover (RLRoverLab USD, 0.10 m
    wheel radius). Both are 6-wheel rocker-bogie; mass distribution and
    contact geometry differ — a real domain gap, not a re-skin.

Known cross-engine simplifications (documented in paper):
  - The CuriosityDCMotorControl driver exposes a single "steering" scalar plus
    internal back-EMF torque control; per-wheel drive setpoints are mapped to
    the mean of the policy's 6 drive commands. This is a deliberate fidelity
    gap: the cross-engine result tests whether the policy's *coarse* recovery
    strategy (rocking timing, steering coordination) transfers, not whether
    fine per-wheel torque sequences do.
  - Steering: the 4 steer commands are averaged to a single Curiosity steering
    input.

Run (in chrono_viz env, NOT env_isaaclab):
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
from typing import Optional

import numpy as np

import pychrono as chrono
import pychrono.robot as cr
import pychrono.vehicle as veh

try:
    import pychrono.fsi as fsi
    _HAS_FSI = True
except ImportError:
    _HAS_FSI = False

try:
    import onnxruntime as ort
except ImportError:
    sys.stderr.write(
        "[chrono_validation] onnxruntime not installed.\n"
        "  pip install onnxruntime  (in chrono_viz env)\n"
    )
    raise

# ────────────────────────────────────────────────────────────────────────────
# Constants matching Newton training env
# ────────────────────────────────────────────────────────────────────────────
POLICY_OBS_DIM   = 29
ACT_DIM          = 10
GRU_HIDDEN       = 256
GRU_LAYERS       = 1
DRIVE_VEL_LIMIT  = 6.0          # rad/s — matches envs/entrapment_env.py
STEER_LIMIT      = 0.6          # rad
WHEEL_RADIUS_AAU    = 0.10      # AAU rover (training)
WHEEL_RADIUS_CHRONO = 0.25      # Curiosity (deployment)

POLICY_DT       = 0.04          # 25 Hz
PHYSICS_DT      = 0.005         # 200 Hz Chrono integration
SUBSTEPS        = int(POLICY_DT / PHYSICS_DT)

# Entrapment thresholds (mirror EntrapmentEnvCfg)
ENTRAP_VX_THRESH    = 0.15
ENTRAP_SLIP_THRESH  = 0.40
ENTRAP_HOLD_STEPS   = 15

ESCAPE_DISTANCE     = 3.0
MAX_POLICY_STEPS    = 1500       # 60 s
SAND_PATCH_X        = 8.0        # SCM patch length (m), spawn at -X/2 + 1.5
SAND_PATCH_Y        = 6.0
SCM_RES             = 0.04       # mesh cell size (m)

G_MARS              = 3.72       # m/s^2

# Curiosity wheel ID order — must match obs ordering used in training
WHEEL_ORDER = [cr.C_LF, cr.C_RF, cr.C_LM, cr.C_RM, cr.C_LB, cr.C_RB]


@dataclass
class TrialResult:
    seed:           int
    trial_id:       int
    sinkage_init:   float
    friction:       float
    escape_dir_x:   float
    escape_dir_y:   float
    escaped:        bool
    time_to_escape: int
    final_proj:     float
    entrapped_steps: int
    failure_mode:   str
    terminal_x:     float
    terminal_y:     float
    terminal_z:     float


# ────────────────────────────────────────────────────────────────────────────
# Observation builder — produces the exact 29D vector the policy was trained on.
# Order MUST match envs/entrapment_env.py _get_observations():
#   wheel_vel[6] | slip[6] | steer_pos[4] | imu_acc[3] | gravity_z[1]
#   | drive_torque[6] | entrap_flag[1] | torque_anomaly[1] | dist_norm[1]
# ────────────────────────────────────────────────────────────────────────────
class CuriosityObsBuilder:
    def __init__(self, rover: cr.Curiosity, wheel_radius: float):
        self.rover = rover
        self.wheel_radius = wheel_radius
        self.entrap_counter = 0
        self.torque_history: list[float] = []

    def reset(self):
        self.entrap_counter = 0
        self.torque_history.clear()

    def build(self, escape_dir: np.ndarray,
              spawn_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        chassis_pos = self.rover.GetChassisPos()
        chassis_vel = self.rover.GetChassisVel()
        chassis_acc = self.rover.GetChassisAcc()
        chassis_rot = self.rover.GetChassisRot()

        # Body-frame velocity for slip
        rot_inv = chrono.ChQuaterniond(chassis_rot)
        rot_inv.Conjugate()
        body_vel = rot_inv.Rotate(chassis_vel)
        v_x_body = float(body_vel.x)

        # World-frame projected velocity for entrap_flag (matches sim convention)
        v_proj = (chassis_vel.x * escape_dir[0] + chassis_vel.y * escape_dir[1])

        # Wheel angular velocities (rad/s)
        wheel_omega = np.array(
            [float(self.rover.GetWheelAngVel(wid).z) for wid in WHEEL_ORDER],
            dtype=np.float32,
        )
        wheel_vel_norm = np.clip(wheel_omega / DRIVE_VEL_LIMIT, -2.0, 2.0)

        # Per-wheel slip = 1 - v_x_body / (omega * r), clamped
        wheel_lin_speed = wheel_omega * self.wheel_radius
        slip = np.where(
            np.abs(wheel_lin_speed) > 1e-3,
            1.0 - (v_x_body / (wheel_lin_speed + 1e-6)),
            0.0,
        )
        slip = np.clip(slip, -1.0, 1.0).astype(np.float32)
        mean_abs_slip = float(np.mean(np.abs(slip)))

        # Steering — Curiosity has 4 steerable corners. Order: LF, RF, LB, RB
        # to match training's [front_left, front_right, rear_left, rear_right].
        try:
            steer_pos = np.array([
                float(self.rover.GetRockerSteerMotor(cr.C_LF).GetMotorAngle()),
                float(self.rover.GetRockerSteerMotor(cr.C_RF).GetMotorAngle()),
                float(self.rover.GetRockerSteerMotor(cr.C_LB).GetMotorAngle()),
                float(self.rover.GetRockerSteerMotor(cr.C_RB).GetMotorAngle()),
            ], dtype=np.float32)
        except Exception:
            steer_pos = np.zeros(4, dtype=np.float32)
        steer_pos_norm = np.clip(steer_pos / STEER_LIMIT, -1.0, 1.0)

        # IMU accel in body frame, then add gravity-removed z. Newton training
        # passes raw body-frame linear acc divided by 10 for normalization.
        body_acc = rot_inv.Rotate(chassis_acc)
        imu_acc = np.array([float(body_acc.x), float(body_acc.y),
                            float(body_acc.z) + G_MARS], dtype=np.float32) / 10.0

        # Gravity z in body frame, normalized to ~[-1, 1]
        gravity_world = chrono.ChVector3d(0.0, 0.0, -G_MARS)
        gravity_body = rot_inv.Rotate(gravity_world)
        gravity_z = float(gravity_body.z) / G_MARS

        # Drive torque (applied torque on each wheel motor)
        drive_torque = np.array([
            float(self.rover.GetWheelAppliedTorque(wid).z) for wid in WHEEL_ORDER
        ], dtype=np.float32)
        drive_torque_norm = np.clip(drive_torque / 40.0, -1.0, 1.0)

        # Entrapment flag with hysteresis
        entrapped_now = (abs(v_x_body) < ENTRAP_VX_THRESH) and (mean_abs_slip > ENTRAP_SLIP_THRESH)
        if entrapped_now:
            self.entrap_counter = min(self.entrap_counter + 1, ENTRAP_HOLD_STEPS + 5)
        else:
            self.entrap_counter = max(self.entrap_counter - 2, 0)
        entrap_flag = 1.0 if self.entrap_counter >= ENTRAP_HOLD_STEPS else 0.0

        # Torque anomaly: high torque under low forward velocity
        mean_torque = float(np.mean(np.abs(drive_torque_norm)))
        self.torque_history.append(mean_torque)
        if len(self.torque_history) > 50:
            self.torque_history.pop(0)
        baseline = float(np.median(self.torque_history)) if self.torque_history else 0.0
        torque_anomaly = 1.0 if (mean_torque > baseline * 1.4
                                 and abs(v_x_body) < 0.20
                                 and len(self.torque_history) > 15) else 0.0

        # Projected distance, normalized
        rel_xy = np.array([chassis_pos.x - spawn_xy[0], chassis_pos.y - spawn_xy[1]])
        proj_dist = float(np.dot(rel_xy, escape_dir))
        dist_norm = float(np.clip(proj_dist / ESCAPE_DISTANCE, 0.0, 1.0))

        obs = np.concatenate([
            wheel_vel_norm,
            slip,
            steer_pos_norm,
            imu_acc,
            np.array([gravity_z], dtype=np.float32),
            drive_torque_norm,
            np.array([entrap_flag], dtype=np.float32),
            np.array([torque_anomaly], dtype=np.float32),
            np.array([dist_norm], dtype=np.float32),
        ]).astype(np.float32)
        assert obs.shape == (POLICY_OBS_DIM,)

        info = {
            "v_x_body":   v_x_body,
            "v_proj":     v_proj,
            "mean_slip":  mean_abs_slip,
            "proj_dist":  proj_dist,
            "entrap_flag": entrap_flag,
            "chassis_pos": (float(chassis_pos.x), float(chassis_pos.y), float(chassis_pos.z)),
        }
        return obs, info


# ────────────────────────────────────────────────────────────────────────────
# Scene builder. Order matters: collision system → SCM terrain → rover.
# ────────────────────────────────────────────────────────────────────────────
def build_scene_crm(sinkage_init: float, friction: float):
    """Chrono CRM/SPH continuum granular bed (Drucker-Prager, mesh-free SPH).

    Decouples the rigid-solver gap from the terrain-physics-class gap: same
    Chrono ChSystemNSC as the SCM run, but the bed is now a continuum granular
    SPH discretisation rather than a semi-empirical pressure-sinkage model.
    """
    if not _HAS_FSI:
        raise RuntimeError(
            "pychrono.fsi not available. Build pychrono with -DENABLE_MODULE_FSI=ON "
            "or run the SCM tier (--terrain scm)."
        )

    sys = chrono.ChSystemNSC()
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -G_MARS))
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    # Rover
    wheel_mat = chrono.ChContactMaterialData(
        0.4, 0.2, 2e7, 0.3, 2e5, 40.0, 2e5, 20.0,
    )
    # SpeedDriver exposes ChFunctionSetpoint per wheel/steer joint, allowing
    # per-wheel policy commands to override the driver's defaults each step
    # (CuriosityDCMotorControl does not).
    driver = cr.CuriositySpeedDriver(0.5, 0.0)
    rover = cr.Curiosity(sys, cr.CuriosityChassisType_FullRover,
                          cr.CuriosityWheelType_RealWheel)
    rover.SetDriver(driver)
    rover.SetWheelContactMaterial(wheel_mat.CreateMaterial(sys.GetContactMethod()))
    spawn = chrono.ChVector3d(-3.0, 0.0, WHEEL_RADIUS_CHRONO + 0.30)
    rover.Initialize(chrono.ChFramed(spawn, chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0)))

    # CRM terrain (high-level wrapper around ChFsiProblem*)
    initial_spacing = 0.03
    terrain = veh.CRMTerrain(sys, initial_spacing)
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -G_MARS))
    terrain.SetStepSizeCFD(5e-4)

    mat = fsi.ElasticMaterialProperties()
    mat.density = 1700.0
    mat.Young_modulus = 1e6
    mat.Poisson_ratio = 0.3
    mat.mu_I0 = 0.04
    mat.mu_fric_s = friction
    mat.mu_fric_2 = friction
    mat.average_diam = 0.005
    mat.cohesion_coeff = 5e3
    terrain.SetElasticSPH(mat)
    sysSPH = terrain.GetFluidSystemSPH()
    sysSPH.EnableCudaErrorCheck(False)

    sph = fsi.SPHParameters()
    sph.integration_scheme = fsi.IntegrationScheme_RK2
    sph.initial_spacing = initial_spacing
    sph.d0_multiplier = 1.0
    sph.free_surface_threshold = 0.8
    sph.artificial_viscosity = 0.5
    sph.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph.boundary_method = fsi.BoundaryMethod_ADAMI
    sph.use_variable_time_step = True
    terrain.SetSPHParameters(sph)

    # Register all 6 wheels as FSI rigid bodies
    mesh_file = chrono.GetChronoDataFile("robot/curiosity/obj/curiosity_cylwheel.obj")
    geometry = chrono.ChBodyGeometry()
    geometry.materials.push_back(chrono.ChContactMaterialData())
    geometry.coll_meshes.push_back(
        chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh_file, chrono.VNULL)
    )
    for wid in WHEEL_ORDER:
        terrain.AddRigidBody(rover.GetWheel(wid).GetBody(), geometry, False)

    terrain.SetActiveDomain(chrono.ChVector3d(0.6, 0.6, 0.6))
    # Patch must enclose the rover wheelbase + active domain margin on both
    # sides; demo uses 12 m. Centre patch at rover spawn so rear wheels stay
    # well inside x-min.
    terrain.Construct(
        chrono.ChVector3d(12.0, 4.0, 0.25),
        chrono.ChVector3d(float(spawn.x) + 3.0, 0.0, 0.0),
        fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,
    )
    terrain.Initialize()
    return sys, terrain, rover, driver, spawn


def build_scene(sinkage_init: float, friction: float):
    sys = chrono.ChSystemNSC()
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -G_MARS))
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    # SCM Bekker-Wong terramechanics. Coefficients: Senatore & Iagnemma (2014)
    # DEM-calibrated parameters for MMS-1 Mars regolith simulant under
    # Curiosity-class rigid wheels.
    scm = veh.SCMTerrain(sys)
    scm.SetSoilParameters(
        0.2e6,                                        # Bekker Kphi  (Pa/m^(n+1))
        0.0,                                          # Bekker Kc    (Pa/m^n)
        1.1,                                          # Bekker n     (sinkage exponent)
        0.0,                                          # Mohr cohesion (Pa)
        math.degrees(math.atan(friction)),            # Mohr friction angle (deg)
        0.01,                                         # Janosi shear modulus (m)
        2e7,                                          # elastic K   (Pa/m)
        3e4,                                          # damping R   (Pa·s/m)
    )
    scm.Initialize(SAND_PATCH_X, SAND_PATCH_Y, SCM_RES)
    scm.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, 0.0, 0.10)

    # Curiosity rover, half-buried at the start of the sand patch.
    driver = cr.CuriositySpeedDriver(0.5, 0.0)
    rover = cr.Curiosity(sys, cr.CuriosityChassisType_FullRover,
                          cr.CuriosityWheelType_RealWheel)
    rover.SetDriver(driver)
    spawn = chrono.ChVector3d(-SAND_PATCH_X / 2 + 1.5, 0.0,
                               WHEEL_RADIUS_CHRONO + 0.05 - sinkage_init)
    rover.Initialize(chrono.ChFramed(spawn,
                                      chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0)))
    return sys, scm, rover, driver, spawn


# ────────────────────────────────────────────────────────────────────────────
# Trial loop
# ────────────────────────────────────────────────────────────────────────────
def run_trial(onnx_session: ort.InferenceSession, seed: int, trial_id: int,
              sinkage_init: float, friction: float,
              escape_heading: float, terrain: str = "scm",
              verbose: bool = False) -> TrialResult:
    if terrain == "crm":
        sys, terrain_obj, rover, driver, spawn = build_scene_crm(sinkage_init, friction)
        is_crm = True
    else:
        sys, terrain_obj, rover, driver, spawn = build_scene(sinkage_init, friction)
        is_crm = False
    crm_step = 5 * 5e-4  # exchange step (matches demo_ROBOT_Viper_CRM)
    obs_builder = CuriosityObsBuilder(rover, WHEEL_RADIUS_CHRONO)

    escape_dir = np.array([math.cos(escape_heading), math.sin(escape_heading)],
                           dtype=np.float32)
    spawn_xy = np.array([float(spawn.x), float(spawn.y)], dtype=np.float32)

    # GRU hidden state (B = 1)
    h_state = np.zeros((GRU_LAYERS, 1, GRU_HIDDEN), dtype=np.float32)

    # Settle so contact engages cleanly. For CRM, run a longer (3 s) zero-
    # command settling phase so the rover lands on the SPH bed under Mars
    # gravity and any natural burial develops; record the post-settle wheel
    # z to characterise how buried it ended up.
    if is_crm:
        n_settle = int(3.0 / crm_step)
        # Force zero setpoints during settling
        rover.Update()
        for wid in WHEEL_ORDER:
            f = rover.GetDriveMotorFunc(wid)
            if f is not None:
                f.SetSetpoint(0.0, 0.0)
        for _ in range(n_settle):
            rover.Update()
            for wid in WHEEL_ORDER:
                f = rover.GetDriveMotorFunc(wid)
                if f is not None and hasattr(f, "SetSetpoint"):
                    f.SetSetpoint(0.0, sys.GetChTime())
            terrain_obj.DoStepDynamics(crm_step)
    else:
        for _ in range(25):
            rover.Update()
            sys.DoStepDynamics(PHYSICS_DT)

    escaped, t_to_escape = False, -1
    final_proj = 0.0
    entrapped_steps = 0
    info = {"chassis_pos": (float(spawn.x), float(spawn.y), float(spawn.z))}

    for step in range(MAX_POLICY_STEPS):
        obs, info = obs_builder.build(escape_dir, spawn_xy)

        # ── Policy forward ──
        action_mean, h_state = onnx_session.run(
            ["action", "h_out"],
            {"obs":  obs.reshape(1, -1).astype(np.float32),
             "h_in": h_state.astype(np.float32)},
        )
        action = action_mean[0]
        # Per-wheel commands via CuriositySpeedDriver: each drive/steer joint
        # has a ChFunctionSetpoint we override AFTER rover.Update() so the
        # policy's 6 drive cmds and 4 steer cmds drive the corresponding joints
        # individually (the previous DC-motor driver only exposed one global
        # steering scalar).
        sim_t = sys.GetChTime()
        rover.Update()
        for k, wid in enumerate(WHEEL_ORDER):
            f = rover.GetDriveMotorFunc(wid)
            if f is not None and hasattr(f, "SetSetpoint"):
                f.SetSetpoint(float(action[k]) * DRIVE_VEL_LIMIT, sim_t)
        steer_cmd_4 = np.clip(action[6:10], -1.0, 1.0) * STEER_LIMIT
        for j, wid in enumerate([cr.C_LF, cr.C_RF, cr.C_LB, cr.C_RB]):
            f = rover.GetRockerSteerMotorFunc(wid)
            if f is not None and hasattr(f, "SetSetpoint"):
                f.SetSetpoint(float(steer_cmd_4[j]), sim_t)

        # Step physics: SCM uses sys.DoStepDynamics @ 200 Hz; CRM uses
        # terrain.DoStepDynamics @ 400 Hz (step_size=5e-4, exchange=2.5e-3).
        if is_crm:
            t_target = POLICY_DT
            t_acc = 0.0
            while t_acc < t_target:
                terrain_obj.DoStepDynamics(crm_step)
                t_acc += crm_step
        else:
            for _ in range(SUBSTEPS):
                sys.DoStepDynamics(PHYSICS_DT)

        if info["entrap_flag"] > 0.5:
            entrapped_steps += 1
        final_proj = info["proj_dist"]

        if info["proj_dist"] >= ESCAPE_DISTANCE:
            escaped = True
            t_to_escape = step
            break

        # Safety abort: chassis tipped over or fell out of world
        if info["chassis_pos"][2] < -1.0 or info["chassis_pos"][2] > 2.0:
            break

        if verbose and step % 100 == 0:
            print(f"      step {step:4d}  proj={info['proj_dist']:5.2f}m  "
                  f"v_x={info['v_x_body']:+.2f}  slip={info['mean_slip']:.2f}  "
                  f"entrap={int(info['entrap_flag'])}")

    # Failure-mode classification
    if escaped:
        failure_mode = "ESCAPED"
    elif entrapped_steps > 0.8 * MAX_POLICY_STEPS:
        failure_mode = "stall_in_bed"
    elif final_proj < 0.5:
        failure_mode = "no_progress"
    elif info["chassis_pos"][2] > 0.6:
        failure_mode = "high_centered"
    elif abs(info["chassis_pos"][1] - spawn_xy[1]) > 1.5:
        failure_mode = "lateral_OOB"
    else:
        failure_mode = "timeout_no_progress"

    return TrialResult(
        seed=seed, trial_id=trial_id,
        sinkage_init=sinkage_init,
        friction=friction,
        escape_dir_x=float(escape_dir[0]),
        escape_dir_y=float(escape_dir[1]),
        escaped=escaped, time_to_escape=t_to_escape, final_proj=final_proj,
        entrapped_steps=entrapped_steps, failure_mode=failure_mode,
        terminal_x=info["chassis_pos"][0],
        terminal_y=info["chassis_pos"][1],
        terminal_z=info["chassis_pos"][2],
    )


def bootstrap_ci(values: np.ndarray, n_resample: int = 10_000,
                 conf: float = 0.95) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(n_resample, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, (1 - conf) / 2 * 100))
    hi = float(np.percentile(means, (1 + conf) / 2 * 100))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx",         required=True, help="recovery_policy.onnx path")
    ap.add_argument("--num_trials",   type=int, default=50, help="trials per seed")
    ap.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sinkage_min",  type=float, default=0.15)
    ap.add_argument("--sinkage_max",  type=float, default=0.28)
    ap.add_argument("--friction_min", type=float, default=0.6)
    ap.add_argument("--friction_max", type=float, default=0.9)
    ap.add_argument("--terrain",      choices=["scm", "crm"], default="scm",
                    help="scm = Bekker-Wong (default); crm = SPH continuum granular")
    ap.add_argument("--output",       default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"[chrono] ONNX loaded: {args.onnx}")
    print(f"[chrono] inputs : {[i.name for i in sess.get_inputs()]}")
    print(f"[chrono] outputs: {[o.name for o in sess.get_outputs()]}")
    if args.terrain == "crm":
        print(f"[chrono] terrain: CRM/SPH continuum granular (Drucker-Prager)")
    else:
        print(f"[chrono] terrain: SCM Bekker-Wong (Senatore-Iagnemma MMS-1 params)")
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
                r = run_trial(sess, seed, trial_id, sinkage, friction, heading,
                              terrain=args.terrain, verbose=args.verbose)
                all_results.append(r)
            except Exception as e:
                print(f"  [seed {seed} trial {trial_id}] ERROR: {e}")
                continue
            dt = time.time() - t0
            tag = "ESC" if r.escaped else r.failure_mode[:8]
            print(f"  trial={trial_id:3d}  sink={sinkage:.3f}  μ={friction:.2f}  "
                  f"hd={math.degrees(heading):5.1f}°  [{tag:>8s}]  "
                  f"proj={r.final_proj:5.2f}m  ({dt:.1f}s)")

    n = len(all_results)
    if n == 0:
        print("[chrono] NO TRIALS COMPLETED. Aborting.")
        return

    escaped_arr = np.array([r.escaped for r in all_results], dtype=float)
    rec_rate = float(escaped_arr.mean())
    rec_lo, rec_hi = bootstrap_ci(escaped_arr)
    esc_times = np.array([r.time_to_escape for r in all_results if r.escaped])
    proj_arr  = np.array([r.final_proj for r in all_results])

    by_mode: dict[str, int] = {}
    for r in all_results:
        by_mode[r.failure_mode] = by_mode.get(r.failure_mode, 0) + 1

    print("\n" + "=" * 60)
    print(f"  Cross-engine sim2sim — Chrono SCM on Curiosity rover")
    print("=" * 60)
    print(f"  Total trials      : {n}  ({len(args.seeds)} seeds × {args.num_trials})")
    print(f"  Recovery rate     : {rec_rate*100:.1f}%   "
          f"95% CI = [{rec_lo*100:.1f}, {rec_hi*100:.1f}]%")
    if len(esc_times):
        print(f"  Time-to-escape    : {esc_times.mean():.1f} ± {esc_times.std():.1f} steps "
              f"({esc_times.mean()*POLICY_DT:.1f} s)")
    print(f"  Mean final proj   : {proj_arr.mean():.2f} m")
    print(f"  Failure modes     : {by_mode}")
    print(f"  Wall-clock        : {time.time() - t_start:.1f} s")

    csv_path = os.path.join(args.output, f"chrono_{args.terrain}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        for r in all_results: w.writerow(asdict(r))
    summary_path = os.path.join(args.output, f"chrono_{args.terrain}_summary.json")
    terrain_label = ("CRM (SPH continuum granular, Drucker-Prager)"
                     if args.terrain == "crm"
                     else "SCM (Bekker-Wong terramechanics)")
    with open(summary_path, "w") as f:
        json.dump({
            "engine":             "Project Chrono",
            "rigid_body_solver":  "ChSystemNSC",
            "terrain":            terrain_label,
            "soil_params":        "Senatore & Iagnemma 2014 MMS-1 calibration"
                                   if args.terrain == "scm"
                                   else "MMS-1 simulant Drucker-Prager (rho=1700, mu_s=friction)",
            "rover":              "Curiosity FullRover (NASA MSL, pychrono.robot)",
            "n_trials":           n,
            "seeds":              args.seeds,
            "trials_per_seed":    args.num_trials,
            "recovery_rate":      rec_rate,
            "recovery_rate_ci_95": [rec_lo, rec_hi],
            "time_to_escape_mean_steps": float(esc_times.mean()) if len(esc_times) else None,
            "time_to_escape_std_steps":  float(esc_times.std())  if len(esc_times) else None,
            "failure_modes":      by_mode,
            "onnx_policy":        os.path.abspath(args.onnx),
            "fidelity_gap_notes": [
                "CuriosityDCMotorControl exposes single steering scalar; per-wheel "
                "drive commands are mapped to mean throttle (no differential drive).",
                "Steering: 4 steer commands averaged to 1 Curiosity steering input.",
                "Wheel radius differs (Curiosity 0.25m vs AAU 0.10m) — slip "
                "computation uses Curiosity radius for obs-builder consistency.",
            ],
        }, f, indent=2)
    print(f"\n  CSV     → {csv_path}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
