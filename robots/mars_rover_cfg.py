"""
Mars Rover ArticulationCfg for Newton regolith entrapment training.

Robot:  6-wheel rocker-bogie Mars rover
USD:    robots/Mars_Rover.usd

Joint layout
  Drive  (velocity control) : FL/CL/RL/FR/CR/RR_Drive_Continuous   ×6
  Steer  (position control) : FL/RL/FR/RR_Steer_Revolute            ×4
  Passive (free)            : Rocker_Revolute, Differential_Revolute ×N

Velocity control:  stiffness=0, damping=4000, effort_limit=40 Nm
Steering control:  stiffness=8000, damping=1000
Passive joints:    stiffness=0,  damping=0,  effort=0

Torque signal rationale (damping=4000, effort_limit=40):
  Free driving steady-state (v_error ≈ 0.005 rad/s) → τ = 20 Nm, ratio = 0.50 → no anomaly ✓
  Sand burial               (v_error ≈ 2   rad/s)   → τ = 8000 → capped 40 Nm → ratio = 1.0 → anomaly ✓
  Effort limit lowered 80 → 40 Nm: torque was saturating ~99.95% of the time at 80,
  giving every step max-thrust → impulsive lurches → visible "hop-and-grab" bouncing.
  Halving the cap restores torque-signal headroom AND gives a smoother thrust profile
  (mid-range τ values are now reachable instead of always saturated).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROVER_USD_PATH = os.path.join(_REPO_ROOT, "robots", "Mars_Rover.usd")

# Wheel radius: 0.10 m
ROVER_WHEEL_RADIUS = 0.10   # m

MARS_ROVER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROVER_USD_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.04,
            rest_offset=0.01,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=3.0,   # raised: escape maneuver needs >1.5 m/s headroom
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.5,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),   # safe spawn above sand — settles onto ground plane via physics
        joint_pos={".*Steer_Revolute": 0.0},
        joint_vel={
            ".*Drive_Continuous": 0.0,
            ".*Steer_Revolute": 0.0,
        },
    ),
    actuators={
        # 6 drive wheels — velocity control (stiffness=0)
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*Drive_Continuous"],
            velocity_limit_sim=6.0,    # rad/s
            effort_limit_sim=22.0,     # Nm — v8: 40 still saturated (raw_mean_torque_ratio=1.000) — policy learned bang-bang at 40 too. Free-driving steady-state τ ≈ 20 Nm, so 22 forces the policy off the saturator and gives genuine thrust dynamic range. Was 80 → 40 in v7 (fixed bouncing) → 22 in v8 (fixes saturation).
            stiffness=0.0,
            damping=4000.0,            # unchanged — dynamics preserved; effort_limit raise alone fixes the signal
        ),
        # 4 corner steering joints — position control
        "wheel_steer": ImplicitActuatorCfg(
            joint_names_expr=[".*Steer_Revolute"],
            velocity_limit_sim=6.0,
            effort_limit_sim=12.0,
            stiffness=8000.0,
            damping=1000.0,
        ),
        # Passive rocker-bogie linkages — slight stiffness to prevent body collapse.
        # Isaac Lab's _process_actuators_cfg overwrites Newton builder's joint_target_ke/kd
        # with these values, so stiffness/damping here is the only effective setting.
        # effort_limit_sim > 0 is required — MuJoCo rejects actfrcrange=[0,0] AND
        # the stiffness torque needs headroom to actually apply.
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*(Rocker|Differential)_Revolute"],
            velocity_limit_sim=15.0,
            effort_limit_sim=200.0,  # headroom for stiffness torque (not driven by policy)
            stiffness=150.0,         # keeps rocker-bogie from collapsing under gravity
            damping=5.0,
        ),
    },
)
