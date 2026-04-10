"""
Mars Rover ArticulationCfg for Newton regolith entrapment training.

Robot:  6-wheel rocker-bogie Mars rover
USD:    robots/Mars_Rover.usd

Joint layout
  Drive  (velocity control) : FL/ML/RL/FR/MR/RR_Drive_Continuous   ×6
  Steer  (position control) : FL/RL/FR/RR_Steer_Revolute            ×4
  Passive (free)            : Rocker_Revolute, Differential_Revolute ×N

Velocity control:  stiffness=0, damping=4000
Steering control:  stiffness=8000, damping=1000
Passive joints:    stiffness=0,  damping=0,  effort=0
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
            max_linear_velocity=1.5,
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
            effort_limit_sim=12.0,     # Nm
            stiffness=0.0,
            damping=4000.0,
        ),
        # 4 corner steering joints — position control
        "wheel_steer": ImplicitActuatorCfg(
            joint_names_expr=[".*Steer_Revolute"],
            velocity_limit_sim=6.0,
            effort_limit_sim=12.0,
            stiffness=8000.0,
            damping=1000.0,
        ),
        # Passive rocker-bogie linkages — free to rotate
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*(Rocker|Differential)_Revolute"],
            velocity_limit_sim=15.0,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
