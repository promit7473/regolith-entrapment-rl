"""Quick v_x per-trial analysis."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from chrono_validation import (run_trial, build_scene_aau, CuriosityObsBuilder,
    BekkerWheelTerrain, _read_omega, _damp_chassis, _torque_from_bekker,
    _probe_motor_func, POLICY_DT, PHYSICS_DT, SUBSTEPS, MAX_POLICY_STEPS,
    DRIVE_VEL_LIMIT, STEER_LIMIT, WHEEL_RADIUS_AAU, GRU_HIDDEN, GRU_LAYERS,
    ESCAPE_DISTANCE, AAU_STEER_INDICES)
import onnxruntime as ort

ONNX = "/media/rmedu/18C6E68BC6E66888/regolith-entrapment-rl/sim2real/onnx_export/output/recovery_policy.onnx"
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

rng = np.random.default_rng(99)
for tid in range(10):
    friction_deg = float(rng.uniform(10, 30))
    heading = float(rng.uniform(0, 2*math.pi))
    r = run_trial(99, tid, "granular", friction_deg, friction_deg, heading,
                  control_mode="policy", onnx_session=sess, rover_type="aau")
    tag = "ESC" if r.escaped else "FAIL"
    print(f"trial {tid:2d}: {tag}  φ={friction_deg:.1f}°  "
          f"proj={r.final_proj:+.2f}m  sink={r.mean_sinkage:.3f}m  "
          f"mode={r.failure_mode}")
