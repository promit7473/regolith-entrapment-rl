"""
Standalone Newton viewer for the Mars Rover (6-wheel) + MPM sand.
No Isaac Lab, no SimulationApp — just Newton + ViewerGL.

Usage:
    ./view.sh                          # rover + sand (interactive)
    ./view.sh --no-sand                # rover only
    ./view.sh --num-frames 500         # 500 frames then exit
"""

import sys
import os
import argparse
import numpy as np

# Load path configuration
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from paths import PXR_EXT, NEWTON_DIR

# pxr MUST be imported before warp — import order matters to avoid segfault
_PXR_EXT_STR = str(PXR_EXT)
if _PXR_EXT_STR not in sys.path:
    sys.path.insert(0, _PXR_EXT_STR)
from pxr import Usd, UsdGeom, UsdPhysics, Sdf  # noqa: F401 — import before warp

# Newton must be importable
sys.path.insert(0, str(NEWTON_DIR))

import mujoco
import warp as wp
import newton
import newton.examples
from newton.viewer import ViewerGL

@wp.kernel
def clamp_escaped_particles(
    particle_q:        wp.array(dtype=wp.vec3),
    env_origins:       wp.array(dtype=wp.vec3),
    particles_per_env: int,
    half_x: float, half_y: float, depth: float,
):
    i   = wp.tid()
    env = i // particles_per_env
    o   = env_origins[env]
    p   = particle_q[i]
    particle_q[i] = wp.vec3(
        wp.clamp(p[0], o[0] - half_x,       o[0] + half_x),
        wp.clamp(p[1], o[1] - half_y,       o[1] + half_y),
        wp.clamp(p[2], o[2] - float(0.05),  o[2] + depth + float(0.10)),
    )

ROVER_USD = os.path.join(_REPO_ROOT, "robots", "Mars_Rover.usd")


def main():
    parser = argparse.ArgumentParser(description="View Mars Rover (6-wheel) in Newton")
    parser.add_argument("--no-sand", action="store_true", help="Skip MPM sand")
    parser.add_argument("--num-frames", type=int, default=0, help="0 = infinite")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    args = parser.parse_args()

    device = wp.get_device()
    print(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5,
    )
    builder.default_shape_cfg.ke = 2.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    # Load Mars rover from USD
    print(f"Loading rover: {ROVER_USD}")
    builder.add_usd(
        ROVER_USD,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.50),
            wp.quat_identity(),
        ),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        load_visual_shapes=True,
    )

    n_bodies = builder.body_count
    n_joints = builder.joint_count
    n_usd_shapes = builder.shape_count
    print(f"Rover loaded: {n_bodies} bodies, {n_joints} joints, {n_usd_shapes} USD shapes")

    # Disable collision on ALL USD mesh shapes — 329 mesh shapes flood the
    # contact buffer (nconmax), causing uneven support → rover tears apart.
    # Keep VISIBLE flag so the rover still renders.
    for s in range(n_usd_shapes):
        builder.shape_flags[s] = builder.shape_flags[s] & ~newton.ShapeFlags.COLLIDE_SHAPES
    print(f"Disabled COLLIDE_SHAPES on {n_usd_shapes} USD mesh shapes")

    # Add proxy collision spheres ONLY on wheel bodies (6 Drive joints).
    # These are the ONLY shapes that collide with the ground — simple and stable.
    proxy_cfg = newton.ModelBuilder.ShapeConfig(
        ke=2.0e3, kd=1.0e2, kf=1.0e3, mu=0.75, density=0.0,
        has_shape_collision=True,
        is_visible=False,
    )
    body_keys = builder.body_key if hasattr(builder, 'body_key') else []
    n_wheels = 0
    for b in range(n_bodies):
        name = body_keys[b] if b < len(body_keys) else ''
        if 'Drive' in name:
            builder.add_shape_sphere(body=b, radius=0.10, cfg=proxy_cfg)
            n_wheels += 1
    print(f"Added {n_wheels} proxy collision spheres on wheel bodies")

    # Joint stiffness/damping for all DOFs
    for i in range(builder.joint_dof_count):
        builder.joint_target_ke[i] = 150
        builder.joint_target_kd[i] = 5

    # Fix: MuJoCo requires actfrcrange[0] < actfrcrange[1].
    # Passive joints in the USD have effort_limit=0 → set small value.
    for i in range(len(builder.joint_effort_limit)):
        if builder.joint_effort_limit[i] <= 0.0:
            builder.joint_effort_limit[i] = 1.0

    builder.add_ground_plane()

    # ── Sand particles ───────────────────────────────────────────────────
    if not args.no_sand:
        print("Adding sand particles...")
        lo = np.array([-0.6, -0.6, 0.0])
        hi = np.array([0.6, 0.6, 0.15])
        ppc = 2.0
        res = np.array(np.ceil(ppc * (hi - lo) / args.voxel_size), dtype=int)
        cell_size = (hi - lo) / res
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_size) * 1700.0)

        builder.add_particle_grid(
            pos=wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )
        print(f"Sand: {len(builder.particle_q)} particles, voxel={args.voxel_size}m")

    # ── Finalize ─────────────────────────────────────────────────────────
    print("Finalizing model...")
    model = builder.finalize()
    print(f"Final model: {model.body_count} bodies, {model.shape_count} shapes, "
          f"{model.particle_count} particles")

    model.set_gravity(wp.vec3(0.0, 0.0, -3.72))  # Mars gravity (3.72 m/s²)

    # ── Solver ───────────────────────────────────────────────────────────
    fps = 50
    frame_dt = 1.0 / fps
    substeps = 4
    sim_dt = frame_dt / substeps

    # MuJoCo solver — stable for articulated robots
    solver = newton.solvers.SolverMuJoCo(
        model,
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
        impratio=100,
        iterations=100,
        ls_iterations=50,
        nconmax=50,
        njmax=200,
    )

    # Collision pipeline
    collision_pipeline = newton.examples.create_collision_pipeline(model)

    # MPM solver (if sand)
    mpm_solver = None
    if not args.no_sand and model.particle_count > 0:
        from newton.solvers import SolverImplicitMPM
        mpm_opt = SolverImplicitMPM.Options()
        mpm_opt.voxel_size = args.voxel_size
        mpm_opt.tolerance = 1.0e-5
        mpm_opt.grid_type = "sparse"
        mpm_opt.transfer_scheme = "pic"
        mpm_opt.strain_basis = "P0"
        mpm_opt.max_iterations = 30
        mpm_opt.hardening = 5.0
        mpm_opt.critical_fraction = 0.025
        mpm_opt.air_drag = 1.0

        mpm_model = SolverImplicitMPM.Model(model, mpm_opt)
        mpm_model.setup_collider(model=model, ground_height=0.0)
        mpm_solver = SolverImplicitMPM(mpm_model, mpm_opt)
        model.particle_mu = 0.7
        model.particle_ke = 5.0e4

    # ── State ────────────────────────────────────────────────────────────
    state_0 = model.state()
    state_1 = model.state()

    if mpm_solver:
        mpm_solver.enrich_state(state_0)
        mpm_solver.enrich_state(state_1)

    control = model.control()

    # Forward kinematics
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Initial collision
    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

    # Print body positions
    body_q = state_0.body_q.numpy()
    print("Body positions after FK:")
    for i in range(min(model.body_count, 5)):
        pos = body_q[i][:3]
        name = builder.body_key[i] if hasattr(builder, 'body_key') else f'body_{i}'
        print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # ── Viewer ───────────────────────────────────────────────────────────
    print("Opening ViewerGL...")
    viewer = ViewerGL(width=1440, height=900, vsync=False)
    viewer.set_model(model)
    if model.particle_count > 0:
        viewer.show_particles = True

    viewer.set_camera(
        pos=wp.vec3(2.5, -2.5, 1.5),
        pitch=-20.0,
        yaw=135.0,
    )

    # ── Simulation loop ──────────────────────────────────────────────────
    sim_time = 0.0
    frame = 0

    print(f"\n{'='*50}")
    print(f"  Mars Rover (6-wheel) — Newton Standalone Viewer")
    print(f"  Solver: MuJoCo (stable for articulated robots)")
    print(f"  Bodies: {model.body_count} | Shapes: {model.shape_count} | "
          f"Particles: {model.particle_count}")
    print(f"  Sand: {'ON' if mpm_solver else 'OFF'}")
    print(f"  Gravity: Mars (3.72 m/s²)")
    print(f"  Controls: WASD + mouse drag to move camera")
    print(f"{'='*50}\n")

    try:
        while viewer.is_running():
            for _ in range(substeps):
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
                solver.step(state_0, state_1, control, contacts=contacts, dt=sim_dt)
                state_0, state_1 = state_1, state_0

            if mpm_solver:
                mpm_solver.step(state_0, state_0, contacts=None, control=None, dt=frame_dt)
                origin = wp.array(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3)
                wp.launch(clamp_escaped_particles,
                          dim=model.particle_count,
                          inputs=[state_0.particle_q, origin,
                                  model.particle_count, 0.6, 0.6, 0.15])

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()

            sim_time += frame_dt
            frame += 1

            if args.num_frames > 0 and frame >= args.num_frames:
                break

            if frame % 200 == 0:
                bq = state_0.body_q.numpy()
                z0 = bq[0][2]
                print(f"  frame {frame}  sim_time={sim_time:.1f}s  body_z={z0:.4f}")

    except KeyboardInterrupt:
        print("\nStopped.")

    viewer.close()
    print("Done.")


if __name__ == "__main__":
    main()
