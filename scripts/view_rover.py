"""
Standalone Newton viewer for the AAU Mars Rover + MPM sand.
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

# pxr MUST be imported before warp — import order matters to avoid segfault
_PXR_EXT = "/home/mhpromit7473/.local/share/ov/data/exts/v2/omni.usd.libs-4fde11c8f289f1f4"
if _PXR_EXT not in sys.path:
    sys.path.insert(0, _PXR_EXT)
from pxr import Usd, UsdGeom, UsdPhysics, Sdf  # noqa: F401 — import before warp

# Newton must be importable
sys.path.insert(0, "/home/mhpromit7473/newton")

import warp as wp
import newton
from newton.viewer import ViewerGL

ROVER_USD = "/home/mhpromit7473/RLRoverLab/rover_envs/assets/robots/aau_rover/Mars_Rover.usd"


def main():
    parser = argparse.ArgumentParser(description="View AAU Mars Rover in Newton")
    parser.add_argument("--no-sand", action="store_true", help="Skip MPM sand")
    parser.add_argument("--num-frames", type=int, default=0, help="0 = infinite")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    args = parser.parse_args()

    device = wp.get_device()
    print(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    # Softer contact to prevent violent bounce on landing
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.kf = 5.0e2
    builder.default_shape_cfg.mu = 0.75

    # Load AAU rover from USD
    print(f"Loading rover: {ROVER_USD}")
    # Wheel centers are at z=-0.167, radius=0.10 → bottom at z=-0.267.
    # So spawn root at z=+0.30 to have wheels just touching ground.
    builder.add_usd(
        ROVER_USD,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.30),
            wp.quat_identity(),
        ),
        load_visual_shapes=True,
        hide_collision_shapes=False,
    )

    n_bodies = builder.body_count
    n_joints = builder.joint_count
    print(f"Rover loaded: {n_bodies} bodies, {n_joints} joints")

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

    # Disable gravity temporarily so rover stays still for inspection
    model.set_gravity(wp.vec3(0.0, 0.0, 0.0))

    # ── Solver ───────────────────────────────────────────────────────────
    fps = 50
    frame_dt = 1.0 / fps
    substeps = 4
    sim_dt = frame_dt / substeps

    solver = newton.solvers.SolverXPBD(model, iterations=4)

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
        mpm_model.setup_collider(
            body_mass=wp.zeros_like(model.body_mass),
        )
        mpm_solver = SolverImplicitMPM(mpm_model, mpm_opt)
        model.particle_mu = 0.7
        model.particle_ke = 1.0e14

    # ── State ────────────────────────────────────────────────────────────
    state_0 = model.state()
    state_1 = model.state()

    if mpm_solver:
        mpm_solver.enrich_state(state_0)
        mpm_solver.enrich_state(state_1)

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Print body positions so we know where the rover actually spawns
    body_q = state_0.body_q.numpy()
    print("Body positions after FK:")
    for i in range(min(model.body_count, 5)):
        pos = body_q[i][:3]
        name = builder.body_key[i] if hasattr(builder, 'body_key') else f'body_{i}'
        print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    control = model.control()

    # ── Viewer ───────────────────────────────────────────────────────────
    print("Opening ViewerGL...")
    viewer = ViewerGL(width=1440, height=900, vsync=False)
    viewer.set_model(model)
    viewer.show_visual = True
    viewer.show_collision = True   # USD shapes load as collision — needed to see them
    if model.particle_count > 0:
        viewer.show_particles = True

    # Camera from the run that showed the rover ("flying" run)
    viewer.set_camera(
        pos=wp.vec3(2.0, -2.0, 1.5),
        pitch=-0.25,
        yaw=0.4,
    )

    # ── Simulation loop ──────────────────────────────────────────────────
    sim_time = 0.0
    frame = 0

    print(f"\n{'='*50}")
    print(f"  AAU Mars Rover — Newton Standalone Viewer")
    print(f"  Bodies: {model.body_count} | Shapes: {model.shape_count} | "
          f"Particles: {model.particle_count}")
    print(f"  Sand: {'ON' if mpm_solver else 'OFF'}")
    print(f"  Controls: WASD + mouse drag to move camera")
    print(f"{'='*50}\n")

    try:
        while viewer.is_running():
            for _ in range(substeps):
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts=contacts, dt=sim_dt)
                state_0, state_1 = state_1, state_0

            if mpm_solver:
                mpm_solver.step(state_0, state_0, contacts=None, control=None, dt=frame_dt)

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

            sim_time += frame_dt
            frame += 1

            if args.num_frames > 0 and frame >= args.num_frames:
                break

            if frame % 100 == 0:
                print(f"  frame {frame}  sim_time={sim_time:.1f}s")

    except KeyboardInterrupt:
        print("\nStopped.")

    viewer.close()
    print("Done.")


if __name__ == "__main__":
    main()
