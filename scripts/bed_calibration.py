"""Standalone MPM bed calibration — pure Newton, no Isaac Sim.

Settles a sand bed identical to the training env's (3.5×3.5×0.6 m, Mars
gravity) under a chosen parameter set and reports:
  - surface height over time (compaction trajectory)
  - final vertical density profile (mass per 5 cm slab)
  - mass-conservation check (particles below ground / above box)

Purpose: the training bed was observed to compact ~33% during pre-settle and
then fail to bear the rover statically (sinks to the rigid floor with zero
action). This script isolates the granular column from the robot stack so the
responsible parameter can be found in minutes per run instead of ~10.

Run (no launch.sh needed — pure newton + warp):
    ~/miniconda3/envs/env_isaaclab/bin/python3 scripts/bed_calibration.py \
        --critical_fraction 0.0 --steps 600
"""

import argparse
import sys

import numpy as np
import warp as wp

import newton as nt
from newton.solvers import SolverImplicitMPM

SAND_HALF = 1.75
SAND_DEPTH = 0.60
DENSITY = 1700.0


@wp.kernel
def clamp_box(
    particle_q: wp.array(dtype=wp.vec3),
    half_x: float,
    half_y: float,
    depth: float,
):
    i = wp.tid()
    p = particle_q[i]
    px = wp.clamp(p[0], -half_x, half_x)
    py = wp.clamp(p[1], -half_y, half_y)
    pz = wp.clamp(p[2], -0.05, depth + 0.10)
    particle_q[i] = wp.vec3(px, py, pz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--ppc", type=float, default=2.0)
    ap.add_argument("--jitter_factor", type=float, default=0.5,
                    help="jitter = factor * radius")
    ap.add_argument("--radius_factor", type=float, default=0.5,
                    help="radius = factor * max(cell_size)")
    ap.add_argument("--gravity", type=float, default=-3.72)
    ap.add_argument("--critical_fraction", type=float, default=0.025)
    ap.add_argument("--hardening", type=float, default=5.0)
    ap.add_argument("--yield_pressure", type=float, default=1.0e12)
    ap.add_argument("--yield_stress", type=float, default=0.0)
    ap.add_argument("--air_drag", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=0.75)
    ap.add_argument("--particle_ke", type=float, default=2.0e5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--strain_basis", type=str, default="P0", choices=["P0", "Q1"])
    # Defaults = PRODUCTION solver settings (audit fix A3): the original
    # 30/1e-5 defaults reproduced the broken (volume-leaking) bed — anyone
    # running this script bare would re-measure the bug, not the physics.
    ap.add_argument("--tolerance", type=float, default=1.0e-7)
    ap.add_argument("--max_iterations", type=int, default=100)
    ap.add_argument("--no_clamp", action="store_true",
                    help="disable the box-wall clamp during settling")
    args = ap.parse_args()

    lo = np.array([-SAND_HALF, -SAND_HALF, 0.0])
    hi = np.array([SAND_HALF, SAND_HALF, SAND_DEPTH])
    res = np.array(np.ceil(args.ppc * (hi - lo) / args.voxel), dtype=int)
    cell = (hi - lo) / res
    radius = float(np.max(cell) * args.radius_factor)
    mass = float(np.prod(cell) * DENSITY)

    builder = nt.ModelBuilder(up_axis=nt.Axis.Z, gravity=args.gravity)
    builder.add_particle_grid(
        pos=wp.vec3(*lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1, dim_y=int(res[1]) + 1, dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
        mass=mass,
        jitter=args.jitter_factor * radius,
        radius_mean=radius,
    )
    builder.add_ground_plane()
    model = builder.finalize()
    n = model.particle_count
    model.particle_mu = wp.full(n, args.friction, dtype=float, device=model.device)
    model.particle_ke = args.particle_ke

    opt = SolverImplicitMPM.Options()
    opt.voxel_size = args.voxel
    opt.tolerance = args.tolerance
    opt.grid_type = "sparse"
    opt.transfer_scheme = "apic"
    opt.strain_basis = args.strain_basis
    opt.max_iterations = args.max_iterations
    opt.critical_fraction = args.critical_fraction
    opt.hardening = args.hardening
    opt.yield_pressure = args.yield_pressure
    opt.yield_stress = args.yield_stress
    opt.air_drag = args.air_drag

    mpm_model = SolverImplicitMPM.Model(model, opt)
    mpm_model.setup_collider(model=None, ground_height=0.0)
    solver = SolverImplicitMPM(mpm_model, opt)
    state = model.state()
    solver.enrich_state(state)

    total_mass = n * mass
    print(f"bed: {n} particles, cell={cell[0]:.4f}, radius={radius:.4f}, "
          f"mass/particle={mass:.5f} kg, total={total_mass:.0f} kg")
    print(f"params: cf={args.critical_fraction} hard={args.hardening} "
          f"yp={args.yield_pressure:.1e} ys={args.yield_stress} "
          f"drag={args.air_drag} mu={args.friction} g={args.gravity} "
          f"jit={args.jitter_factor}r")

    def surface(q):
        z = q[:, 2]
        return float(np.quantile(z, 0.99))

    q0 = state.particle_q.numpy()
    print(f"step {0:4d}: surface={surface(q0):.3f}  mean|v|=0.0000")

    import time
    t_loop = time.perf_counter()
    n_done = 0
    for i in range(args.steps):
        solver.step(state, state, contacts=None, control=None, dt=args.dt)
        # Match the env loop: evict particles that penetrated colliders
        # (here: the ground plane) — without this ~5% of the bed leaks below z=0.
        solver.project_outside(state, state, dt=args.dt)
        if not args.no_clamp:
            wp.launch(clamp_box, dim=n,
                      inputs=[state.particle_q, SAND_HALF, SAND_HALF, SAND_DEPTH],
                      device=model.device)
        n_done = i + 1
        if (i + 1) % 50 == 0:
            q = state.particle_q.numpy()
            v = np.linalg.norm(state.particle_qd.numpy(), axis=1).mean()
            print(f"step {i+1:4d}: surface={surface(q):.3f}  mean|v|={v:.4f}")
            sys.stdout.flush()
            if v < 0.003:
                break

    dt_wall = time.perf_counter() - t_loop
    print(f"timing: {n_done} steps in {dt_wall:.1f}s = {dt_wall/max(1,n_done)*1000:.0f} ms/step")
    q = state.particle_q.numpy()
    s = surface(q)
    print(f"\nFINAL surface = {s:.3f} m (as-built {SAND_DEPTH} m, "
          f"compaction {SAND_DEPTH - s:.3f} m = {(SAND_DEPTH - s)/SAND_DEPTH*100:.0f}%)")
    below = int((q[:, 2] < -0.01).sum())
    print(f"particles below ground: {below} ({below/n*100:.2f}%)")
    # Vertical density profile, central 1×1 m column (avoids wall piles)
    central = (np.abs(q[:, 0]) < 0.5) & (np.abs(q[:, 1]) < 0.5)
    zc = q[central, 2]
    print("density profile (central 1 m² column, 5 cm slabs):")
    for z0 in np.arange(0.0, 0.65, 0.05):
        m_slab = ((zc >= z0) & (zc < z0 + 0.05)).sum() * mass
        rho = m_slab / (1.0 * 1.0 * 0.05)
        print(f"  z {z0:.2f}-{z0+0.05:.2f}: rho = {rho:7.0f} kg/m³")


if __name__ == "__main__":
    main()
