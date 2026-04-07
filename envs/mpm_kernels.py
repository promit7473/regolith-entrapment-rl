"""
Warp kernels for MPM ↔ XPBD two-way coupling, per-env particle reset,
and escaped-particle clamping.

Ported from newton/examples/mpm/example_mpm_twoway_coupling.py and extended
with a partial-reset kernel so individual envs can be reset without disturbing
neighbouring sand patches.
"""

import warp as wp


# ── Two-way coupling ───────────────────────────────────────────────────────────

@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids:        wp.array(dtype=int),
    collider_impulses:   wp.array(dtype=wp.vec3),
    collider_impulse_pos:wp.array(dtype=wp.vec3),
    body_ids:            wp.array(dtype=int),
    body_q:              wp.array(dtype=wp.transform),
    body_com:            wp.array(dtype=wp.vec3),
    body_f:              wp.array(dtype=wp.spatial_vector),
):
    """Convert sand impulses on MPM grid nodes into forces/torques on rigid bodies."""
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        X_wb    = body_q[body_index]
        X_com   = body_com[body_index]
        r       = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index,
                      wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt:              float,
    body_q:          wp.array(dtype=wp.transform),
    body_qd:         wp.array(dtype=wp.spatial_vector),
    body_f:          wp.array(dtype=wp.spatial_vector),
    body_inv_inertia:wp.array(dtype=wp.mat33),
    body_inv_mass:   wp.array(dtype=float),
    body_q_res:      wp.array(dtype=wp.transform),
    body_qd_res:     wp.array(dtype=wp.spatial_vector),
):
    """
    Write body_q/qd into sand_state, subtracting the effect of the sand
    force that was already applied in the last XPBD step to avoid
    double-counting during MPM collider evaluation.
    """
    b = wp.tid()
    f       = body_f[b]
    delta_v = dt * body_inv_mass[b] * wp.spatial_top(f)
    r       = wp.transform_get_rotation(body_q[b])
    delta_w = dt * wp.quat_rotate(
        r, body_inv_inertia[b] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    )
    body_q_res[b]  = body_q[b]
    body_qd_res[b] = body_qd[b] - wp.spatial_vector(delta_v, delta_w)


# ── Escaped-particle clamp ─────────────────────────────────────────────────────

@wp.kernel
def clamp_escaped_particles(
    particle_q:        wp.array(dtype=wp.vec3),
    env_origins:       wp.array(dtype=wp.vec3),
    particles_per_env: int,
    half_x:            float,
    half_y:            float,
    depth:             float,
):
    """
    After each MPM step, snap any particle that left its env's sand box back
    inside.  Keeps the sparse VDB bounding box bounded → prevents OOM in
    volume_builder.cu when a single particle drifts to a large position.

    Bounds per env:
        x ∈ [origin_x - half_x,  origin_x + half_x]
        y ∈ [origin_y - half_y,  origin_y + half_y]
        z ∈ [origin_z - 0.05,    origin_z + depth + 0.10]
    """
    i   = wp.tid()
    env = i // particles_per_env
    o   = env_origins[env]
    p   = particle_q[i]
    px  = wp.clamp(p[0], o[0] - half_x,       o[0] + half_x)
    py  = wp.clamp(p[1], o[1] - half_y,       o[1] + half_y)
    pz  = wp.clamp(p[2], o[2] - float(0.05),  o[2] + depth + float(0.10))
    particle_q[i] = wp.vec3(px, py, pz)


# ── Per-env particle reset ─────────────────────────────────────────────────────

@wp.kernel
def reset_particle_range(
    q_init:               wp.array(dtype=wp.vec3),
    qd_init:              wp.array(dtype=wp.vec3),
    q_out:                wp.array(dtype=wp.vec3),
    qd_out:               wp.array(dtype=wp.vec3),
    elastic_strain_out:   wp.array(dtype=wp.mat33),
    Jp_out:               wp.array(dtype=float),
    qd_grad_out:          wp.array(dtype=wp.mat33),
    transform_out:        wp.array(dtype=wp.mat33),
    start:                int,
    count:                int,
):
    """Reset one env's particles to their initial (undeformed) state."""
    i = wp.tid()
    if i < count:
        idx = start + i
        q_out[idx]  = q_init[idx]
        qd_out[idx] = qd_init[idx]
        I = wp.mat33(1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0)
        Z = wp.mat33(0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)
        elastic_strain_out[idx] = I
        transform_out[idx]      = I
        qd_grad_out[idx]        = Z
        Jp_out[idx]             = 1.0
