"""Definitive AAU wheel geometry extraction from Mars_Rover.usd.

Resolves the discrepancies between quick bbox measurements (axis-aligned,
transform-sensitive) and raw local-vertex measurements (ignore prim
transforms): every vertex is mapped through its prim's full
local-to-world transform, the axle axis is identified from the geometry
itself (PCA: the axle is the axis of minimal extent for a wheel), and all
quantities are computed in the wheel's own cylindrical frame.

Outputs (per mesh of one wheel):
  - outer/inner radius (95th/5th percentile of vertex radii — robust)
  - axial width
  - blade count + per-blade angular width + blade plate thickness (arc length)
  - blade radial depth (blade-ring outer radius − tire outer radius)

Run:
  ~/miniconda3/envs/env_isaaclab/bin/python3 scripts/wheel_geometry.py
"""

import numpy as np
from pxr import Usd, UsdGeom

USD = "/home/mhpromit7473/regolith_entrapment_research/robots/Mars_Rover.usd"
WHEEL_PRIM = "/rover/FL_Drive"


def world_points(prim, cache):
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    xf = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)  # row-major 4x4
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (homo @ xf)[:, :3]


def analyze(name, pts, axle_dir, center):
    rel = pts - center
    ax = rel @ axle_dir
    radial = rel - np.outer(ax, axle_dir)
    r = np.linalg.norm(radial, axis=1)
    r_out = float(np.percentile(r, 95))
    r_in = float(np.percentile(r, 5))
    width = float(np.percentile(ax, 97.5) - np.percentile(ax, 2.5))
    print(f"\n{name}: {len(pts)} verts")
    print(f"  outer radius (p95) = {r_out:.4f} m   max = {r.max():.4f}")
    print(f"  inner radius (p5)  = {r_in:.4f} m")
    print(f"  axial width (p95)  = {width:.4f} m   full = {ax.max()-ax.min():.4f}")
    return r, radial, ax, r_out


def blade_analysis(r, radial, axle_dir, r_out, r_tire):
    # Blade region: vertices clearly beyond the tire surface
    m = r > r_tire + 0.25 * (r_out - r_tire)
    if m.sum() < 10:
        print("  no blade region found")
        return
    # Build in-plane basis
    a = axle_dir
    e1 = np.cross(a, [1.0, 0, 0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(a, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ang = np.arctan2(radial[m] @ e2, radial[m] @ e1)
    hist, edges = np.histogram(ang, bins=1440, range=(-np.pi, np.pi))
    occ = hist > 0
    # wrap-aware cluster count and widths
    occ_i = occ.astype(int)
    starts = np.where((occ_i - np.roll(occ_i, 1)) == 1)[0]
    n_blades = len(starts)
    # mean angular width of occupied clusters
    total_occ = occ.sum() * (2 * np.pi / 1440)
    mean_width_ang = total_occ / max(1, n_blades)
    r_mid = (r[m].mean())
    arc = mean_width_ang * r_mid
    print(f"  BLADES: count = {n_blades}")
    print(f"  blade mean radius  = {r_mid:.4f} m")
    print(f"  blade tip radius   = {np.percentile(r[m], 97):.4f} m")
    print(f"  blade root radius  = {np.percentile(r[m], 3):.4f} m")
    print(f"  blade angular width = {np.degrees(mean_width_ang):.2f} deg "
          f"→ plate arc thickness ≈ {arc*1000:.1f} mm")


def main():
    stage = Usd.Stage.Open(USD)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    wheel = stage.GetPrimAtPath(WHEEL_PRIM)
    meshes = []

    def walk(p):
        for c in p.GetChildren():
            if c.IsA(UsdGeom.Mesh):
                meshes.append(c)
            walk(c)
    walk(wheel)

    all_pts = np.concatenate([world_points(m, cache) for m in meshes])
    center = all_pts.mean(axis=0)
    # Axle = PCA axis of minimal extent (a wheel is wide in 2 axes, thin in 1)
    rel = all_pts - center
    cov = rel.T @ rel
    w, v = np.linalg.eigh(cov)
    axle_dir = v[:, 0] / np.linalg.norm(v[:, 0])
    print(f"wheel: {WHEEL_PRIM}")
    print(f"axle direction (world) = {np.round(axle_dir, 4)}")

    results = {}
    for m in meshes:
        pts = world_points(m, cache)
        r, radial, ax, r_out = analyze(m.GetName(), pts, axle_dir, center)
        results[m.GetName()] = (r, radial, r_out)

    # Tire = mesh named "Mesh"; blade ring = "Mesh_01"
    if "Mesh" in results and "Mesh_01" in results:
        r_tire = results["Mesh"][2]
        r_b, radial_b, r_bout = results["Mesh_01"]
        print(f"\n=== blade ring vs tire ===")
        print(f"  tire outer radius   = {r_tire:.4f} m")
        print(f"  blade radial depth  = {r_bout - r_tire:.4f} m")
        blade_analysis(r_b, radial_b, axle_dir, r_bout, r_tire)

    print("\n=== RECOMMENDED PROXY CONSTANTS ===")
    print("(equivalent cylinder: r_eff = r_tire + 0.5*blade_depth;")
    print(" blades: measured depth/arc-thickness/width; count limited by 5 cm voxel)")


if __name__ == "__main__":
    main()
