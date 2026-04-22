# Pinned Versions

These are the exact versions/commits used on the dev PC (RTX 5070 Ti, Ubuntu).
Match them on the lab PC to keep the patch file and training pipeline working.

## External repos

| Repo | Path | Commit | Branch |
|------|------|--------|--------|
| Newton | `~/newton/` | `551f6ee` | (Warp Raytrace: Renamed geom to shape) |
| IsaacLab | `~/IsaacLab/` | `44c26e31` | `feature/newton` |
| RLRoverLab | `~/RLRoverLab/` | `64aeb78` | main |

To pin on lab PC after copying:
```bash
cd ~/newton       && git checkout 551f6ee
cd ~/IsaacLab     && git checkout 44c26e31
cd ~/RLRoverLab   && git checkout 64aeb78
```

## Isaac Sim
- Version: **5.1.0** (isaacsim-* pip packages == 5.1.0.0)
- Install location: `~/isaac-sim/` (do NOT relocate)

## Python env
- See `environment.yml` for the full conda spec.
- Python 3.11, PyTorch 2.7.0+cu128, warp-lang 1.13.0, skrl 1.4.3, mujoco 3.6.0

## NVIDIA driver
- Driver: **580.126.09**
- CUDA runtime: **13.0**
- Lab PC must have driver ≥ 580.

## Patches
- `patches/newton_mujoco_bugfixes.patch` — apply against `~/newton/` after checkout.
  ```bash
  cd ~/newton && git apply ~/regolith_entrapment_research/patches/newton_mujoco_bugfixes.patch
  ```
