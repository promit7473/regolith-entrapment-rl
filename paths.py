"""
Centralized path configuration with environment variable overrides.

Set these environment variables to customize paths for your system:
    ISAAC_SIM_PATH     — Isaac Sim installation (default: ~/isaac-sim)
    ISAACLAB_SRC_PATH  — Isaac Lab source directory (default: ~/IsaacLab/source)
    NEWTON_PATH        — Newton installation (default: ~/newton)
    CONDA_ENV_PATH     — Conda environment path (default: ~/miniconda3/envs/env_isaaclab)
    RLROVER_ASSETS     — RLRoverLab assets directory (default: ~/RLRoverLab/rover_envs/assets)
    PXR_EXT_PATH       — PXR extension path (default: auto-detected)

Usage:
    from paths import ISAAC_SIM, ISAACLAB_SRC, NEWTON_DIR, CONDA_ENV, RLROVER_ASSETS
"""

import os
from pathlib import Path

HOME = Path.home()

ISAAC_SIM = Path(os.environ.get(
    "ISAAC_SIM_PATH",
    HOME / "isaac-sim"
))

ISAACLAB_SRC = Path(os.environ.get(
    "ISAACLAB_SRC_PATH",
    HOME / "IsaacLab" / "source"
))

NEWTON_DIR = Path(os.environ.get(
    "NEWTON_PATH",
    HOME / "newton"
))

CONDA_ENV = Path(os.environ.get(
    "CONDA_ENV_PATH",
    HOME / "miniconda3" / "envs" / "env_isaaclab"
))

RLROVER_ASSETS = Path(os.environ.get(
    "RLROVER_ASSETS",
    HOME / "RLRoverLab" / "rover_envs" / "assets"
))

# Auto-detect PXR extension if not set
_pxr_default = HOME / ".local" / "share" / "ov" / "data" / "exts" / "v2"
_pxr_candidates = list(_pxr_default.glob("omni.usd.libs-*")) if _pxr_default.exists() else []
PXR_EXT = Path(os.environ.get(
    "PXR_EXT_PATH",
    _pxr_candidates[0] if _pxr_candidates else _pxr_default / "omni.usd.libs-4fde11c8f289f1f4"
))

PYTHON = CONDA_ENV / "bin" / "python3"
