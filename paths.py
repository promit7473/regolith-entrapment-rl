import os
from pathlib import Path

HOME = Path.home()

ISAAC_SIM = Path(os.environ.get(
    "ISAAC_SIM_PATH",
    HOME / "Downloads" / "isaac-sim-standalone-5.0.0-linux-x86_64"
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
    Path("/media/rmedu/18C6E68BC6E66888/conda-envs/env_isaaclab")
))

RLROVER_ASSETS = Path(os.environ.get(
    "RLROVER_ASSETS",
    HOME / "RLRoverLab" / "rover_envs" / "assets"
))


_pxr_default = HOME / ".local" / "share" / "ov" / "data" / "exts" / "v2"
_pxr_candidates = list(_pxr_default.glob("omni.usd.libs-*")) if _pxr_default.exists() else []
PXR_EXT = Path(os.environ.get(
    "PXR_EXT_PATH",
    _pxr_candidates[0] if _pxr_candidates else _pxr_default / "omni.usd.libs-4fde11c8f289f1f4"
))

PYTHON = CONDA_ENV / "bin" / "python3"
