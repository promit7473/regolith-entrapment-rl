#!/usr/bin/env bash
# Path configuration for bash scripts.
# Sourced by launch.sh, view.sh, and other shell scripts.
# Override these by setting environment variables before running.
#
# Note: Variable names match paths.py (e.g., ISAAC_SIM_PATH not ISAAC_SIM)

: "${ISAAC_SIM_PATH:=$HOME/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64}"
: "${ISAACLAB_SRC_PATH:=$HOME/IsaacLab/source}"
: "${NEWTON_PATH:=$HOME/newton}"
: "${CONDA_ENV_PATH:=/media/rmedu/18C6E68BC6E66888/conda-envs/env_isaaclab}"
: "${RLROVER_ASSETS:=$HOME/RLRoverLab/rover_envs/assets}"

# For bash scripts, also export shorter aliases (used by launch.sh)
export ISAAC_SIM="$ISAAC_SIM_PATH"
export ISAACLAB_SRC="$ISAACLAB_SRC_PATH"
export NEWTON_DIR="$NEWTON_PATH"
export CONDA_ENV="$CONDA_ENV_PATH"

# Auto-detect PXR extension
PXR_EXT_BASE="$HOME/.local/share/ov/data/exts/v2"
if [ -d "$PXR_EXT_BASE" ]; then
    PXR_EXT_CANDIDATE=$(find "$PXR_EXT_BASE" -maxdepth 1 -type d -name "omni.usd.libs-*" 2>/dev/null | head -n1)
    : "${PXR_EXT_PATH:=${PXR_EXT_CANDIDATE:-$PXR_EXT_BASE/omni.usd.libs-4fde11c8f289f1f4}}"
else
    : "${PXR_EXT_PATH:=$PXR_EXT_BASE/omni.usd.libs-4fde11c8f289f1f4}"
fi
export PXR_EXT="$PXR_EXT_PATH"
