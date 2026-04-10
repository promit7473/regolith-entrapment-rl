#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Regolith Entrapment Research — Launcher
#
# Usage:
#   ./launch.sh scripts/train.py --headless --num_envs 64              # Headless training
#   ./launch.sh scripts/eval.py  --num_envs 4                           # Newton ViewerGL
#   ./launch.sh scripts/eval.py  --headless  --num_envs 4               # Headless evaluation
#
# For full Isaac Sim GUI from desktop: open view_gui.sh from your GNOME Terminal.
#
# Environment variables (optional, see paths.py):
#   ISAAC_SIM_PATH     — Isaac Sim installation
#   ISAACLAB_SRC_PATH  — Isaac Lab source directory
#   NEWTON_PATH        — Newton installation
#   CONDA_ENV_PATH     — Conda environment path
#   RLROVER_ASSETS     — RLRoverLab assets directory
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source path configuration (with environment variable overrides)
source "$REPO_DIR/paths.sh"

PYTHON="$CONDA_ENV/bin/python3"

# Kit kernel must be importable BEFORE sourcing Isaac Sim env (for omni.kit bootstrap)
export PYTHONPATH="$ISAAC_SIM/kit:${PYTHONPATH}"

# Isaac Sim's env setup (adds kit/python libs, plugin bindings, etc.)
source "$ISAAC_SIM/setup_python_env.sh"

# Our packages take priority
export PYTHONPATH=\
"$NEWTON_DIR":\
"$REPO_DIR":\
"$ISAACLAB_SRC/isaaclab_rl":\
"$ISAACLAB_SRC/isaaclab_assets":\
"$ISAACLAB_SRC/isaaclab_tasks":\
"$ISAACLAB_SRC/isaaclab_tasks_experimental":\
"$ISAACLAB_SRC/isaaclab_newton":\
"$ISAACLAB_SRC/isaaclab_experimental":\
"$ISAAC_SIM/exts/isaacsim.simulation_app":\
"$ISAAC_SIM/python_packages":\
"${PYTHONPATH}"

export LD_LIBRARY_PATH="$PXR_EXT/bin:$CONDA_ENV/lib:${LD_LIBRARY_PATH}"
export ISAAC_PATH="$ISAAC_SIM"
export OMNI_KIT_ACCEPT_EULA=YES

# Preload jemalloc to prevent heap corruption when Isaac Sim's native code
# (which uses jemalloc) runs alongside glibc malloc from conda Python.
# Must be set before the Python process starts — use exec with env override.
export LD_PRELOAD="$ISAAC_SIM/kit/libjemalloc.so:${LD_PRELOAD}"

echo "[launch.sh] Python : $PYTHON"
echo "[launch.sh] Script : $1"
echo "[launch.sh] Args   : ${@:2}"
echo ""

exec "$PYTHON" "$REPO_DIR/_bootstrap.py" "$@"
