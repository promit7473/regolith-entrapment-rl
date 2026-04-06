#!/usr/bin/env bash
# Standalone Newton viewer for the AAU Mars Rover
# Usage: ./view.sh [--no-sand] [--num-frames N]

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="/home/mhpromit7473/miniconda3/envs/env_isaaclab"
PXR_EXT="/home/mhpromit7473/.local/share/ov/data/exts/v2/omni.usd.libs-4fde11c8f289f1f4"
PYTHON="$CONDA_ENV/bin/python3"

export LD_PRELOAD="/home/mhpromit7473/isaac-sim/kit/libjemalloc.so"
export LD_LIBRARY_PATH="$PXR_EXT/bin:$CONDA_ENV/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "[view.sh] Attempt $i..."
    "$PYTHON" "$REPO_DIR/_bootstrap.py" "$REPO_DIR/scripts/view_rover.py" "$@"
    CODE=$?
    [ $CODE -eq 0 ] && exit 0
    [ $CODE -eq 139 ] || [ $CODE -eq 134 ] && { sleep 1; continue; }
    exit $CODE
done
echo "[view.sh] Failed after 10 attempts."
exit 1
