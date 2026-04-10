"""
Bootstrap: injects pxr and omni stubs into sys.path, then runs the target script.
Called by launch.sh — not meant to be run directly.
"""
import sys
import os
import runpy

# Load path configuration
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from paths import ISAAC_SIM, PXR_EXT

# 0. Bootstrap the Kit kernel FIRST — isaacsim.__init__.bootstrap_kernel() skips this
#    for standalone python_packages/isaacsim installs when using conda Python.
#    We must import kit_app before any isaacsim import so omni.kit.app is available.
_KIT_DIR = str(ISAAC_SIM / "kit")
if _KIT_DIR not in sys.path:
    sys.path.insert(0, _KIT_DIR)
try:
    import kit_app as _kit_app  # noqa: F401 — side-effect: registers omni.kit C++ bindings
except Exception as _e:
    print(f"[bootstrap] WARNING: kit_app bootstrap failed: {_e}")

# 1. Inject pxr (must be via sys.path inside the process due to LD_LIBRARY_PATH timing)
_PXR_EXT_STR = str(PXR_EXT)
if _PXR_EXT_STR not in sys.path:
    sys.path.insert(0, _PXR_EXT_STR)

# 2. Inject omni.client stub (for Isaac Lab asset utilities — we don't use Nucleus)
_STUBS = os.path.join(_REPO, "stubs")
if _STUBS not in sys.path:
    sys.path.insert(1, _STUBS)

# 3. Run the target script
if len(sys.argv) < 2:
    print("Usage: python _bootstrap.py <script.py> [args...]")
    sys.exit(1)

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
