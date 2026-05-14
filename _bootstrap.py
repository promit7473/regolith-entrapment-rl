import sys
import os
import runpy


_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from paths import ISAAC_SIM, PXR_EXT


_KIT_DIR = str(ISAAC_SIM / "kit")
if _KIT_DIR not in sys.path:
    sys.path.insert(0, _KIT_DIR)
try:
    import kit_app as _kit_app
except Exception as _e:
    print(f"[bootstrap] WARNING: kit_app bootstrap failed: {_e}")


_PXR_EXT_STR = str(PXR_EXT)
if _PXR_EXT_STR not in sys.path:
    sys.path.insert(0, _PXR_EXT_STR)


_STUBS = os.path.join(_REPO, "stubs")
if _STUBS not in sys.path:
    sys.path.insert(1, _STUBS)


if len(sys.argv) < 2:
    print("Usage: python _bootstrap.py <script.py> [args...]")
    sys.exit(1)

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
