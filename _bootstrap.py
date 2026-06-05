import sys
import os
import runpy


# ── Fix: Isaac Sim's bundled platform.py ships an old _sys_version regex that
# lacks the optional "| packaged by conda-forge |" segment, so it raises
# ValueError on the conda-forge interpreter's sys.version string (e.g.
# '3.11.15 | packaged by conda-forge | (main, ...)'). Any launch-time import of
# rerun → pyarrow → cloudpickle calls platform.python_implementation() and
# crashes Kit during AppLauncher startup. Patch the regex in-process before the
# app boots. No-op when the loaded platform.py already parses correctly.
def _patch_platform_version_parser():
    import re
    import platform
    try:
        platform.python_implementation()
        return  # already parses fine — nothing to do
    except ValueError:
        pass
    # Newer CPython regex: adds the optional "(?:\|[^|]*\|)?\s*" build-info group.
    platform._sys_version_parser = re.compile(
        r"([\w.+]+)\s*"
        r"(?:\|[^|]*\|)?\s*"
        r"\(#?([^,]+)"
        r"(?:,\s*([\w ]*)"
        r"(?:,\s*([\w :]*))?)?\)\s*"
        r"\[([^\]]+)\]?",
        re.ASCII,
    )
    platform._sys_version_cache.clear()


_patch_platform_version_parser()


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
