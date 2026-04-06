"""
Minimal omni.client stub for Newton standalone mode (no Nucleus server needed).
All Nucleus-path operations will gracefully return 'not found'.
"""

class Result:
    OK = 0
    ERROR = 1
    ERROR_NOT_FOUND = 2

class CopyBehavior:
    OVERWRITE = 0

def stat(path):
    """Always returns not-found for Nucleus paths in standalone mode."""
    return (Result.ERROR_NOT_FOUND, None)

def copy(src, dst, behavior=CopyBehavior.OVERWRITE):
    return Result.ERROR_NOT_FOUND

def read_file(path):
    return (Result.ERROR_NOT_FOUND, None, b"")
