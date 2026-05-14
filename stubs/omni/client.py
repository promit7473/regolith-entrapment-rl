class Result:
    OK = 0
    ERROR = 1
    ERROR_NOT_FOUND = 2

class CopyBehavior:
    OVERWRITE = 0

def stat(path):
    return (Result.ERROR_NOT_FOUND, None)

def copy(src, dst, behavior=CopyBehavior.OVERWRITE):
    return Result.ERROR_NOT_FOUND

def read_file(path):
    return (Result.ERROR_NOT_FOUND, None, b"")
