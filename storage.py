# storage.py
import os

from network import format_bytes


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def sqlite_size(db_path):
    return (file_size(db_path)
            + file_size(db_path + "-wal")
            + file_size(db_path + "-shm"))


def dir_size(path):
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += file_size(os.path.join(root, f))
    return total


def storage_report(db_path,
                   models_dir="models",
                   logs_dir="logs",
                   exports_dir="exports"):
    db = sqlite_size(db_path)
    models = dir_size(models_dir)
    logs = dir_size(logs_dir)
    exports = dir_size(exports_dir)
    total = db + models + logs + exports
    return {
        "database": db,
        "models": models,
        "logs": logs,
        "exports": exports,
        "total": total,
        "models_count": _count_files(models_dir, ".txt"),
    }


def _count_files(path, ext):
    if not os.path.isdir(path):
        return 0
    n = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(ext):
                n += 1
    return n