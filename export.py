# export.py
import csv
import json
import os

from config import EXPORT_DEFAULT_DIR


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def export_rows(rows, columns, path, fmt):
    ensure_dir(os.path.dirname(path) or EXPORT_DEFAULT_DIR)
    if fmt == "csv":
        return _export_csv(rows, columns, path)
    if fmt == "xlsx":
        return _export_xlsx(rows, columns, path)
    if fmt == "json":
        return _export_json(rows, columns, path)
    raise ValueError(f"unknown format: {fmt}")


def _export_csv(rows, columns, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
    return path


def _export_xlsx(rows, columns, path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    for r in rows:
        ws.append([_coerce(r.get(c)) for c in columns])
    wb.save(path)
    return path


def _export_json(rows, columns, path):
    payload = [{c: r.get(c) for c in columns} for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def _coerce(v):
    if v is None:
        return ""
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)