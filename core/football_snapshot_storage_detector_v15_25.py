from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scan_storage():
    patterns = [
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "*.json",
        "*.csv",
        "*.parquet",
    ]

    files = []
    for pattern in patterns:
        files.extend(Path(".").rglob(pattern))

    return list(dict.fromkeys(files))


def inspect_sqlite(path):
    result = []

    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )

        tables = [row[0] for row in cur.fetchall()]

        for table in tables:
            if any(
                word in table.lower()
                for word in ("snapshot", "market", "odds", "football")
            ):
                result.append({
                    "file": str(path),
                    "table": table,
                })

        conn.close()

    except Exception:
        pass

    return result


def run_snapshot_storage_detector_v15_25():
    files = scan_storage()
    matches = []

    for file in files:
        if file.suffix in (".db", ".sqlite", ".sqlite3"):
            matches.extend(inspect_sqlite(file))

        elif any(
            word in file.name.lower()
            for word in ("snapshot", "market", "odds", "football")
        ):
            matches.append({
                "file": str(file),
                "table": None,
            })

    report = {
        "version": "v15.25",
        "created_at": _now(),
        "files_scanned": len(files),
        "storage_matches": matches,
        "status": "FOUND" if matches else "SEARCHING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_snapshot_storage_detector_v15_25.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report
