from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_correct_dataset(
    source="exports/history_football_dataset_v15.csv"
):
    path = Path(source)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def run_correct_snapshot_source_loader_v15_39():
    rows = load_correct_dataset()

    return {
        "version": "v15.39",
        "created_at": now(),
        "source": "exports/history_football_dataset_v15.csv",
        "rows_loaded": len(rows),
        "status": "READY" if rows else "BUILDING",
    }, rows
