from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_real_closing_snapshot_finder_v15_52(
    source="exports/history_football_market_snapshots_v14.csv"
):
    rows = load_rows(source)

    groups = {}

    for row in rows:
        key = row.get("source_hash")
        if not key:
            continue

        groups.setdefault(key, []).append(row)

    selected = []

    for key, items in groups.items():
        items.sort(
            key=lambda x: x.get("captured_at", ""),
            reverse=True
        )

        selected.append(items[0])

    return {
        "version": "v15.52",
        "created_at": now(),
        "snapshots_scanned": len(rows),
        "closing_candidates": len(selected),
        "status": "READY" if selected else "BUILDING",
    }, selected
