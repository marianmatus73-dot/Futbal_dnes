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


def normalize(v):
    return (v or "").strip().lower()


def overlap(rows_a, rows_b, key_fields):
    a = {
        tuple(normalize(r.get(k)) for k in key_fields)
        for r in rows_a
    }
    b = {
        tuple(normalize(r.get(k)) for k in key_fields)
        for r in rows_b
    }
    return len(a & b)


def run_dataset_key_diagnostics_v15_62(
    learning="exports/history_football_learning_dataset_v15_58.csv",
    snapshots="exports/history_football_market_snapshots_v14.csv",
):
    learning_rows = load_rows(learning)
    snapshot_rows = load_rows(snapshots)

    tests = {
        "source_hash": ["source_hash"],
        "event": ["event"],
        "teams": ["home_team", "away_team"],
        "event_selection": ["event", "selection"],
        "teams_selection": ["home_team", "away_team", "selection"],
        "teams_bookmaker_selection": [
            "home_team",
            "away_team",
            "bookmaker",
            "selection",
        ],
    }

    results = {
        name: overlap(learning_rows, snapshot_rows, fields)
        for name, fields in tests.items()
    }

    best = max(results, key=results.get) if results else None

    return {
        "version": "v15.62",
        "created_at": now(),
        "learning_rows": len(learning_rows),
        "snapshot_rows": len(snapshot_rows),
        "overlap_tests": results,
        "best_join_key": best,
        "status": "READY",
    }
