
"""
V16.01.5 Snapshot Storage Module

Creates a simple storage layer for collected football market snapshots.

Safe extension:
- does not modify main.py
- prepares data source for CLV pipeline
"""

import json
from pathlib import Path
from datetime import datetime, timezone


STORAGE_FILE = Path("data/football_snapshots.json")


def save_snapshots(rows):
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "v16.01.5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": rows,
    }

    STORAGE_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    return {
        "records_saved": len(rows),
        "file": str(STORAGE_FILE),
        "status": "READY",
    }


def load_snapshots():
    if not STORAGE_FILE.exists():
        return []

    payload = json.loads(
        STORAGE_FILE.read_text(encoding="utf-8")
    )

    return payload.get("records", [])


if __name__ == "__main__":
    test_rows = [
        {
            "event_id": "storage_test_001",
            "opening_odds": 2.10,
            "closing_odds": 1.90,
            "home_team": "Team A",
            "away_team": "Team B",
        }
    ]

    result = save_snapshots(test_rows)
    loaded = load_snapshots()

    print("=== V16.01.5 SNAPSHOT STORAGE TEST ===")
    print(result)
    print({
        "records_loaded": len(loaded),
        "status": "READY"
    })
