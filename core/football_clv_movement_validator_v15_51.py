from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_clv_movement_validator_v15_51(
    source="exports/history_football_clv_results_v15_49.csv"
):
    path = Path(source)

    if not path.exists():
        return {
            "version": "v15.51",
            "records_checked": 0,
            "status": "BUILDING",
        }

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    moved_up = 0
    moved_down = 0
    unchanged = 0

    for row in rows:
        try:
            opening = float(row.get("opening_odds"))
            closing = float(row.get("closing_odds"))

            if closing > opening:
                moved_up += 1
            elif closing < opening:
                moved_down += 1
            else:
                unchanged += 1

        except (TypeError, ValueError):
            unchanged += 1

    return {
        "version": "v15.51",
        "created_at": now(),
        "records_checked": len(rows),
        "closing_higher": moved_up,
        "closing_lower": moved_down,
        "unchanged": unchanged,
        "status": "READY",
    }
