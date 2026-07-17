from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_clv_results_v15_49.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_clv_storage_learning_v15_49(clv_rows):

    OUTPUT.parent.mkdir(exist_ok=True)

    rows = []

    for row in clv_rows:
        item = {
            "created_at": now(),
            "source_hash": row.get("source_hash"),
            "event": row.get("event"),
            "opening_odds": row.get("opening_odds"),
            "closing_odds": row.get("closing_odds"),
            "clv_percent": row.get("clv_percent"),
            "result": row.get("result"),
            "target": row.get("target"),
        }
        rows.append(item)

    if rows:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    return {
        "version": "v15.49",
        "created_at": now(),
        "clv_records_saved": len(rows),
        "learning_rows_added": len(rows),
        "output": str(OUTPUT),
        "status": "READY" if rows else "BUILDING",
    }
