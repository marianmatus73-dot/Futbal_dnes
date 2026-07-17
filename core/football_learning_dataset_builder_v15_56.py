from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_56.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_learning_dataset_builder_v15_56(
    source="exports/history_football_clv_results_v15_49.csv"
):
    path = Path(source)

    if not path.exists():
        return {
            "version": "v15.56",
            "records": 0,
            "status": "BUILDING",
        }

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    dataset = []

    for row in rows:
        dataset.append({
            "created_at": now(),
            "source_hash": row.get("source_hash"),
            "event": row.get("event"),
            "opening_odds": row.get("opening_odds"),
            "closing_odds": row.get("closing_odds"),
            "clv_percent": row.get("clv_percent"),
            "result": row.get("result"),
            "target": row.get("target"),
        })

    if dataset:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=dataset[0].keys()
            )
            writer.writeheader()
            writer.writerows(dataset)

    return {
        "version": "v15.56",
        "created_at": now(),
        "records": len(dataset),
        "output": str(OUTPUT),
        "status": "READY" if dataset else "BUILDING",
    }
