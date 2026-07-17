from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_59.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_market_probability_joiner_v15_59(
    source="exports/history_football_learning_dataset_v15_58.csv"
):
    base = load_rows(source)
    snapshots = load_rows(
        "exports/history_football_market_snapshots_v14.csv"
    )

    market_map = {
        row.get("source_hash"): row
        for row in snapshots
        if row.get("source_hash")
    }

    found = 0
    output = []

    for row in base:
        item = dict(row)
        snap = market_map.get(row.get("source_hash"), {})

        item["market_probability"] = (
            snap.get("market_selection_probability")
        )

        if item["market_probability"] not in (None, ""):
            found += 1

        output.append(item)

    if output:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output[0].keys())
            writer.writeheader()
            writer.writerows(output)

    return {
        "version": "v15.59",
        "created_at": now(),
        "records": len(output),
        "market_probability_found": found,
        "missing_market_probability": len(output) - found,
        "output": str(OUTPUT),
        "status": "READY" if output else "BUILDING",
    }
