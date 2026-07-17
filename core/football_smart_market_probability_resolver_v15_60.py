from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_60.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick_probability(snapshot, selection):
    s = (selection or "").lower()

    if "home" in s:
        return snapshot.get("market_home_probability")
    if "draw" in s:
        return snapshot.get("market_draw_probability")
    if "away" in s:
        return snapshot.get("market_away_probability")

    return snapshot.get("market_selection_probability")


def run_smart_market_probability_resolver_v15_60(
    source="exports/history_football_learning_dataset_v15_58.csv"
):
    base = load_rows(source)
    snapshots = load_rows("exports/history_football_market_snapshots_v14.csv")

    by_hash = {
        r.get("source_hash"): r
        for r in snapshots
        if r.get("source_hash")
    }

    found = 0
    output = []

    for row in base:
        item = dict(row)
        snap = by_hash.get(row.get("source_hash"))

        if snap:
            probability = pick_probability(
                snap,
                row.get("selection")
            )
        else:
            probability = None

        item["market_probability"] = probability

        if probability not in (None, ""):
            found += 1

        output.append(item)

    if output:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output[0].keys())
            writer.writeheader()
            writer.writerows(output)

    return {
        "version": "v15.60",
        "created_at": now(),
        "records": len(output),
        "market_probability_found": found,
        "missing_market_probability": len(output) - found,
        "output": str(OUTPUT),
        "status": "READY" if output else "BUILDING",
    }
