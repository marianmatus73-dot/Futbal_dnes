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


def run_closing_features_bridge_v15_48(opening_rows):

    feature_rows = load_rows(
        "exports/history_football_features.csv"
    )

    closing_map = {
        row.get("source_hash"): row.get("odds")
        for row in feature_rows
        if row.get("source_hash")
    }

    output = []

    for row in opening_rows:
        item = dict(row)
        item["closing_odds"] = closing_map.get(
            row.get("source_hash")
        )
        output.append(item)

    return {
        "version": "v15.48",
        "created_at": now(),
        "records": len(output),
        "closing_odds_found": sum(
            1 for r in output if r.get("closing_odds")
        ),
        "status": "READY" if output else "BUILDING",
    }, output
