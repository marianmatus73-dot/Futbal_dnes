from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_57.csv")

FEATURES = [
    "elo_difference",
    "xg_home",
    "xg_away",
    "form_difference",
    "market_probability",
    "raw_edge",
    "market_overround",
    "bookmaker_count",
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_feature_enrichment_v15_57(
    source="exports/history_football_learning_dataset_v15_56.csv"
):
    rows = load_rows(source)

    enriched = []
    added = set()

    for row in rows:
        item = dict(row)

        for feature in FEATURES:
            if feature not in item:
                item[feature] = None
            if item.get(feature) is not None:
                added.add(feature)

        enriched.append(item)

    if enriched:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=enriched[0].keys())
            writer.writeheader()
            writer.writerows(enriched)

    return {
        "version": "v15.57",
        "created_at": now(),
        "records": len(enriched),
        "features_added": list(added),
        "missing_features": [x for x in FEATURES if x not in added],
        "output": str(OUTPUT),
        "status": "READY" if enriched else "BUILDING",
    }, enriched
