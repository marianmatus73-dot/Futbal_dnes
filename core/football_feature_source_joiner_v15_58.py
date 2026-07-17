from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_58.csv")

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


def run_feature_source_joiner_v15_58(
    source="exports/history_football_learning_dataset_v15_56.csv"
):
    base = load_rows(source)
    features = load_rows("exports/history_football_features.csv")

    feature_map = {
        row.get("source_hash"): row
        for row in features
        if row.get("source_hash")
    }

    output = []
    found = set()

    for row in base:
        item = dict(row)
        source_row = feature_map.get(row.get("source_hash"), {})

        for feature in FEATURES:
            item[feature] = source_row.get(feature)
            if item[feature] not in (None, ""):
                found.add(feature)

        output.append(item)

    if output:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output[0].keys())
            writer.writeheader()
            writer.writerows(output)

    return {
        "version": "v15.58",
        "created_at": now(),
        "records": len(output),
        "features_found": list(found),
        "missing_features": [x for x in FEATURES if x not in found],
        "output": str(OUTPUT),
        "status": "READY" if output else "BUILDING",
    }
