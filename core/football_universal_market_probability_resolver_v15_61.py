from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/history_football_learning_dataset_v15_61.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize(value):
    return (value or "").strip().lower()


def pick_probability(snapshot, selection):
    s = normalize(selection)

    if "home" in s:
        return snapshot.get("market_home_probability")
    if "draw" in s:
        return snapshot.get("market_draw_probability")
    if "away" in s:
        return snapshot.get("market_away_probability")

    return snapshot.get("market_selection_probability")


def run_universal_market_probability_resolver_v15_61(
    source="exports/history_football_learning_dataset_v15_58.csv"
):
    base = load_rows(source)
    snapshots = load_rows(
        "exports/history_football_market_snapshots_v14.csv"
    )

    hash_map = {
        r.get("source_hash"): r
        for r in snapshots
        if r.get("source_hash")
    }

    event_map = {}

    for r in snapshots:
        key = (
            normalize(r.get("event")),
            normalize(r.get("selection")),
        )
        event_map.setdefault(key, r)

    hash_matches = 0
    event_matches = 0
    not_found = 0
    found = 0
    output = []

    for row in base:
        item = dict(row)
        snapshot = None
        method = None

        if row.get("source_hash") in hash_map:
            snapshot = hash_map[row.get("source_hash")]
            method = "source_hash"
            hash_matches += 1
        else:
            key = (
                normalize(row.get("event")),
                normalize(row.get("selection")),
            )
            snapshot = event_map.get(key)

            if snapshot:
                method = "event_selection"
                event_matches += 1

        if snapshot:
            item["market_probability"] = pick_probability(
                snapshot,
                row.get("selection")
            )
        else:
            item["market_probability"] = None
            not_found += 1

        item["market_probability_match_method"] = method

        if item["market_probability"] not in (None, ""):
            found += 1

        output.append(item)

    if output:
        with OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=output[0].keys()
            )
            writer.writeheader()
            writer.writerows(output)

    return {
        "version": "v15.61",
        "created_at": now(),
        "records": len(output),
        "market_probability_found": found,
        "missing": not_found,
        "hash_matches": hash_matches,
        "event_matches": event_matches,
        "output": str(OUTPUT),
        "status": "READY" if output else "BUILDING",
    }
