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


def run_opening_odds_bridge_v15_43(resolved_matches):
    rows = load_rows("exports/history_football_dataset_v15.csv")

    by_hash = {
        r.get("source_hash"): r
        for r in rows
        if r.get("source_hash")
    }

    output = []

    for item in resolved_matches:
        post = item.get("postmatch", {})
        source = by_hash.get(post.get("source_hash"), {})

        merged = dict(post)
        merged["opening_odds"] = (
            source.get("opening_odds")
            or source.get("selected_odds")
        )

        output.append(merged)

    return {
        "version": "v15.43",
        "created_at": now(),
        "records": len(output),
        "opening_odds_found": sum(
            1 for r in output if r.get("opening_odds")
        ),
        "status": "READY" if output else "BUILDING",
    }, output
