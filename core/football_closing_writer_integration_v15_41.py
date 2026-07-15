from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


OUTPUT = Path("exports/football_closing_records_v15_41.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_closing_writer_integration_v15_41(resolved_matches):

    rows = []

    for item in resolved_matches:
        snapshot = item.get("snapshot", {})
        postmatch = item.get("postmatch", {})

        rows.append({
            "created_at": now(),
            "source_hash": snapshot.get("source_hash")
                or postmatch.get("source_hash"),
            "event": snapshot.get("event")
                or postmatch.get("event"),
            "closing_odds": postmatch.get("closing_odds"),
            "closing_probability": postmatch.get("closing_probability"),
            "clv_probability": postmatch.get("clv_probability"),
            "method": item.get("method", "source_hash"),
        })

    OUTPUT.parent.mkdir(exist_ok=True)

    if rows:
        with OUTPUT.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    return {
        "version": "v15.41",
        "created_at": now(),
        "closing_written": len(rows),
        "clv_ready": sum(
            1 for row in rows
            if row.get("clv_probability")
        ),
        "output": str(OUTPUT),
        "status": "READY" if rows else "BUILDING",
    }
