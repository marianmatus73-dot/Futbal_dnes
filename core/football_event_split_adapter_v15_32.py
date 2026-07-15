from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_event(event: str):
    if not event:
        return "", ""

    text = event.strip()

    separators = [
        " vs ",
        " v ",
        " - ",
        " – ",
        " @ ",
    ]

    for sep in separators:
        if sep in text:
            parts = [p.strip() for p in text.split(sep, 1)]
            return parts[0], parts[1]

    return "", ""


def enrich_postmatch_dataset(
    source="exports/history_football_postmatch_dataset_v14.csv"
):
    path = Path(source)

    if not path.exists():
        return []

    output = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)

        for row in rows:
            home, away = split_event(row.get("event", ""))

            row["home_team"] = home
            row["away_team"] = away

            output.append(row)

    return output


def run_event_split_adapter_v15_32():
    rows = enrich_postmatch_dataset()

    report = {
        "version": "v15.32",
        "created_at": _now(),
        "postmatch_rows": len(rows),
        "team_rows_ready": sum(
            1 for r in rows
            if r.get("home_team") and r.get("away_team")
        ),
        "status": "READY" if rows else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)

    Path("exports/football_event_split_adapter_v15_32.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, rows
