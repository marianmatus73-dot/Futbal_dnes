from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_team(name):
    if not name:
        return ""
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )


def resolve_matches(join_ready_rows, source="exports/history_football_postmatch_dataset_v14.csv"):
    path = Path(source)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        matches = list(csv.DictReader(file))

    resolved = []

    for row in join_ready_rows:
        for match in matches:
            home_match = normalize_team(match.get("home_team"))
            away_match = normalize_team(match.get("away_team"))

            if (
                normalize_team(row.get("home_team")) == home_match
                and normalize_team(row.get("away_team")) == away_match
            ):
                resolved.append({
                    "snapshot": row,
                    "match": match,
                    "matched": True,
                })
                break

    return resolved


def run_universal_match_resolver_v15_28(join_ready_rows=None):
    join_ready_rows = join_ready_rows or []

    resolved = resolve_matches(join_ready_rows)

    report = {
        "version": "v15.28",
        "created_at": _now(),
        "join_ready_rows": len(join_ready_rows),
        "matched_rows": len(resolved),
        "status": "READY" if resolved else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_universal_match_resolver_v15_28.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, resolved
