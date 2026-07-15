from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_team(name):
    if not name:
        return ""

    value = name.lower()
    replacements = [
        "football club",
        " fc",
        "cf",
        "afc",
        "sc",
    ]

    for item in replacements:
        value = value.replace(item, "")

    value = re.sub(r"[^a-z0-9]", "", value)

    aliases = {
        "manchesterunited": "manutd",
        "manutd": "manutd",
        "internazionale": "inter",
        "intermilan": "inter",
        "psg": "parissaintgermain",
    }

    return aliases.get(value, value)


def team_match(a, b):
    return normalize_team(a) == normalize_team(b)


def resolve_smart_matches(
    join_ready_rows,
    source="exports/history_football_postmatch_dataset_v14.csv",
):
    path = Path(source)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        matches = list(csv.DictReader(file))

    resolved = []

    for row in join_ready_rows:
        for match in matches:
            direct = (
                team_match(row.get("home_team"), match.get("home_team"))
                and team_match(row.get("away_team"), match.get("away_team"))
            )

            reverse = (
                team_match(row.get("home_team"), match.get("away_team"))
                and team_match(row.get("away_team"), match.get("home_team"))
            )

            if direct or reverse:
                resolved.append({
                    "snapshot": row,
                    "match": match,
                    "confidence": 1.0 if direct else 0.9,
                })
                break

    return resolved


def run_smart_match_resolver_v15_29(join_ready_rows=None):
    join_ready_rows = join_ready_rows or []

    resolved = resolve_smart_matches(join_ready_rows)

    report = {
        "version": "v15.29",
        "created_at": _now(),
        "input_rows": len(join_ready_rows),
        "matched_rows": len(resolved),
        "status": "READY" if resolved else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_smart_match_resolver_v15_29.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, resolved
