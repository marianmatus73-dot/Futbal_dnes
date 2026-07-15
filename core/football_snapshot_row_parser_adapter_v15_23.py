from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_snapshot_row(row: dict) -> dict:
    return {
        "event_id": row.get("event_id"),
        "league": row.get("league") or row.get("sport_key"),
        "home_team": row.get("home_team") or row.get("home"),
        "away_team": row.get("away_team") or row.get("away"),
        "kickoff_time": row.get("commence_time") or row.get("start_time"),
        "odds": row.get("odds"),
        "bookmaker": row.get("bookmaker"),
    }


def create_fingerprint(row: dict) -> str:
    return "|".join(
        [
            str(row.get("league") or ""),
            str(row.get("home_team") or ""),
            str(row.get("away_team") or ""),
            str(row.get("kickoff_time") or ""),
        ]
    )


def run_snapshot_row_parser_adapter_v15_23(
    snapshots: list[dict] | None = None,
):
    snapshots = snapshots or []

    parsed = []
    for item in snapshots:
        row = normalize_snapshot_row(item)
        row["fingerprint"] = create_fingerprint(row)
        if row["home_team"] and row["away_team"] and row["kickoff_time"]:
            row["join_ready"] = True
        else:
            row["join_ready"] = False
        parsed.append(row)

    join_ready = sum(1 for r in parsed if r["join_ready"])

    report = {
        "version": "v15.23",
        "created_at": _now(),
        "snapshots": len(snapshots),
        "parsed": len(parsed),
        "join_ready": join_ready,
        "status": "READY" if join_ready else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_snapshot_row_parser_adapter_v15_23.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, parsed
