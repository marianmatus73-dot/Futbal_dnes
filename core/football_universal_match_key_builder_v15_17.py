from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_team(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def build_match_key(
    league: str,
    home_team: str,
    away_team: str,
    kickoff_date: str,
) -> str:
    return "|".join(
        [
            normalize_team(league),
            normalize_team(home_team),
            normalize_team(away_team),
            kickoff_date,
        ]
    )


def build_universal_match_key_builder_v15_17(
    *,
    matches: int = 0,
    snapshots: int = 0,
    keys_created: int = 0,
    joins_completed: int = 0,
) -> dict:
    key_rate = (
        round((keys_created / matches) * 100, 1)
        if matches
        else 0.0
    )

    join_rate = (
        round((joins_completed / matches) * 100, 1)
        if matches
        else 0.0
    )

    blockers = []

    if keys_created < matches:
        blockers.append("Universal match keys incomplete")

    if joins_completed < matches:
        blockers.append("Market snapshot joins incomplete")

    if snapshots == 0:
        blockers.append("No market snapshots")

    return {
        "version": "v15.17",
        "created_at": _now(),
        "matches": matches,
        "snapshots": snapshots,
        "keys_created": keys_created,
        "joins_completed": joins_completed,
        "key_coverage_percent": key_rate,
        "join_coverage_percent": join_rate,
        "status": "READY" if join_rate >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_universal_match_key_builder_v15_17(
    report: dict,
    export_dir: str = "exports",
):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_universal_match_key_builder_v15_17.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_universal_match_key_builder_v15_17(**kwargs) -> dict:
    report = build_universal_match_key_builder_v15_17(**kwargs)
    export_universal_match_key_builder_v15_17(report)
    return report
