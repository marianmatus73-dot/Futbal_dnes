from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_closing_odds_database_join_engine_v15_19(
    *,
    matches: int = 0,
    snapshots: int = 0,
    joins_executed: int = 0,
    closing_written: int = 0,
    clv_calculated: int = 0,
):
    join_rate = round((joins_executed / matches) * 100, 1) if matches else 0.0
    closing_rate = round((closing_written / matches) * 100, 1) if matches else 0.0

    blockers = []

    if joins_executed < matches:
        blockers.append("Database join incomplete")

    if closing_written < matches:
        blockers.append("Closing odds missing")

    if clv_calculated < closing_written:
        blockers.append("CLV calculation incomplete")

    if snapshots == 0:
        blockers.append("No market snapshot source")

    return {
        "version": "v15.19",
        "created_at": _now(),
        "matches": matches,
        "snapshots": snapshots,
        "joins_executed": joins_executed,
        "closing_written": closing_written,
        "clv_calculated": clv_calculated,
        "join_coverage_percent": join_rate,
        "closing_coverage_percent": closing_rate,
        "status": "READY" if closing_rate >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_closing_odds_database_join_engine_v15_19(
    report: dict,
    export_dir: str = "exports",
):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_odds_database_join_engine_v15_19.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_closing_odds_database_join_engine_v15_19(**kwargs):
    report = build_closing_odds_database_join_engine_v15_19(**kwargs)
    export_closing_odds_database_join_engine_v15_19(report)
    return report
