from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_universal_join_executor_v15_18(
    *,
    matches: int = 0,
    snapshots: int = 0,
    keys_created: int = 0,
    joins_completed: int = 0,
    closing_written: int = 0,
):
    join_rate = round((joins_completed / matches) * 100, 1) if matches else 0.0
    closing_rate = round((closing_written / matches) * 100, 1) if matches else 0.0

    blockers = []

    if joins_completed < matches:
        blockers.append("Universal joins incomplete")

    if closing_written < matches:
        blockers.append("Closing odds still missing")

    if snapshots == 0:
        blockers.append("No market snapshots")

    return {
        "version": "v15.18",
        "created_at": _now(),
        "matches": matches,
        "snapshots": snapshots,
        "keys_created": keys_created,
        "joins_completed": joins_completed,
        "closing_written": closing_written,
        "join_coverage_percent": join_rate,
        "closing_coverage_percent": closing_rate,
        "status": "READY" if closing_rate >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_universal_join_executor_v15_18(report, export_dir="exports"):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "football_universal_join_executor_v15_18.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_universal_join_executor_v15_18(**kwargs):
    report = build_universal_join_executor_v15_18(**kwargs)
    export_universal_join_executor_v15_18(report)
    return report
