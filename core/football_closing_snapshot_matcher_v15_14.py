from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_snapshot_matcher_v15_14(
    *,
    matches: int = 0,
    snapshots: int = 0,
    closing_matched: int = 0,
) -> dict:
    coverage = (
        round((closing_matched / matches) * 100, 1)
        if matches
        else 0.0
    )

    blockers = []

    if closing_matched < matches:
        blockers.append("Closing snapshot not matched for all matches")

    if snapshots == 0:
        blockers.append("No market snapshots available")

    return {
        "version": "v15.14",
        "created_at": _now(),
        "matches": matches,
        "snapshots": snapshots,
        "closing_matched": closing_matched,
        "coverage_percent": coverage,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_snapshot_matcher_v15_14(
    report: dict,
    export_dir: str = "exports",
):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_snapshot_matcher_v15_14.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_snapshot_matcher_v15_14(**kwargs) -> dict:
    report = build_snapshot_matcher_v15_14(**kwargs)
    export_snapshot_matcher_v15_14(report)
    return report
