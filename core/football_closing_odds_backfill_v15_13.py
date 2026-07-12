from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_closing_backfill_v15_13(
    *,
    postmatch_samples: int = 0,
    market_snapshots: int = 0,
    closing_recovered: int = 0,
) -> dict:
    coverage = (
        round((closing_recovered / postmatch_samples) * 100, 1)
        if postmatch_samples
        else 0.0
    )

    blockers = []

    if closing_recovered < postmatch_samples:
        blockers.append("Some matches still missing closing odds")

    if market_snapshots == 0:
        blockers.append("No market snapshots available")

    return {
        "version": "v15.13",
        "created_at": _now(),
        "postmatch_samples": postmatch_samples,
        "market_snapshots": market_snapshots,
        "closing_recovered": closing_recovered,
        "coverage_percent": coverage,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_closing_backfill_v15_13(report: dict, export_dir: str = "exports"):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_odds_backfill_v15_13.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_closing_backfill_v15_13(**kwargs) -> dict:
    report = build_closing_backfill_v15_13(**kwargs)
    export_closing_backfill_v15_13(report)
    return report
