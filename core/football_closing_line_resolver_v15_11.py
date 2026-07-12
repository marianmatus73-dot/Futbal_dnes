from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_closing_line_resolver_v15_11(
    *,
    samples: int = 0,
    opening_odds: int = 0,
    market_snapshots: int = 0,
    closing_odds_found: int = 0,
) -> dict:
    coverage = (
        round((closing_odds_found / samples) * 100, 1)
        if samples
        else 0.0
    )

    blockers = []

    if closing_odds_found < samples:
        blockers.append("Closing line not resolved for all matches")

    if market_snapshots == 0:
        blockers.append("No market snapshots")

    return {
        "version": "v15.11",
        "created_at": _now(),
        "samples": samples,
        "opening_odds": opening_odds,
        "market_snapshots": market_snapshots,
        "closing_odds_found": closing_odds_found,
        "closing_coverage_percent": coverage,
        "clv_ready": closing_odds_found > 0,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_closing_line_resolver_v15_11(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_line_resolver_v15_11.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_closing_line_resolver_v15_11(**kwargs) -> dict:
    report = build_closing_line_resolver_v15_11(**kwargs)
    export_closing_line_resolver_v15_11(report)
    return report
