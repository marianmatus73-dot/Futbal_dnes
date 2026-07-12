from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_closing_odds_capture_v15_10(
    *,
    samples: int = 0,
    opening_odds_samples: int = 0,
    closing_odds_samples: int = 0,
    market_snapshots: int = 0,
) -> dict:
    coverage = (
        round((closing_odds_samples / samples) * 100, 1)
        if samples
        else 0.0
    )

    blockers = []

    if opening_odds_samples < samples:
        blockers.append("Missing opening odds records")

    if closing_odds_samples < samples:
        blockers.append("Missing closing odds history")

    if market_snapshots == 0:
        blockers.append("Missing market snapshots")

    return {
        "version": "v15.10",
        "created_at": _now(),
        "samples": samples,
        "opening_odds_samples": opening_odds_samples,
        "closing_odds_samples": closing_odds_samples,
        "market_snapshots": market_snapshots,
        "closing_coverage_percent": coverage,
        "clv_available": closing_odds_samples > 0,
        "status": (
            "READY"
            if coverage >= 90
            else "BUILDING"
        ),
        "blockers": blockers,
    }


def export_closing_odds_capture_v15_10(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_odds_capture_v15_10.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_closing_odds_capture_v15_10(**kwargs) -> dict:
    report = build_closing_odds_capture_v15_10(**kwargs)
    export_closing_odds_capture_v15_10(report)
    return report
