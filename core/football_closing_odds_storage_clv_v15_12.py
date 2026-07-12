from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def calculate_clv(open_odds: float, closing_odds: float) -> float:
    if open_odds <= 0 or closing_odds <= 0:
        return 0.0
    return round(((open_odds - closing_odds) / open_odds) * 100, 3)


def build_closing_storage_v15_12(
    *,
    samples: int = 0,
    market_snapshots: int = 0,
    closing_written: int = 0,
    avg_clv: float = 0.0,
) -> dict:
    coverage = (
        round((closing_written / samples) * 100, 1)
        if samples
        else 0.0
    )

    blockers = []

    if closing_written < samples:
        blockers.append("Closing odds not written for all matches")

    if market_snapshots == 0:
        blockers.append("No market snapshots available")

    return {
        "version": "v15.12",
        "created_at": _now(),
        "samples": samples,
        "market_snapshots": market_snapshots,
        "closing_written": closing_written,
        "closing_coverage_percent": coverage,
        "avg_clv": avg_clv,
        "clv_ready": closing_written > 0,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_closing_storage_v15_12(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_closing_storage_clv_v15_12.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_closing_storage_clv_v15_12(**kwargs) -> dict:
    report = build_closing_storage_v15_12(**kwargs)
    export_closing_storage_v15_12(report)
    return report
