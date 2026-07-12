from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_real_closing_snapshot_extractor_v15_15(
    *,
    matches: int = 0,
    snapshots: int = 0,
    closing_extracted: int = 0,
    clv_ready: int = 0,
) -> dict:
    coverage = (
        round((closing_extracted / matches) * 100, 1)
        if matches
        else 0.0
    )

    blockers = []

    if closing_extracted < matches:
        blockers.append("Closing odds extraction incomplete")

    if clv_ready < closing_extracted:
        blockers.append("CLV calculation incomplete")

    if snapshots == 0:
        blockers.append("No market snapshots")

    return {
        "version": "v15.15",
        "created_at": _now(),
        "matches": matches,
        "snapshots": snapshots,
        "closing_extracted": closing_extracted,
        "clv_ready": clv_ready,
        "coverage_percent": coverage,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_real_closing_snapshot_extractor_v15_15(
    report: dict,
    export_dir: str = "exports",
):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_real_closing_snapshot_extractor_v15_15.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_real_closing_snapshot_extractor_v15_15(**kwargs) -> dict:
    report = build_real_closing_snapshot_extractor_v15_15(**kwargs)
    export_real_closing_snapshot_extractor_v15_15(report)
    return report
