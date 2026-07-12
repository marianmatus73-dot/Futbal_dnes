from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_xg_report_v15_7(
    *,
    total_samples: int = 0,
    xg_samples: int = 0,
    xg_history_rows: int = 0,
) -> dict:
    coverage = 0.0
    if total_samples:
        coverage = round((xg_samples / total_samples) * 100, 1)

    blockers = []

    if xg_samples < total_samples:
        blockers.append("Missing xG coverage")

    if xg_history_rows == 0:
        blockers.append("No xG history available")

    return {
        "version": "v15.7",
        "created_at": _now(),
        "total_samples": total_samples,
        "xg_samples": xg_samples,
        "xg_history_rows": xg_history_rows,
        "coverage_percent": coverage,
        "status": "READY" if coverage >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_xg_report_v15_7(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_xg_collector_v15_7.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL XG COLLECTOR V15.7 ===",
        "",
        f"Samples: {report['total_samples']}",
        f"xG samples: {report['xg_samples']}",
        f"Coverage: {report['coverage_percent']}%",
        f"Status: {report['status']}",
        "",
        "BLOCKERS",
    ]

    lines.extend(f"- {x}" for x in report["blockers"])

    (directory / "football_xg_collector_v15_7.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_xg_collector_v15_7(**kwargs) -> dict:
    report = build_xg_report_v15_7(**kwargs)
    export_xg_report_v15_7(report)
    return report
