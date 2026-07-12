from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_closing_odds_report_v15_6(
    *,
    total_samples: int = 0,
    closing_odds_samples: int = 0,
    market_snapshots: int = 0,
) -> dict:
    coverage = 0.0
    if total_samples > 0:
        coverage = round(
            (closing_odds_samples / total_samples) * 100,
            1,
        )

    blockers = []

    if closing_odds_samples < total_samples:
        blockers.append("Missing closing odds coverage")

    if market_snapshots == 0:
        blockers.append("No market snapshots available")

    return {
        "version": "v15.6",
        "created_at": _now(),
        "total_samples": total_samples,
        "closing_odds_samples": closing_odds_samples,
        "coverage_percent": coverage,
        "market_snapshots": market_snapshots,
        "status": (
            "READY"
            if coverage >= 90
            else "BUILDING"
        ),
        "blockers": blockers,
        "recommendation": (
            "Closing odds collector active"
            if coverage < 100
            else "Closing odds complete"
        ),
    }


def export_closing_odds_report_v15_6(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (directory / "football_closing_odds_v15_6.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL CLOSING ODDS COLLECTOR V15.6 ===",
        "",
        f"Total samples: {report['total_samples']}",
        f"Closing odds samples: {report['closing_odds_samples']}",
        f"Coverage: {report['coverage_percent']}%",
        f"Status: {report['status']}",
        "",
        "BLOCKERS",
    ]

    for blocker in report["blockers"]:
        lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            f"Recommendation: {report['recommendation']}",
        ]
    )

    (directory / "football_closing_odds_v15_6.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_closing_odds_collector_v15_6(**kwargs) -> dict:
    report = build_closing_odds_report_v15_6(**kwargs)
    export_closing_odds_report_v15_6(report)
    return report
