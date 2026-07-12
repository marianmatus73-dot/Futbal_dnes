from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_data_capture_v15_9(
    *,
    samples: int = 0,
    closing_odds_samples: int = 0,
    xg_samples: int = 0,
    market_snapshots: int = 0,
) -> dict:
    closing_coverage = (
        round(closing_odds_samples / samples * 100, 1)
        if samples else 0.0
    )

    xg_coverage = (
        round(xg_samples / samples * 100, 1)
        if samples else 0.0
    )

    quality_score = 0

    if market_snapshots > 0:
        quality_score += 20
    if closing_odds_samples >= samples and samples > 0:
        quality_score += 20
    if xg_samples >= samples and samples > 0:
        quality_score += 20
    if samples >= 20:
        quality_score += 20
    if quality_score >= 80:
        quality_score += 20

    blockers = []

    if closing_coverage < 90:
        blockers.append("Closing odds capture incomplete")

    if xg_coverage < 90:
        blockers.append("xG capture incomplete")

    if samples < 20:
        blockers.append("Not enough samples for Meta AI")

    return {
        "version": "v15.9",
        "created_at": _now(),
        "samples": samples,
        "closing_odds": {
            "samples": closing_odds_samples,
            "coverage": closing_coverage,
        },
        "xg": {
            "samples": xg_samples,
            "coverage": xg_coverage,
        },
        "market_snapshots": market_snapshots,
        "quality_score": min(quality_score, 100),
        "meta_ai_ready": (
            samples >= 20 and quality_score >= 80
        ),
        "status": (
            "READY"
            if samples >= 20 and quality_score >= 80
            else "BUILDING"
        ),
        "blockers": blockers,
    }


def export_data_capture_v15_9(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_data_capture_v15_9.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (directory / "football_data_capture_v15_9.txt").write_text(
        "\n".join(
            [
                "=== FOOTBALL DATA CAPTURE ENGINE V15.9 ===",
                "",
                f"Samples: {report['samples']}",
                f"Quality: {report['quality_score']}/100",
                f"Closing odds coverage: {report['closing_odds']['coverage']}%",
                f"xG coverage: {report['xg']['coverage']}%",
                f"Meta AI ready: {report['meta_ai_ready']}",
                f"Status: {report['status']}",
                "",
                "BLOCKERS",
                *[f"- {x}" for x in report["blockers"]],
            ]
        ),
        encoding="utf-8",
    )


def run_data_capture_v15_9(**kwargs) -> dict:
    report = build_data_capture_v15_9(**kwargs)
    export_data_capture_v15_9(report)
    return report
