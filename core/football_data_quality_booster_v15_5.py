from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_data_quality_report_v15_5(
    *,
    elo_available: bool = False,
    form_available: bool = False,
    market_available: bool = False,
    xg_available: bool = False,
    closing_odds_available: bool = False,
    settled_samples: int = 0,
) -> dict:
    features = {
        "elo": elo_available,
        "form": form_available,
        "market": market_available,
        "xg": xg_available,
        "closing_odds": closing_odds_available,
    }

    quality_score = sum(
        20 for value in features.values() if value
    )

    blockers = []

    if not xg_available:
        blockers.append("Missing xG history")

    if not closing_odds_available:
        blockers.append("Missing closing odds")

    if settled_samples < 20:
        blockers.append(
            f"Small sample size ({settled_samples}/20)"
        )

    meta_training_ready = (
        quality_score >= 80
        and settled_samples >= 20
    )

    return {
        "version": "v15.5",
        "created_at": _now(),
        "quality_score": quality_score,
        "features": features,
        "blockers": blockers,
        "settled_samples": settled_samples,
        "meta_ai_training_ready": meta_training_ready,
        "status": (
            "READY"
            if meta_training_ready
            else "BUILDING"
        ),
    }


def export_data_quality_report_v15_5(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (directory / "football_data_quality_v15_5.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL DATA QUALITY BOOSTER V15.5 ===",
        "",
        f"Quality score: {report['quality_score']}/100",
        f"Settled samples: {report['settled_samples']}",
        "",
        "FEATURE STATUS",
    ]

    for name, available in report["features"].items():
        lines.append(
            f"{name.upper()}: "
            f"{'READY' if available else 'MISSING'}"
        )

    lines.extend(
        [
            "",
            f"Status: {report['status']}",
            f"Meta AI training ready: {report['meta_ai_training_ready']}",
            "",
            "BLOCKERS",
        ]
    )

    for blocker in report["blockers"]:
        lines.append(f"- {blocker}")

    (directory / "football_data_quality_v15_5.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_data_quality_booster_v15_5(**kwargs) -> dict:
    report = build_data_quality_report_v15_5(**kwargs)
    export_data_quality_report_v15_5(report)
    return report
