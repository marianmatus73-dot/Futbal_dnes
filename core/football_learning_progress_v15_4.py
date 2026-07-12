from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _quality_points(value: bool) -> int:
    return 20 if value else 0


def build_learning_progress_v15_4(
    *,
    settled_samples: int = 0,
    elo_available: bool = False,
    form_available: bool = False,
    market_available: bool = False,
    xg_available: bool = False,
    closing_odds_available: bool = False,
) -> dict:
    quality = {
        "elo": {
            "available": elo_available,
            "score": _quality_points(elo_available),
        },
        "form": {
            "available": form_available,
            "score": _quality_points(form_available),
        },
        "market": {
            "available": market_available,
            "score": _quality_points(market_available),
        },
        "xg": {
            "available": xg_available,
            "score": _quality_points(xg_available),
        },
        "closing_odds": {
            "available": closing_odds_available,
            "score": _quality_points(closing_odds_available),
        },
    }

    quality_score = sum(
        item["score"]
        for item in quality.values()
    )

    blockers = []

    if settled_samples < 20:
        blockers.append("Small sample size")

    if not xg_available:
        blockers.append("Missing xG history")

    if not closing_odds_available:
        blockers.append("Missing closing odds")

    return {
        "version": "v15.4",
        "created_at": _now(),
        "settled_samples": settled_samples,
        "meta_ai_progress": round(
            min(settled_samples / 20 * 100, 100),
            1,
        ),
        "weight_optimizer_progress": round(
            min(settled_samples / 100 * 100, 100),
            1,
        ),
        "data_quality": quality,
        "data_quality_score": quality_score,
        "learning_readiness": (
            "READY"
            if quality_score >= 80 and settled_samples >= 100
            else "EARLY"
        ),
        "blockers": blockers,
    }


def export_learning_progress_v15_4(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (directory / "football_learning_progress_v15_4.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL LEARNING PROGRESS V15.4 ===",
        "",
        f"Training samples: {report['settled_samples']}",
        f"Meta AI progress: {report['meta_ai_progress']}%",
        f"Weight Optimizer progress: {report['weight_optimizer_progress']}%",
        "",
        f"Data quality score: {report['data_quality_score']}/100",
        "",
        "DATA QUALITY MAP",
    ]

    for key, item in report["data_quality"].items():
        status = "READY" if item["available"] else "MISSING"
        lines.append(
            f"{key.upper()}: {item['score']}% - {status}"
        )

    lines.extend(
        [
            "",
            f"Learning readiness: {report['learning_readiness']}",
            "",
            "Blockers:",
        ]
    )

    for blocker in report["blockers"]:
        lines.append(f"- {blocker}")

    (directory / "football_learning_progress_v15_4.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_learning_progress_v15_4(**kwargs) -> dict:
    report = build_learning_progress_v15_4(**kwargs)
    export_learning_progress_v15_4(report)
    return report
