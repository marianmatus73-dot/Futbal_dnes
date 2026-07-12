from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_learning_progress_v15(
    *,
    settled_samples: int = 0,
    elo_available: bool = False,
    form_available: bool = False,
    xg_available: bool = False,
    closing_odds_available: bool = False,
    market_snapshots_available: bool = False,
) -> dict:
    meta_progress = min(
        settled_samples / 20 * 100,
        100,
    )

    optimizer_progress = min(
        settled_samples / 100 * 100,
        100,
    )

    quality = {
        "elo": elo_available,
        "form": form_available,
        "xg": xg_available,
        "closing_odds": closing_odds_available,
        "market_snapshots": market_snapshots_available,
    }

    return {
        "version": "v15.3",
        "created_at": _now(),
        "settled_samples": settled_samples,
        "meta_ai": {
            "progress_percent": round(meta_progress, 1),
            "milestone": 20,
            "status": (
                "READY_FOR_TRAINING"
                if settled_samples >= 20
                else "LOCKED"
            ),
        },
        "weight_optimizer": {
            "progress_percent": round(optimizer_progress, 1),
            "milestone": 100,
            "status": (
                "READY_FOR_REVIEW"
                if settled_samples >= 100
                else "LOCKED"
            ),
        },
        "data_quality": quality,
        "learning_readiness": (
            "READY"
            if settled_samples >= 100
            else "EARLY"
        ),
    }


def export_learning_progress_v15(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (directory / "football_learning_progress_v15.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL LEARNING PROGRESS V15.3 ===",
        "",
        f"Settled samples: {report['settled_samples']}",
        "",
        "Meta AI:",
        f"- Progress: {report['meta_ai']['progress_percent']}%",
        f"- Status: {report['meta_ai']['status']}",
        "",
        "Weight Optimizer:",
        f"- Progress: {report['weight_optimizer']['progress_percent']}%",
        f"- Status: {report['weight_optimizer']['status']}",
        "",
        f"Learning readiness: {report['learning_readiness']}",
    ]

    (directory / "football_learning_progress_v15.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_learning_progress_v15(**kwargs) -> dict:
    report = build_learning_progress_v15(**kwargs)
    export_learning_progress_v15(report)
    return report
