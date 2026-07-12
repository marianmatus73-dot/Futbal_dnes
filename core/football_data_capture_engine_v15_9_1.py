from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_data_capture_v15_9_1(
    *,
    samples: int = 0,
    elo_available: bool = False,
    form_available: bool = False,
    market_available: bool = False,
    closing_odds_available: bool = False,
    xg_available: bool = False,
) -> dict:
    features = {
        "elo": elo_available,
        "form": form_available,
        "market": market_available,
        "closing_odds": closing_odds_available,
        "xg": xg_available,
    }

    quality_score = sum(
        20 for value in features.values() if value
    )

    blockers = [
        f"Missing {name}"
        for name, value in features.items()
        if not value
    ]

    if samples < 20:
        blockers.append(
            f"Sample size {samples}/20 for Meta AI"
        )

    return {
        "version": "v15.9.1",
        "created_at": _now(),
        "samples": samples,
        "features": features,
        "quality_score": quality_score,
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


def export_data_capture_v15_9_1(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_data_capture_v15_9_1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_data_capture_v15_9_1(**kwargs) -> dict:
    report = build_data_capture_v15_9_1(**kwargs)
    export_data_capture_v15_9_1(report)
    return report
