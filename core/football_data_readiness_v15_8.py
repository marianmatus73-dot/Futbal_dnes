from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_data_readiness_v15_8(
    *,
    samples: int = 0,
    elo: bool = False,
    form: bool = False,
    market: bool = False,
    closing_odds: bool = False,
    xg: bool = False,
) -> dict:
    features = {
        "elo": elo,
        "form": form,
        "market": market,
        "closing_odds": closing_odds,
        "xg": xg,
    }

    score = sum(20 for value in features.values() if value)

    blockers = [
        f"Missing {name}"
        for name, value in features.items()
        if not value
    ]

    if samples < 20:
        blockers.append(f"Sample size {samples}/20 for Meta AI")

    return {
        "version": "v15.8",
        "created_at": _now(),
        "samples": samples,
        "quality_score": score,
        "features": features,
        "meta_ai_ready": score >= 80 and samples >= 20,
        "blockers": blockers,
    }


def export_data_readiness_v15_8(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_data_readiness_v15_8.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_data_readiness_v15_8(**kwargs) -> dict:
    report = build_data_readiness_v15_8(**kwargs)
    export_data_readiness_v15_8(report)
    return report
