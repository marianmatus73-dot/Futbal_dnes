from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_ai_health_report_v15(
    *,
    dataset_samples: int = 0,
    meta_samples: int = 0,
    meta_milestone: int = 20,
    explainability_rows: int = 0,
    top_feature: str | None = None,
    missing_features: list[str] | None = None,
) -> dict:
    missing_features = missing_features or []

    return {
        "version": "v15.2",
        "created_at": _now(),
        "model_maturity": (
            "READY"
            if dataset_samples >= 100
            else "EARLY"
        ),
        "training_samples": dataset_samples,
        "meta_ai": {
            "samples": meta_samples,
            "milestone": meta_milestone,
            "status": (
                "UNLOCKED"
                if meta_samples >= meta_milestone
                else "LOCKED"
            ),
        },
        "weight_tuning": {
            "enabled": dataset_samples >= 100,
            "reason": (
                None
                if dataset_samples >= 100
                else "Need 100+ settled samples"
            ),
        },
        "explainability_records": explainability_rows,
        "top_feature": top_feature or "n/a",
        "current_weights": {
            "elo": 0.25,
            "xg": 0.25,
            "form": 0.20,
            "market": 0.20,
            "context": 0.10,
        },
        "warnings": [
            "Small sample size"
            if dataset_samples < 50
            else None,
            *[
                f"Missing feature: {item}"
                for item in missing_features
            ],
        ],
    }


def export_ai_health_report_v15(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_ai_health_v15.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL AI HEALTH REPORT V15.2 ===",
        "",
        f"Model maturity: {report['model_maturity']}",
        f"Training samples: {report['training_samples']}",
        f"Explainability records: {report['explainability_records']}",
        f"Top feature: {report['top_feature']}",
        "",
        f"Meta AI status: {report['meta_ai']['status']}",
        f"Weight tuning: {report['weight_tuning']['enabled']}",
    ]

    (directory / "football_ai_health_v15.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_ai_health_report_v15(**kwargs) -> dict:
    report = build_ai_health_report_v15(**kwargs)
    export_ai_health_report_v15(report)
    return report
