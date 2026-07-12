from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_football_insights_v15(
    *,
    dataset_report: dict | None = None,
    feature_report: dict | None = None,
    explainability_rows: int = 0,
) -> dict:
    dataset_report = dataset_report or {}
    feature_report = feature_report or {}

    samples = int(
        dataset_report.get(
            "training_samples",
            0,
        )
    )

    ranking = feature_report.get(
        "feature_ranking",
        [],
    )

    top_signal = (
        ranking[0].get("feature")
        if ranking
        else None
    )

    return {
        "version": "v15.1",
        "created_at": _now(),
        "training_samples": samples,
        "model_status": (
            "READY_FOR_REVIEW"
            if samples >= 100
            else "EARLY"
        ),
        "automatic_weight_tuning": (
            samples >= 100
        ),
        "weight_lock_reason": (
            None
            if samples >= 100
            else "Need 100+ settled football samples"
        ),
        "current_weights": {
            "elo": 0.25,
            "xg": 0.25,
            "form": 0.20,
            "market": 0.20,
            "context": 0.10,
        },
        "top_signals": (
            [top_signal]
            if top_signal
            else []
        ),
        "feature_importance_warning": (
            feature_report.get("warning")
        ),
        "explainability_rows": explainability_rows,
    }


def export_football_insights_v15(
    report: dict,
    export_dir: str = "exports",
) -> None:
    directory = Path(export_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = directory / "football_insights_v15.json"
    txt_path = directory / "football_insights_v15.txt"

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== FOOTBALL MODEL INSIGHTS V15.1 ===",
        "",
        f"Training samples: {report['training_samples']}",
        f"Model status: {report['model_status']}",
        f"Explainability rows: {report['explainability_rows']}",
        "",
        "Current weights:",
    ]

    for key, value in report["current_weights"].items():
        lines.append(
            f"- {key}: {value:.2f}"
        )

    if report["weight_lock_reason"]:
        lines.extend(
            [
                "",
                "Weight tuning locked:",
                report["weight_lock_reason"],
            ]
        )

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_football_insights_v15(
    *,
    dataset_report: dict | None = None,
    feature_report: dict | None = None,
    explainability_rows: int = 0,
) -> dict:
    report = build_football_insights_v15(
        dataset_report=dataset_report,
        feature_report=feature_report,
        explainability_rows=explainability_rows,
    )

    export_football_insights_v15(report)

    return report
