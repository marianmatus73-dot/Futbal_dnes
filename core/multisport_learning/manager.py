from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .database import connect
from .models import PipelineResult, SportProfile
from .stages import (
    settle_events,
    learn_results,
    rebuild_ratings,
    rebuild_form,
    collect_market,
    build_dataset,
    evaluate,
    ai_health,
    maintenance,
)
from core.sports.registry import SPORT_PROFILES


class MultisportLearningManager:
    def __init__(self, database: str = "multisport_learning.db") -> None:
        self.database = database
        with connect(database):
            pass

    def run_sport(self, sport: str) -> PipelineResult:
        if sport not in SPORT_PROFILES:
            raise ValueError(f"Unsupported sport: {sport}")

        profile = SPORT_PROFILES[sport]
        stages = {}
        errors = []

        pipeline = [
            ("settlement", settle_events),
            ("result_learning", learn_results),
            ("ratings", rebuild_ratings),
            ("form", rebuild_form),
            ("market", collect_market),
            ("dataset", build_dataset),
            ("evaluation", evaluate),
            ("ai_health", ai_health),
            ("maintenance", maintenance),
        ]

        for name, func in pipeline:
            try:
                stages[name] = func(self.database, profile)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                stages[name] = {"status": "FAILED"}

        status = "READY" if not errors else "PARTIAL"
        result = PipelineResult(
            sport=sport,
            stages=stages,
            status=status,
            errors=errors,
        )

        with connect(self.database) as conn:
            conn.execute(
                """
                INSERT INTO ml_cycle_history (sport, status, stages_json, errors_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    sport,
                    status,
                    json.dumps(stages, ensure_ascii=False),
                    json.dumps(errors, ensure_ascii=False),
                ),
            )
            conn.commit()

        return result

    def run_all(self, sports: Iterable[str] | None = None) -> dict:
        selected = list(sports or SPORT_PROFILES.keys())
        results = {sport: self.run_sport(sport).as_dict() for sport in selected}
        return {
            "sports": results,
            "sports_completed": len(results),
            "ready": sum(1 for item in results.values() if item["status"] == "READY"),
            "status": "READY" if all(
                item["status"] == "READY" for item in results.values()
            ) else "PARTIAL",
        }

    def export_report(self, destination: str | Path) -> dict:
        result = self.run_all()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result
