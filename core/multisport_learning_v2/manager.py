from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dashboard import export_dashboard
from .meta import adaptive_weights, recommendations
from .metrics import calculate_metrics, load_sport_rows
from .schema import SUPPORTED_SPORTS, detect_sport_bets_schema


class MultisportLearningV2Manager:
    def __init__(
        self,
        database: str | Path,
        table: str = "sport_bets",
    ) -> None:
        self.database = Path(database)
        self.table = table

    def run_all(
        self,
        sports: Iterable[str] | None = None,
        export_dir: str | Path = "exports",
    ) -> dict[str, Any]:
        schema = detect_sport_bets_schema(self.database, self.table)
        selected = list(sports or SUPPORTED_SPORTS)
        sport_results: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for sport in selected:
            try:
                rows = load_sport_rows(self.database, schema, sport)
                sport_results[sport] = calculate_metrics(rows)
                sport_results[sport]["status"] = "READY"
            except Exception as exc:
                errors.append(
                    f"{sport}: {type(exc).__name__}: {exc}"
                )
                sport_results[sport] = {
                    "status": "FAILED",
                    "error": str(exc),
                }

        weights = adaptive_weights(sport_results)
        recs = recommendations(sport_results)

        result = {
            "version": "V2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_table": schema.table,
            "sports": sport_results,
            "adaptive_weights": weights,
            "recommendations": recs,
            "sports_completed": len(sport_results),
            "sports_ready": sum(
                1
                for item in sport_results.values()
                if item.get("status") == "READY"
            ),
            "errors": errors,
            "status": "READY" if not errors else "PARTIAL",
        }

        result["artifacts"] = export_dashboard(result, export_dir)
        return result
