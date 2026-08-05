from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .dashboard import export_dashboard
from .meta import adaptive_weights
from .metrics import calculate, load_rows
from .schema import SUPPORTED_SPORTS, detect_schema


class MultisportLearningV2Manager:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)

    def run_all(
        self,
        export_dir: str | Path = "exports",
    ) -> dict:
        schema = detect_schema(self.database)
        sports = {}
        errors = []

        for sport in SUPPORTED_SPORTS:
            try:
                sports[sport] = calculate(
                    load_rows(self.database, schema, sport)
                )
                sports[sport]["status"] = "READY"
            except Exception as exc:
                errors.append(
                    f"{sport}: {type(exc).__name__}: {exc}"
                )
                sports[sport] = {
                    "status": "FAILED",
                    "error": str(exc),
                }

        result = {
            "version": "V2.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sports": sports,
            "adaptive_weights": adaptive_weights(sports),
            "sports_completed": len(sports),
            "sports_ready": sum(
                1
                for item in sports.values()
                if item.get("status") == "READY"
            ),
            "errors": errors,
            "status": "READY" if not errors else "PARTIAL",
        }
        result["artifacts"] = export_dashboard(result, export_dir)
        return result
