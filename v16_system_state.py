"""
Shared state container for the integrated V16.00–V16.16 cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class V16SystemState:
    cycle_id: int = 1
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    status: str = "INITIALIZED"

    def record(self, stage: str, result: dict[str, Any]) -> None:
        self.stages[stage] = result

    def record_error(self, stage: str, error: Exception) -> None:
        self.errors.append({
            "stage": stage,
            "error": f"{type(error).__name__}: {error}",
        })

    def get(self, stage: str, key: str, default: Any = None) -> Any:
        return self.stages.get(stage, {}).get(key, default)

    def summary(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "stages_completed": len(self.stages),
            "errors": self.errors,
            "status": self.status,
            "stages": self.stages,
        }
