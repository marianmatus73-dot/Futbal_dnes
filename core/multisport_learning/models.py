from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class SportProfile:
    name: str
    rating_system: str
    form_window: int
    supports_draw: bool
    competition_label: str
    min_training_samples: int = 100
    closing_odds_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    sport: str
    stages: dict[str, dict[str, Any]]
    status: str
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sport": self.sport,
            "stages": self.stages,
            "status": self.status,
            "errors": self.errors,
        }
