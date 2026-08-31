from __future__ import annotations

from dataclasses import dataclass, replace

from core.config import Settings


@dataclass(frozen=True)
class SportPolicy:
    min_edge: float
    max_edge: float
    min_confidence: int
    min_odds: float
    max_odds: float
    max_tips: int
    max_stake_pct: float
    max_sport_exposure_pct: float


POLICIES: dict[str, SportPolicy] = {
    # The football module already applies its raw-edge guard.  The risk layer
    # works with a lower confidence-bound edge after uncertainty is deducted,
    # so its 4% threshold is aligned with the final professional value filter
    # instead of applying the same 6% hurdle twice.
    "football": SportPolicy(.04, .18, 68, 1.35, 4.00, 5, .0075, .020),
    "baseball": SportPolicy(.04, .18, 65, 1.35, 4.50, 5, .0100, .025),
    "basketball": SportPolicy(.06, .16, 68, 1.30, 3.50, 4, .0075, .020),
    "hockey": SportPolicy(.07, .16, 70, 1.35, 4.00, 3, .0060, .015),
    "mma": SportPolicy(.09, .15, 74, 1.35, 3.00, 2, .0050, .010),
    "nfl": SportPolicy(.07, .16, 70, 1.30, 3.50, 3, .0060, .015),
    "tennis": SportPolicy(.07, .16, 70, 1.30, 3.50, 3, .0060, .015),
    "esports": SportPolicy(.08, .15, 72, 1.35, 3.25, 2, .0050, .010),
}

DEFAULT_POLICY = SportPolicy(.08, .15, 70, 1.35, 3.50, 3, .0050, .015)


def sport_policy(sport: str) -> SportPolicy:
    return POLICIES.get(str(sport or "").strip().lower(), DEFAULT_POLICY)


def settings_for_sport(settings: Settings, sport: str) -> Settings:
    policy = sport_policy(sport)
    return replace(
        settings,
        min_edge=policy.min_edge,
        max_edge=policy.max_edge,
        max_odds=policy.max_odds,
        max_stake_pct=min(settings.max_stake_pct, policy.max_stake_pct),
    )


