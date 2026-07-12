from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass(frozen=True)
class FootballContextV15:
    competition_importance: float
    is_international: float
    is_knockout: float
    is_qualification: float
    season_stage: float
    context_reliability: float
    reason: str


INTERNATIONAL_TOKENS = (
    "uefa",
    "fifa",
    "world cup",
    "nations league",
    "champions league",
    "europa league",
    "conference league",
    "qualifier",
    "qualification",
)

KNOCKOUT_TOKENS = (
    "round of 16",
    "quarter-final",
    "quarterfinal",
    "semi-final",
    "semifinal",
    "final",
    "playoff",
    "play-off",
    "knockout",
    "second leg",
    "1st leg",
    "2nd leg",
)

QUALIFICATION_TOKENS = (
    "qualifier",
    "qualification",
    "qualifying",
    "preliminary",
)


def _competition_importance(
    league: str,
    sport_key: str,
    event: str,
) -> float:
    text = " ".join(
        (
            normalize(league),
            normalize(sport_key),
            normalize(event),
        )
    )

    if "fifa world cup" in text:
        return 1.00

    if "champions league" in text:
        return 0.95

    if (
        "europa league" in text
        or "conference league" in text
    ):
        return 0.88

    if "nations league" in text:
        return 0.82

    if any(
        token in text
        for token in QUALIFICATION_TOKENS
    ):
        return 0.78

    if any(
        token in text
        for token in (
            "premier league",
            "la liga",
            "bundesliga",
            "serie a",
            "ligue 1",
        )
    ):
        return 0.85

    if any(
        token in text
        for token in (
            "championship",
            "segunda",
            "serie b",
            "ligue 2",
            "2. bundesliga",
        )
    ):
        return 0.72

    return 0.68


def _season_stage(
    commence_time: str,
    is_international: bool,
) -> float:
    start = parse_datetime(commence_time)

    if start is None:
        return 0.50

    month = start.month

    if is_international:
        if month in {6, 7}:
            return 0.90

        if month in {3, 9, 10, 11}:
            return 0.65

        return 0.50

    # European-style league calendar approximation:
    # 0.0 = early season, 1.0 = late/decisive season.
    if month in {7, 8, 9}:
        return 0.15
    if month in {10, 11}:
        return 0.35
    if month in {12, 1, 2}:
        return 0.55
    if month in {3, 4}:
        return 0.78
    if month in {5, 6}:
        return 0.95

    return 0.50


def build_football_context_v15(
    *,
    league: str,
    sport_key: str,
    event: str,
    commence_time: str,
) -> FootballContextV15:
    text = " ".join(
        (
            normalize(league),
            normalize(sport_key),
            normalize(event),
        )
    )

    is_international = any(
        token in text
        for token in INTERNATIONAL_TOKENS
    )
    is_knockout = any(
        token in text
        for token in KNOCKOUT_TOKENS
    )
    is_qualification = any(
        token in text
        for token in QUALIFICATION_TOKENS
    )

    importance = _competition_importance(
        league,
        sport_key,
        event,
    )
    season_stage = _season_stage(
        commence_time,
        is_international,
    )

    # Reliability is intentionally limited because these are deterministic
    # context heuristics, not confirmed lineup/weather/tournament-state data.
    context_reliability = 0.55

    if is_knockout or is_qualification:
        context_reliability = 0.70

    reason = (
        f"context_v15: importance={importance:.2f}; "
        f"international={int(is_international)}; "
        f"knockout={int(is_knockout)}; "
        f"qualification={int(is_qualification)}; "
        f"season_stage={season_stage:.2f}; "
        f"reliability={context_reliability:.2f}"
    )

    return FootballContextV15(
        competition_importance=importance,
        is_international=float(is_international),
        is_knockout=float(is_knockout),
        is_qualification=float(is_qualification),
        season_stage=season_stage,
        context_reliability=context_reliability,
        reason=reason,
    )
