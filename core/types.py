from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bet:
    sport: str
    league: str
    event: str
    market: str
    selection: str
    odds: float
    prob_model: float
    prob_market: float
    prob_final: float
    edge: float
    stake: float
    bookmaker: str
    start_time: str
    score: float = 0.0
    external_event_id: str = ""
    release_stage: str = ""
    lineup_verified: bool = False
    opening_odds: float | None = None
    final_odds: float | None = None


@dataclass
class SportResult:
    sport: str
    mode: str
    bets: list[Bet] = field(default_factory=list)
    message: str = ""
