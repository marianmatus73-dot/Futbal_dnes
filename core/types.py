from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Bet:
    sport: str
    league: str
    event: str
    market: str
    selection: str
    odds: float
    prob_model: float
    prob_market: Optional[float]
    prob_final: float
    edge: float
    stake: float
    bookmaker: str
    start_time: str
    score: float = 0.0


@dataclass
class SportResult:
    sport: str
    mode: str
    bets: list[Bet]
    message: str = ""
