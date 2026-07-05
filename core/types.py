from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SportTip:
    sport: str
    league: str
    match: str
    pick: str
    odds: float
    model_probability: float
    bookmaker: str = "N/A"
    reason: str = ""
    start_time: str = "N/A"
    edge: float = 0.0
    stake: float = 0.0
    market_probability: Optional[float] = None
    raw_data: dict = field(default_factory=dict)

@dataclass
class SportResult:
    sport: str
    mode: str
    bets: list[SportTip] = field(default_factory=list)
    message: str = ""
    ok: bool = True
    error: Optional[str] = None
    duration_sec: float = 0.0
