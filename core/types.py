from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any

# Tento model bude držať konzistenciu dát v celom systéme
class SportTip(BaseModel):
    sport: str
    league: str
    match: str
    pick: str
    odds: float
    model_probability: float
    bookmaker: str = "N/A"
    reason: str = ""
    start_time: str = "N/A"
    
    # Štatistiky, ktoré používajú tvoje moduly
    edge: float = 0.0
    stake: float = 0.0
    market_probability: Optional[float] = None
    
    class Config:
        frozen = True  # Tip sa po vytvorení už nesmie meniť (bezpečnosť)

class SportResult(BaseModel):
    sport: str
    mode: str
    bets: list[SportTip] = Field(default_factory=list)
    message: str = ""
    ok: bool = True
    error: Optional[str] = None
    duration_sec: float = 0.0
