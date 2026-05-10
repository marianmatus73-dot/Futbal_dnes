from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import Settings
from core.types import SportResult


class SportModule(ABC):
    name: str

    @abstractmethod
    async def scan(self, settings: Settings) -> SportResult:
        pass

    async def backtest(self, settings: Settings, days: int = 180) -> SportResult:
        return SportResult(
            sport=self.name,
            mode="backtest",
            bets=[],
            message="Backtest not implemented yet for this sport.",
        )

    async def analytics(self, settings: Settings) -> SportResult:
        return SportResult(
            sport=self.name,
            mode="analytics",
            bets=[],
            message="Analytics not implemented yet for this sport.",
        )
