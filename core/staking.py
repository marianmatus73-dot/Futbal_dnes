from __future__ import annotations

from core.config import Settings


def kelly_stake(prob: float, odds: float, settings: Settings) -> float:
    if odds <= 1.01 or prob <= 0:
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    kelly = ((b * prob) - q) / b

    stake_pct = max(0.0, kelly * settings.kelly_frac)
    stake_pct = min(stake_pct, settings.max_stake_pct)

    return round(settings.bank * stake_pct, 2)
