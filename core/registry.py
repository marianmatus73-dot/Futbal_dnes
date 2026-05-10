from __future__ import annotations

from sports.base import SportModule
from sports.football import FootballModule
from sports.tennis import TennisModule
from sports.basketball import BasketballModule
from sports.hockey import HockeyModule


_SPORTS: dict[str, SportModule] = {
    "football": FootballModule(),
    "tennis": TennisModule(),
    "basketball": BasketballModule(),
    "hockey": HockeyModule(),
}


def get_sports() -> dict[str, SportModule]:
    return _SPORTS


def get_sport(name: str) -> SportModule:
    return _SPORTS[name]
