from __future__ import annotations

import asyncio
import logging
import os

from core.config import Settings
from core.football_settlement import settle_football_bets
from core.settlement_integrity import build_settlement_integrity
from core.sport_settlement import expire_historical_unresolved, settle_sport_bets
from main import restore_learning_history, save_learning_history


log = logging.getLogger("settlement-cycle")


SPORT_KEYS = {
    "baseball": "BASEBALL_SPORT_KEYS",
    "basketball": "BASKETBALL_SPORT_KEYS",
    "hockey": "HOCKEY_SPORT_KEYS",
    "nfl": "NFL_SPORT_KEYS",
    "tennis": "TENNIS_SPORT_KEYS",
}

DEFAULT_KEYS = {
    "baseball": "baseball_mlb,baseball_npb,baseball_kbo",
    "basketball": "basketball_nba,basketball_euroleague,basketball_ncaab,basketball_wnba",
    "hockey": "icehockey_nhl,icehockey_ahl,icehockey_sweden_hockey_league,icehockey_sweden_allsvenskan,icehockey_liiga,icehockey_mestis",
    "nfl": "americanfootball_nfl",
    "tennis": "",
}


async def run_cycle() -> None:
    settings = Settings.from_env()
    restore_learning_history(settings)

    total = 0
    for sport, env_name in SPORT_KEYS.items():
        configured = os.getenv(env_name, DEFAULT_KEYS[sport])
        keys = [item.strip() for item in configured.split(",") if item.strip()]
        settled = await settle_sport_bets(settings, sport, keys)
        total += settled
        log.info("%s settlement: %s", sport, settled)

    football = await settle_football_bets(
        settings,
        days_from=int(os.getenv("FOOTBALL_SETTLEMENT_DAYS_FROM", "3")),
    )
    total += football.settled_won + football.settled_lost + football.settled_void

    unresolved = expire_historical_unresolved(
        settings,
        older_than_days=int(os.getenv("UNRESOLVED_EXPIRY_DAYS", "4")),
    )
    save_learning_history(settings)
    build_settlement_integrity(
        settings,
        export_dir=os.getenv("EXPORT_DIR", "exports"),
    )
    log.info("Settlement cycle finished: settled=%s unresolved=%s", total, unresolved)


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    asyncio.run(run_cycle())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
