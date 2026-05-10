from __future__ import annotations

import os

from core.config import Settings
from core.market import consensus_h2h, best_outlier_prices, dedupe_best_bets
from core.odds_api import fetch_odds
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule


class TennisModule(SportModule):
    name = "tennis"

    async def scan(self, settings: Settings) -> SportResult:
        sport_keys = os.getenv(
            "TENNIS_SPORT_KEYS",
            (
                "tennis_atp_french_open,"
                "tennis_wta_french_open,"
                "tennis_atp_wimbledon,"
                "tennis_wta_wimbledon,"
                "tennis_atp_us_open,"
                "tennis_wta_us_open"
            ),
        ).split(",")

        min_books = int(os.getenv("MIN_TENNIS_BOOKMAKERS", "3"))
        bets: list[Bet] = []

        for sport_key in [s.strip() for s in sport_keys if s.strip()]:
            data = await fetch_odds(
                settings.odds_api_key,
                sport_key,
                markets="h2h",
            )

            if not data:
                continue

            for event in data:
                league = sport_key

                home = str(event.get("home_team", ""))
                away = str(event.get("away_team", ""))

                start = str(event.get("commence_time", ""))

                event_name = f"{home} vs {away}"

                bookmakers = event.get("bookmakers", [])

                consensus = consensus_h2h(
                    bookmakers,
                    min_books=min_books,
                )

                if not consensus:
                    continue

                for bookmaker, selection, odds in best_outlier_prices(bookmakers):

                    prob_market = consensus.get(selection)

                    if not prob_market:
                        continue

                    prob_final = prob_market

                    edge = prob_final * odds - 1.0

                    if edge < settings.min_edge:
                        continue

                    if edge > settings.max_edge:
                        continue

                    if odds > settings.max_odds:
                        continue

                    stake = kelly_stake(
                        prob_final,
                        odds,
                        settings,
                    )

                    if stake <= 0:
                        continue

                    bets.append(
                        Bet(
                            sport=self.name,
                            league=league,
                            event=event_name,
                            market="h2h",
                            selection=selection,
                            odds=odds,
                            prob_model=prob_market,
                            prob_market=prob_market,
                            prob_final=prob_final,
                            edge=edge,
                            stake=stake,
                            bookmaker=bookmaker,
                            start_time=start,
                            score=edge * 100,
                        )
                    )

        bets = dedupe_best_bets(bets)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[: int(os.getenv("TOP_N_REPORT", "8"))],
            message="Tennis: tournament-based market consensus model.",
        )
