from __future__ import annotations
import os
import sqlite3
import hashlib
import logging
from contextlib import closing
from pathlib import Path
from typing import Any

from core.adaptive_weights import bookmaker_weight, league_weight, sport_weight
from core.config import Settings
from core.ensemble_model import EnsembleInput, build_ensemble_probability
from core.market import best_outlier_prices, consensus_h2h, dedupe_best_bets
from core.odds_api import fetch_odds
from core.sport_quant import (
    bookmaker_grade, discover_active_sport_keys, elo_adjustment, 
    filter_active_keys, init_sport_db, refresh_bookmaker_stats, 
    sport_analytics_report, tennis_surface_adjustment, update_closing_lines
)
from core.sport_settlement import settle_sport_bets
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule
from core.meta_model import MetaFeatures, predict_probability

log = logging.getLogger("multisport-main")


def tennis_confidence(edge: float) -> int:
    """Map a qualified market-price edge onto the shared 0-100 scale."""
    return int(round(max(1.0, min(90.0, 65.0 + float(edge) * 100.0))))


class TennisModule(SportModule):
    name = "tennis"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _save_bet(self, settings: Settings, bet: Bet) -> None:
        home, separator, away = bet.event.partition(" vs ")
        source_hash = hashlib.sha256(
            f"tennis|{bet.external_event_id or bet.event}|{bet.market}".encode("utf-8")
        ).hexdigest()[:32]
        with closing(self._connect(settings)) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sport_bets
                   (sport, league, event, home_team, away_team, market,
                    selection, odds, prob_model, prob_market, prob_final,
                    edge, stake, bookmaker, start_time, score,
                    external_event_id, source_hash, result)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, 'OPEN')""",
                (
                    bet.sport, bet.league, bet.event, home if separator else "",
                    away if separator else "", bet.market, bet.selection,
                    bet.odds, bet.prob_model, bet.prob_market, bet.prob_final,
                    bet.edge, bet.stake, bet.bookmaker, bet.start_time,
                    bet.score, bet.external_event_id, source_hash,
                ),
            )
            conn.commit()

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        # Oficiálne kľúče podľa dokumentácie The Odds API
        configured_value = os.getenv("TENNIS_SPORT_KEYS", "").strip()
        configured_keys = configured_value.split(",") if configured_value else []

        clean_sport_keys = [k.strip() for k in configured_keys if k.strip()]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Tennis"],
            )

            # The Odds API adds short-lived ATP/WTA tournament keys during
            # the season.  A hard-coded allow-list silently removed all of
            # them and made tennis look inactive.  Use every active tennis
            # key unless the operator explicitly configured a restriction.
            if configured_value:
                clean_sport_keys = filter_active_keys(
                    clean_sport_keys,
                    active_keys,
                )
            elif active_keys:
                clean_sport_keys = sorted(active_keys)
            
            log.info("Tennis: Skenujem kľúče v sezóne: %s", clean_sport_keys)
        else:
            log.info("Tennis: Skenujem všetky nakonfigurované kľúče (auto-discovery vypnuté)")

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )
        if not clean_sport_keys:
            log.info("Tennis: Žiadne turnaje práve teraz nie sú v sezóne.")
            return SportResult(
                sport=self.name,
                mode="scan",
                bets=[],
                message=f"Tennis: No active events. Settled: {settled}.",
            )

        updated_clv = update_closing_lines(settings, self.name)
        refresh_bookmaker_stats(settings, self.name)

        min_books = int(os.getenv("MIN_TENNIS_BOOKMAKERS", "2"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))
        grade_min_samples = int(os.getenv("TENNIS_BOOKMAKER_GRADE_MIN_SAMPLES", "20"))

        bets, scanned_events = [], 0

        for sport_key in clean_sport_keys:
            try:
                data = await fetch_odds(settings.odds_api_key, sport_key, markets="h2h")
                if not data: continue

                for event in data:
                    league = sport_key
                    home, away = event.get("home_team", ""), event.get("away_team", "")
                    event_name = f"{home} vs {away}"
                    scanned_events += 1
                    
                    consensus = consensus_h2h(event.get("bookmakers", []), min_books=min_books)
                    if not consensus: continue

                    for bookmaker, selection, odds in best_outlier_prices(event.get("bookmakers", [])):
                        prob_market = consensus.get(selection)
                        if not prob_market: continue

                        grade = bookmaker_grade(settings, self.name, bookmaker, min_samples=grade_min_samples)
                        elo_adj = elo_adjustment(settings, self.name, home, away, selection)
                        
                        edge = (prob_market * odds) - 1.0 
                        stake = round(kelly_stake(prob_market, odds, settings) * grade, 2)

                        if settings.min_edge <= edge <= settings.max_edge and stake > 0:
                            bet = Bet(
                                sport=self.name, league=league, event=event_name, market="h2h", selection=selection, 
                                odds=odds, prob_model=prob_market, prob_market=prob_market, prob_final=prob_market,
                                edge=edge, stake=stake, bookmaker=bookmaker,
                                start_time=str(event.get("commence_time")),
                                score=float(tennis_confidence(edge)),
                                external_event_id=str(event.get("id", "")),
                            )
                            bets.append(bet)
            except Exception as e:
                log.warning("Tennis: Chyba pri spracovaní kľúča %s: %s", sport_key, e)

        bets = dedupe_best_bets(bets)
        for bet in bets:
            self._save_bet(settings, bet)
        return SportResult(sport=self.name, mode="scan", bets=bets[:top_n], message=f"Tennis scan hotový. Events: {scanned_events}, Stored: {len(bets)}")

