from __future__ import annotations
import os
import sqlite3
import hashlib
import logging
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

class TennisModule(SportModule):
    name = "tennis"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _save_bet(self, settings: Settings, bet: Bet) -> None:
        with self._connect(settings) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sport_bets (sport, league, event, selection, odds, stake, bookmaker, result) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')",
                (bet.sport, bet.league, bet.event, bet.selection, bet.odds, bet.stake, bet.bookmaker)
            )

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        # Oficiálne kľúče podľa dokumentácie The Odds API
        configured_keys = os.getenv(
            "TENNIS_SPORT_KEYS",
            "tennis_atp_aus_open_singles,tennis_wta_aus_open_singles,tennis_atp_french_open,tennis_wta_french_open,tennis_atp_wimbledon,tennis_wta_wimbledon,tennis_atp_us_open,tennis_wta_us_open,tennis_atp_indian_wells,tennis_atp_miami_open,tennis_atp_monte_carlo_masters,tennis_atp_madrid_open,tennis_atp_italian_open,tennis_atp_canadian_open,tennis_atp_cincinnati_open,tennis_atp_shanghai_masters,tennis_atp_paris_masters"
        ).split(",")

        clean_sport_keys = [k.strip() for k in configured_keys if k.strip()]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(settings.odds_api_key, ["Tennis"])
            
            # Táto riadka je kľúčová - filtruje iba kľúče, ktoré sú naozaj v sezóne
            clean_sport_keys = [k for k in clean_sport_keys if k in active_keys]
            
            log.info("Tennis: Skenujem kľúče v sezóne: %s", clean_sport_keys)
        else:
            log.info("Tennis: Skenujem všetky nakonfigurované kľúče (auto-discovery vypnuté)")

        if not clean_sport_keys:
            log.info("Tennis: Žiadne turnaje práve teraz nie sú v sezóne.")
            return SportResult(sport=self.name, mode="scan", bets=[], message="Tennis: No active events.")

        settled = await settle_sport_bets(settings=settings, sport=self.name, sport_keys=clean_sport_keys)
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
                                edge=edge, stake=stake, bookmaker=bookmaker, start_time=str(event.get("commence_time")), score=float(edge*100)
                            )
                            bets.append(bet)
                            self._save_bet(settings, bet)
            except Exception as e:
                log.warning("Tennis: Chyba pri spracovaní kľúča %s: %s", sport_key, e)

        return SportResult(sport=self.name, mode="scan", bets=bets[:top_n], message=f"Tennis scan hotový. Events: {scanned_events}, Stored: {len(bets)}")
