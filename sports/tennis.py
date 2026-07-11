from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.adaptive_weights import (
    bookmaker_weight,
    league_weight,
    sport_weight,
)
from core.confidence_model import ConfidenceInput, confidence_score
from core.config import Settings
from core.ensemble_model import EnsembleInput, build_ensemble_probability
from core.market import best_outlier_prices, consensus_h2h, dedupe_best_bets
from core.monte_carlo import (
    format_monte_carlo_reason,
    monte_carlo_score,
    simulate_single_bet,
)
from core.odds_api import fetch_odds
from core.sport_quant import (
    bookmaker_grade,
    discover_active_sport_keys,
    elo_adjustment,
    filter_active_keys,
    init_sport_db,
    refresh_bookmaker_stats,
    sport_analytics_report,
    tennis_surface_adjustment,
    update_closing_lines,
)
from core.sport_settlement import settle_sport_bets
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule
from core.meta_model import MetaFeatures, predict_probability



def _mc_probability(result: Any) -> float:
    """Read Monte Carlo probability across older/newer result objects."""
    value = getattr(result, "win_probability", None)
    if value is None:
        value = getattr(result, "simulated_win_probability", None)
    if value is None:
        raise AttributeError("Monte Carlo result has no win probability field")
    return max(0.01, min(0.99, float(value)))

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class TennisModule(SportModule):
    name = "tennis"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _save_snapshot(
        self,
        settings: Settings,
        sport_key: str,
        event_name: str,
        home: str,
        away: str,
        bookmakers: list[dict],
    ) -> int:
        if os.getenv("SNAPSHOT_ODDS", "1") != "1":
            return 0

        captured_at = now_utc()
        rows = []

        for bookmaker in bookmakers:
            book = str(bookmaker.get("title", ""))

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    selection = str(outcome.get("name", ""))
                    odds = float(outcome.get("price", 0) or 0)

                    if odds <= 1.01:
                        continue

                    source_hash = make_hash(
                        captured_at,
                        self.name,
                        sport_key,
                        event_name,
                        book,
                        "h2h",
                        selection,
                        odds,
                    )

                    rows.append(
                        (
                            captured_at,
                            self.name,
                            sport_key,
                            event_name,
                            home,
                            away,
                            book,
                            "h2h",
                            selection,
                            odds,
                            source_hash,
                        )
                    )

        if not rows:
            return 0

        with self._connect(settings) as conn:
            before = conn.total_changes

            conn.executemany(
                """
                INSERT OR IGNORE INTO sport_odds_snapshots
                (
                    captured_at,
                    sport,
                    league,
                    event,
                    home_team,
                    away_team,
                    bookmaker,
                    market,
                    selection,
                    odds,
                    source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            return conn.total_changes - before

    def _save_bet(self, settings: Settings, bet: Bet) -> None:
        source_hash = make_hash(
            bet.sport,
            bet.league,
            bet.event,
            bet.market,
            bet.selection,
            bet.odds,
            bet.bookmaker,
            bet.start_time,
        )

        home_team = ""
        away_team = ""

        if " vs " in bet.event:
            home_team, away_team = bet.event.split(" vs ", 1)

        with self._connect(settings) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sport_bets
                (
                    sport,
                    league,
                    event,
                    home_team,
                    away_team,
                    market,
                    selection,
                    odds,
                    prob_model,
                    prob_market,
                    prob_final,
                    edge,
                    stake,
                    bookmaker,
                    start_time,
                    score,
                    source_hash,
                    result
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet.sport,
                    bet.league,
                    bet.event,
                    home_team,
                    away_team,
                    bet.market,
                    bet.selection,
                    bet.odds,
                    bet.prob_model,
                    bet.prob_market,
                    bet.prob_final,
                    bet.edge,
                    bet.stake,
                    bet.bookmaker,
                    bet.start_time,
                    bet.score,
                    source_hash,
                    "OPEN",
                ),
            )

    def _audit(
        self,
        settings: Settings,
        sport_key: str,
        event_name: str,
        selection: str,
        bookmaker: str,
        odds: float,
        prob_market: float | None,
        edge: float | None,
        decision: str,
        reason: str,
    ) -> None:
        with self._connect(settings) as conn:
            conn.execute(
                """
                INSERT INTO sport_decision_audit
                (
                    sport,
                    league,
                    event,
                    selection,
                    bookmaker,
                    odds,
                    prob_market,
                    edge,
                    decision,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.name,
                    sport_key,
                    event_name,
                    selection,
                    bookmaker,
                    odds,
                    prob_market,
                    edge,
                    decision,
                    reason,
                ),
            )

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        configured_keys = os.getenv(
            "TENNIS_SPORT_KEYS",
            ",".join(
                [
                    "tennis_atp_australian_open",
                    "tennis_wta_australian_open",
                    "tennis_atp_french_open",
                    "tennis_wta_french_open",
                    "tennis_atp_wimbledon",
                    "tennis_wta_wimbledon",
                    "tennis_atp_us_open",
                    "tennis_wta_us_open",
                    "tennis_atp_indian_wells",
                    "tennis_wta_indian_wells",
                    "tennis_atp_miami_open",
                    "tennis_wta_miami_open",
                    "tennis_atp_monte_carlo_masters",
                    "tennis_atp_madrid_open",
                    "tennis_wta_madrid_open",
                    "tennis_atp_italian_open",
                    "tennis_wta_italian_open",
                    "tennis_atp_canadian_open",
                    "tennis_wta_canadian_open",
                    "tennis_atp_cincinnati_open",
                    "tennis_wta_cincinnati_open",
                    "tennis_atp_shanghai_masters",
                    "tennis_atp_paris_masters",
                    "tennis_atp_dubai",
                    "tennis_wta_dubai",
                    "tennis_atp_qatar_open",
                    "tennis_wta_qatar_open",
                    "tennis_atp_halle",
                    "tennis_atp_queens_club",
                    "tennis_atp_stuttgart",
                    "tennis_wta_berlin",
                    "tennis_wta_eastbourne",
                    "tennis_atp_eastbourne",
                    "tennis_atp_basel",
                    "tennis_atp_vienna",
                    "tennis_atp_tokyo",
                    "tennis_wta_tokyo",
                    "tennis_atp_beijing",
                    "tennis_wta_beijing",
                    "tennis_atp_wta_generic",
                    "tennis_atp_challenger",
                    "tennis_wta_125",
                    "tennis_itf",
                ]
            ),
        ).split(",")

        clean_sport_keys = [
            sport_key.strip()
            for sport_key in configured_keys
            if sport_key.strip()
        ]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Tennis"],
            )

            clean_sport_keys = filter_active_keys(
                clean_sport_keys,
                active_keys,
            )

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )

        updated_clv = update_closing_lines(settings, self.name)
        refresh_bookmaker_stats(settings, self.name)

        min_books = int(os.getenv("MIN_TENNIS_BOOKMAKERS", "2"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))
        grade_min_samples = int(
            os.getenv("TENNIS_BOOKMAKER_GRADE_MIN_SAMPLES", "20")
        )

        bets: list[Bet] = []
        snapshots_saved = 0
        blocked = 0
        scanned_events = 0

        for sport_key in clean_sport_keys:
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

                scanned_events += 1

                snapshots_saved += self._save_snapshot(
                    settings,
                    sport_key,
                    event_name,
                    home,
                    away,
                    bookmakers,
                )

                consensus = consensus_h2h(
                    bookmakers,
                    min_books=min_books,
                )

                if not consensus:
                    blocked += 1

                    self._audit(
                        settings,
                        sport_key,
                        event_name,
                        "",
                        "",
                        0,
                        None,
                        None,
                        "BLOCK",
                        "no market consensus",
                    )
                    continue

                for bookmaker, selection, odds in best_outlier_prices(bookmakers):
                    prob_market = consensus.get(selection)

                    if not prob_market:
                        blocked += 1

                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            None,
                            None,
                            "BLOCK",
                            "selection missing in consensus",
                        )
                        continue

                    grade = bookmaker_grade(
                        settings,
                        self.name,
                        bookmaker,
                        min_samples=grade_min_samples,
                    )

                    elo_adj = elo_adjustment(
                        settings,
                        self.name,
                        home,
                        away,
                        selection,
                    )

                    surface_adj = tennis_surface_adjustment(sport_key)

                    ensemble = build_ensemble_probability(
                        EnsembleInput(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_adjustment=(grade - 1.0) * 0.02,
                            sport_adjustment=surface_adj,
                        ),
                        odds=odds,
                    )

                    fallback_probability = ensemble.probability
                    fallback_edge = ensemble.edge

                    current_sport_weight = sport_weight(self.name)
                    current_bookmaker_weight = bookmaker_weight(bookmaker)
                    current_league_weight = league_weight(league)

                    mc_preview = simulate_single_bet(
                        probability=fallback_probability,
                        odds=odds,
                    )

                    preview_confidence = confidence_score(
                        ConfidenceInput(
                            edge=fallback_edge,
                            consensus=max(0.0, min(0.20, fallback_edge)),
                            bayesian=0.50,
                            clv=0.0,
                            bookmaker_weight=current_bookmaker_weight,
                            sport_weight=current_sport_weight,
                            league_weight=current_league_weight,
                            samples=0,
                        )
                    )

                    probability_source = "FALLBACK"
                    probability_reason = "original tennis ensemble"

                    try:
                        features = MetaFeatures(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_grade=grade,
                            sport_weight=current_sport_weight,
                            league_weight=current_league_weight,
                            confidence=float(preview_confidence),
                            monte_carlo_probability=_mc_probability(mc_preview),
                        )
                        prob_final = max(0.01, min(0.99, predict_probability(features)))
                        edge = prob_final * odds - 1.0
                        probability_source = "META_MODEL"
                        probability_reason = f"meta_probability={prob_final:.4f}"
                    except Exception as exc:
                        prob_final = fallback_probability
                        edge = fallback_edge
                        probability_reason = f"{type(exc).__name__}: {exc}"

                    mc = simulate_single_bet(
                        probability=prob_final,
                        odds=odds,
                    )

                    mc_score = monte_carlo_score(mc)

                    adjusted_edge = (
                        edge
                        * grade
                        * current_sport_weight
                        * current_bookmaker_weight
                        * current_league_weight
                    )

                    base_confidence = confidence_score(
                        ConfidenceInput(
                            edge=edge,
                            consensus=max(0.0, min(0.20, edge)),
                            bayesian=0.50,
                            clv=0.0,
                            bookmaker_weight=current_bookmaker_weight,
                            sport_weight=current_sport_weight,
                            league_weight=current_league_weight,
                            samples=0,
                        )
                    )

                    confidence = int(
                        round(
                            base_confidence * 0.80
                            + mc_score * 0.20
                        )
                    )

                    confidence = max(1, min(100, confidence))

                    if edge < settings.min_edge:
                        blocked += 1

                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            (
                                f"edge below minimum; probability_source={probability_source}; probability_reason={probability_reason}; "
                                f"confidence={confidence}; "
                                f"base_confidence={base_confidence}; "
                                f"mc_score={mc_score:.2f}; "
                                f"adaptive_edge={adjusted_edge:.4f}; "
                                f"{format_monte_carlo_reason(mc)}"
                            ),
                        )
                        continue

                    if edge > settings.max_edge:
                        blocked += 1

                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            (
                                f"edge above max guard; probability_source={probability_source}; probability_reason={probability_reason}; "
                                f"confidence={confidence}; "
                                f"base_confidence={base_confidence}; "
                                f"mc_score={mc_score:.2f}; "
                                f"adaptive_edge={adjusted_edge:.4f}; "
                                f"{format_monte_carlo_reason(mc)}"
                            ),
                        )
                        continue

                    if odds > settings.max_odds:
                        blocked += 1

                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            (
                                f"odds above max odds; probability_source={probability_source}; probability_reason={probability_reason}; "
                                f"confidence={confidence}; "
                                f"base_confidence={base_confidence}; "
                                f"mc_score={mc_score:.2f}; "
                                f"adaptive_edge={adjusted_edge:.4f}; "
                                f"{format_monte_carlo_reason(mc)}"
                            ),
                        )
                        continue

                    stake = round(
                        kelly_stake(
                            prob_final,
                            odds,
                            settings,
                        )
                        * grade,
                        2,
                    )

                    if stake <= 0:
                        blocked += 1

                        self._audit(
                            settings,
                            sport_key,
                            event_name,
                            selection,
                            bookmaker,
                            odds,
                            prob_market,
                            edge,
                            "BLOCK",
                            (
                                f"stake <= 0; probability_source={probability_source}; probability_reason={probability_reason}; "
                                f"confidence={confidence}; "
                                f"base_confidence={base_confidence}; "
                                f"mc_score={mc_score:.2f}; "
                                f"adaptive_edge={adjusted_edge:.4f}; "
                                f"{format_monte_carlo_reason(mc)}"
                            ),
                        )
                        continue

                    bet = Bet(
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
                        score=float(confidence),
                    )

                    bets.append(bet)
                    self._save_bet(settings, bet)

                    self._audit(
                        settings,
                        sport_key,
                        event_name,
                        selection,
                        bookmaker,
                        odds,
                        prob_market,
                        edge,
                        "PASS",
                        (
                            f"{ensemble.reason}; probability_source={probability_source}; probability_reason={probability_reason}; "
                            f"confidence={confidence}; "
                            f"base_confidence={base_confidence}; "
                            f"mc_score={mc_score:.2f}; "
                            f"adaptive_edge={adjusted_edge:.4f}; "
                            f"grade={grade:.3f}; "
                            f"sport_weight={current_sport_weight:.4f}; "
                            f"bookmaker_weight={current_bookmaker_weight:.4f}; "
                            f"league_weight={current_league_weight:.4f}; "
                            f"elo_adj={elo_adj:.4f}; "
                            f"surface_adj={surface_adj:.4f}; "
                            f"{format_monte_carlo_reason(mc)}"
                        ),
                    )

        bets = dedupe_best_bets(bets)
        analytics = sport_analytics_report(settings, self.name)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                "Tennis v12: meta-model/fallback/ensemble/Bayesian/adaptive confidence/Monte Carlo model. "
                f"Settled: {settled}. "
                f"CLV updated: {updated_clv}. "
                f"Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}.\n"
                f"{analytics}"
            ),
        )
