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
    update_closing_lines,
)
from core.sport_settlement import settle_sport_bets
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule
from core.meta_model import (
    MetaFeatures,
    predict_probability,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def clamp(
    value: float,
    low: float = 0.01,
    high: float = 0.99,
) -> float:
    return max(low, min(high, value))


class BaseballModule(SportModule):
    name = "baseball"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _extra_adjustment(
        self,
        home: str,
        away: str,
        selection: str,
        league: str,
    ) -> float:
        """
        Malý baseballový kontextový bonus pre domáci tím.

        Parameter league je ponechaný, aby sa sem neskôr dali doplniť
        samostatné úpravy pre MLB, NPB alebo KBO.
        """

        if os.getenv("BASEBALL_HOME_ADV_ENABLED", "1") != "1":
            return 0.0

        try:
            home_adv = float(
                os.getenv("BASEBALL_HOME_PROB_ADV", "0.006")
            )
        except (TypeError, ValueError):
            home_adv = 0.006

        if selection == home:
            return home_adv

        if selection == away:
            return -home_adv

        return 0.0

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
        rows: list[tuple[Any, ...]] = []

        for bookmaker in bookmakers:
            book = str(bookmaker.get("title", ""))

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    selection = str(outcome.get("name", ""))

                    try:
                        odds = float(outcome.get("price", 0) or 0)
                    except (TypeError, ValueError):
                        continue

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
            "BASEBALL_SPORT_KEYS",
            "baseball_mlb,baseball_npb,baseball_kbo",
        ).split(",")

        clean_sport_keys = [
            sport_key.strip()
            for sport_key in configured_keys
            if sport_key.strip()
        ]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Baseball"],
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

        min_books = int(
            os.getenv("MIN_BASEBALL_BOOKMAKERS", "3")
        )
        top_n = int(
            os.getenv("TOP_N_REPORT", "8")
        )
        grade_min_samples = int(
            os.getenv(
                "BASEBALL_BOOKMAKER_GRADE_MIN_SAMPLES",
                "20",
            )
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
                    settings=settings,
                    sport_key=sport_key,
                    event_name=event_name,
                    home=home,
                    away=away,
                    bookmakers=bookmakers,
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

                for bookmaker, selection, odds in best_outlier_prices(
                    bookmakers
                ):
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

                    extra_adj = self._extra_adjustment(
                        home=home,
                        away=away,
                        selection=selection,
                        league=league,
                    )

                    features = MetaFeatures(
    market_probability=prob_market,
    elo_adjustment=elo_adj,
    form_adjustment=0.0,
    clv_adjustment=0.0,
    bookmaker_grade=grade,
    sport_weight=current_sport_weight,
    league_weight=current_league_weight,
    confidence=base_confidence,
    monte_carlo_probability=mc.win_probability,
)

prob_final = predict_probability(features)

edge = prob_final * odds - 1.0

                    current_sport_weight = sport_weight(self.name)
                    current_bookmaker_weight = bookmaker_weight(bookmaker)
                    current_league_weight = league_weight(league)

                    adjusted_edge = (
                        edge
                        * grade
                        * current_sport_weight
                        * current_bookmaker_weight
                        * current_league_weight
                    )

                    mc = simulate_single_bet(
                        probability=prob_final,
                        odds=odds,
                    )

                    mc_score = monte_carlo_score(mc)

                    base_confidence = confidence_score(
                        ConfidenceInput(
                            edge=edge,
                            consensus=max(
                                0.0,
                                min(0.20, edge),
                            ),
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

                    confidence = max(
                        1,
                        min(100, confidence),
                    )

                    audit_context = (
                        f"confidence={confidence}; "
                        f"base_confidence={base_confidence}; "
                        f"mc_score={mc_score:.2f}; "
                        f"adaptive_edge={adjusted_edge:.4f}; "
                        f"grade={grade:.4f}; "
                        f"sport_weight={current_sport_weight:.4f}; "
                        f"bookmaker_weight={current_bookmaker_weight:.4f}; "
                        f"league_weight={current_league_weight:.4f}; "
                        f"elo_adj={elo_adj:.4f}; "
                        f"extra_adj={extra_adj:.4f}; "
                        f"{format_monte_carlo_reason(mc)}"
                    )

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
                            f"edge below minimum; {audit_context}",
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
                            f"edge above max guard; {audit_context}",
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
                            f"odds above max odds; {audit_context}",
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
                            f"stake <= 0; {audit_context}",
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
                        audit_context,
                    )

        bets = dedupe_best_bets(bets)
        analytics = sport_analytics_report(settings, self.name)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                "Baseball: CLV/ELO/Bayesian/adaptive confidence/"
                "Monte Carlo model. "
                f"Settled: {settled}. "
                f"CLV updated: {updated_clv}. "
                f"Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}.\n"
                f"{analytics}"
            ),
        )
