from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.market import consensus_h2h, best_outlier_prices, dedupe_best_bets
from core.odds_api import fetch_odds
from core.monte_carlo import simulate_single_bet
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
from core.meta_model import MetaFeatures, predict_probability
from core.adaptive_weights import (
    sport_weight,
    bookmaker_weight,
    league_weight,
)



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
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class MMAModule(SportModule):
    name = "mma"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


    def _extra_adjustment(self, home: str, away: str, selection: str, league: str) -> float:
        # MMA has no home side. Keep this neutral until enough fighter ELO data exists.
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

                    rows.append((
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
                    ))

        if not rows:
            return 0

        with self._connect(settings) as conn:
            before = conn.total_changes
            conn.executemany("""
                INSERT OR IGNORE INTO sport_odds_snapshots
                (
                    captured_at, sport, league, event, home_team, away_team,
                    bookmaker, market, selection, odds, source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

            return conn.total_changes - before

    def _save_bet(self, settings: Settings, bet: Bet) -> None:
        source_hash = make_hash(
    bet.sport,
    bet.league,
    bet.event,
    bet.market,
    bet.selection,
    bet.start_time,
)

        with self._connect(settings) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO sport_bets
                (
                    sport, league, event, home_team, away_team, market,
                    selection, odds, prob_model, prob_market, prob_final,
                    edge, stake, bookmaker, start_time, score, source_hash,
                    result
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet.sport,
                bet.league,
                bet.event,
                bet.event.split(" vs ")[0] if " vs " in bet.event else "",
                bet.event.split(" vs ")[1] if " vs " in bet.event else "",
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
            ))

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
            conn.execute("""
                INSERT INTO sport_decision_audit
                (
                    sport, league, event, selection, bookmaker,
                    odds, prob_market, edge, decision, reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
            ))

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        configured_keys = os.getenv(
            "MMA_SPORT_KEYS",
            "mma_mixed_martial_arts",
        ).split(",")

        clean_sport_keys = [s.strip() for s in configured_keys if s.strip()]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Mixed Martial Arts"],
            )
            clean_sport_keys = filter_active_keys(clean_sport_keys, active_keys)

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )

        updated_clv = update_closing_lines(settings, self.name)
        refresh_bookmaker_stats(settings, self.name)

        min_books = int(os.getenv("MIN_MMA_BOOKMAKERS", "3"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))
        grade_min_samples = int(os.getenv("MMA_BOOKMAKER_GRADE_MIN_SAMPLES", "20"))

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

                consensus = consensus_h2h(bookmakers, min_books=min_books)

                if not consensus:
                    blocked += 1
                    self._audit(
                        settings, sport_key, event_name, "", "", 0,
                        None, None, "BLOCK", "no market consensus"
                    )
                    continue

                for bookmaker, selection, odds in best_outlier_prices(bookmakers):
                    prob_market = consensus.get(selection)

                    if not prob_market:
                        blocked += 1
                        self._audit(
                            settings, sport_key, event_name, selection, bookmaker, odds,
                            None, None, "BLOCK", "selection missing in consensus"
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

                    fallback_probability = max(
                        0.01,
                        min(0.99, prob_market + elo_adj + extra_adj),
                    )

                    current_sport_weight = sport_weight(self.name)
                    current_bookmaker_weight = bookmaker_weight(bookmaker)
                    current_league_weight = league_weight(league)

                    mc_preview = simulate_single_bet(
                        probability=fallback_probability,
                        odds=odds,
                    )

                    probability_source = "FALLBACK"
                    probability_reason = "original mma probability"

                    try:
                        features = MetaFeatures(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_grade=grade,
                            sport_weight=current_sport_weight,
                            league_weight=current_league_weight,
                            confidence=50.0,
                            monte_carlo_probability=_mc_probability(mc_preview),
                        )
                        prob_final = max(0.01, min(0.99, predict_probability(features)))
                        probability_source = "META_MODEL"
                        probability_reason = f"meta_probability={prob_final:.4f}"
                    except Exception as exc:
                        prob_final = fallback_probability
                        probability_reason = f"{type(exc).__name__}: {exc}"

                    edge = prob_final * odds - 1.0
                    adjusted_edge = (
                        edge
                        * grade
                        * current_sport_weight
                        * current_bookmaker_weight
                        * current_league_weight
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
                            (
                                "edge below minimum; "
                                f"probability_source={probability_source}; "
                                f"probability_reason={probability_reason}"
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
                                "edge above max guard; "
                                f"probability_source={probability_source}; "
                                f"probability_reason={probability_reason}"
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
                                "odds above max odds; "
                                f"probability_source={probability_source}; "
                                f"probability_reason={probability_reason}"
                            ),
                        )
                        continue

                    stake = round(
                        kelly_stake(prob_final, odds, settings) * grade,
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
                                "stake <= 0; "
                                f"probability_source={probability_source}; "
                                f"probability_reason={probability_reason}"
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
                        score=adjusted_edge * 100,
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
                        f"bookmaker grade {grade:.2f}, elo_adj {elo_adj:.3f}, extra_adj {extra_adj:.3f}, probability_source={probability_source}, probability_reason={probability_reason}",
                    )

        bets = dedupe_best_bets(bets)
        analytics = sport_analytics_report(settings, self.name)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                "MMA v12: meta-model/fallback/CLV/ELO market model. "
                f"Settled: {settled}. "
                f"CLV updated: {updated_clv}. "
                f"Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}.\n"
                f"{analytics}"
            ),
        )
