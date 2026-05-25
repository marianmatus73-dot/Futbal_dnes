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
from core.sport_settlement import settle_sport_bets
from core.staking import kelly_stake
from core.types import Bet, SportResult
from sports.base import SportModule


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_hash(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class TennisModule(SportModule):
    name = "tennis"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self, settings: Settings) -> None:
        with self._connect(settings) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sport_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport TEXT,
                    league TEXT,
                    event TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    market TEXT,
                    selection TEXT,
                    odds REAL,
                    prob_model REAL,
                    prob_market REAL,
                    prob_final REAL,
                    edge REAL,
                    stake REAL,
                    bookmaker TEXT,
                    start_time TEXT,
                    score REAL,
                    source_hash TEXT UNIQUE,
                    result TEXT,
                    closing_odds REAL,
                    clv_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    settled_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sport_odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT,
                    sport TEXT,
                    league TEXT,
                    event TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    bookmaker TEXT,
                    market TEXT,
                    selection TEXT,
                    odds REAL,
                    source_hash TEXT UNIQUE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sport_decision_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    sport TEXT,
                    league TEXT,
                    event TEXT,
                    selection TEXT,
                    bookmaker TEXT,
                    odds REAL,
                    prob_market REAL,
                    edge REAL,
                    decision TEXT,
                    reason TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sport_bookmaker_stats (
                    bookmaker TEXT,
                    sport TEXT,
                    bets INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    turnover REAL DEFAULT 0,
                    profit REAL DEFAULT 0,
                    avg_clv REAL DEFAULT 0,
                    updated_at TEXT,
                    PRIMARY KEY(bookmaker, sport)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_bets_sport ON sport_bets(sport, league, start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_bets_result ON sport_bets(result)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_snapshots_event ON sport_odds_snapshots(sport, league, event, selection)")

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
            bet.odds,
            bet.bookmaker,
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
                "",
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

    def _bookmaker_grade(self, settings: Settings, bookmaker: str) -> float:
        min_samples = int(os.getenv("TENNIS_BOOKMAKER_GRADE_MIN_SAMPLES", "20"))

        with self._connect(settings) as conn:
            row = conn.execute("""
                SELECT bets, profit, turnover, avg_clv
                FROM sport_bookmaker_stats
                WHERE bookmaker=? AND sport=?
            """, (bookmaker, self.name)).fetchone()

        if not row:
            return 1.0

        bets, profit, turnover, avg_clv = row

        if int(bets or 0) < min_samples:
            return 1.0

        yld = float(profit or 0) / float(turnover or 1)
        clv = float(avg_clv or 0)

        grade = 1.0 + clv * 6.0 + yld * 2.0
        return max(0.60, min(1.40, grade))

    async def scan(self, settings: Settings) -> SportResult:
        self._init_db(settings)

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

        clean_sport_keys = [s.strip() for s in sport_keys if s.strip()]

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )

        min_books = int(os.getenv("MIN_TENNIS_BOOKMAKERS", "2"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))

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

                    grade = self._bookmaker_grade(settings, bookmaker)

                    prob_final = prob_market
                    edge = prob_final * odds - 1.0
                    adjusted_edge = edge * grade

                    if edge < settings.min_edge:
                        blocked += 1
                        self._audit(
                            settings, sport_key, event_name, selection, bookmaker, odds,
                            prob_market, edge, "BLOCK", "edge below minimum"
                        )
                        continue

                    if edge > settings.max_edge:
                        blocked += 1
                        self._audit(
                            settings, sport_key, event_name, selection, bookmaker, odds,
                            prob_market, edge, "BLOCK", "edge above max guard"
                        )
                        continue

                    if odds > settings.max_odds:
                        blocked += 1
                        self._audit(
                            settings, sport_key, event_name, selection, bookmaker, odds,
                            prob_market, edge, "BLOCK", "odds above max odds"
                        )
                        continue

                    stake = kelly_stake(prob_final, odds, settings)
                    stake = round(stake * grade, 2)

                    if stake <= 0:
                        blocked += 1
                        self._audit(
                            settings, sport_key, event_name, selection, bookmaker, odds,
                            prob_market, edge, "BLOCK", "stake <= 0"
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
                        settings, sport_key, event_name, selection, bookmaker, odds,
                        prob_market, edge, "PASS", f"bookmaker grade {grade:.2f}"
                    )

        bets = dedupe_best_bets(bets)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                "Tennis: enhanced history/CLV-ready market model. "
                f"Settled: {settled}. "
                f"Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}."
            ),
        )
