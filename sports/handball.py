from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.market import best_outlier_prices, consensus_h2h
from core.learning_observation_settlement import settle_learning_observations
from core.odds_api import fetch_odds
from core.sport_quant import (
    discover_active_sport_keys,
    filter_active_keys,
    init_sport_db,
)
from core.types import SportResult
from sports.base import SportModule


log = logging.getLogger("multisport-main")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class HandballModule(SportModule):
    """Handball data collector running in observation-only mode.

    Observations are deliberately kept outside ``sport_bets``.  Until the
    result matcher and calibration have enough verified samples, nothing from
    this module can become a published or staked tip.
    """

    name = "handball"

    def _db_path(self, settings: Settings) -> Path:
        return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))

    def _connect(self, settings: Settings) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path(settings))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_tables(self, settings: Settings) -> None:
        with closing(self._connect(settings)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sport_learning_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    league TEXT NOT NULL,
                    external_event_id TEXT,
                    event TEXT NOT NULL,
                    home_team TEXT,
                    away_team TEXT,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    bookmaker TEXT,
                    odds REAL NOT NULL,
                    market_probability REAL,
                    start_time TEXT,
                    mode TEXT NOT NULL DEFAULT 'SHADOW',
                    result TEXT NOT NULL DEFAULT 'OPEN',
                    final_score TEXT,
                    settled_at TEXT,
                    source_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_observations_lookup "
                "ON sport_learning_observations(sport, league, external_event_id, result)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sport_shadow_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    league TEXT NOT NULL,
                    external_event_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    home_team TEXT,
                    away_team TEXT,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    bookmaker TEXT,
                    odds REAL NOT NULL,
                    probability REAL NOT NULL,
                    start_time TEXT,
                    model_version TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT 'OPEN',
                    profit_units REAL,
                    final_score TEXT,
                    settled_at TEXT,
                    UNIQUE(sport, external_event_id, market, model_version)
                )
                """
            )
            conn.commit()

    def _save_snapshot_rows(
        self,
        settings: Settings,
        sport_key: str,
        event_name: str,
        home: str,
        away: str,
        bookmakers: list[dict],
    ) -> int:
        observed_at = _now_utc()
        bucket = observed_at[:13]
        rows: list[tuple[Any, ...]] = []
        for bookmaker in bookmakers:
            book = str(bookmaker.get("title", "")).strip()
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    selection = str(outcome.get("name", "")).strip()
                    try:
                        odds = float(outcome.get("price", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if not book or not selection or odds <= 1.01:
                        continue
                    rows.append((
                        observed_at, self.name, sport_key, event_name, home, away,
                        book, "h2h", selection, odds,
                        _hash(bucket, self.name, sport_key, event_name, book, selection, odds),
                    ))
        if not rows:
            return 0
        with closing(self._connect(settings)) as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO sport_odds_snapshots
                (captured_at, sport, league, event, home_team, away_team,
                 bookmaker, market, selection, odds, source_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return conn.total_changes - before

    def _save_baseline_candidate(
        self,
        settings: Settings,
        sport_key: str,
        event: dict,
        consensus: dict[str, float],
    ) -> int:
        """Store one transparent market benchmark, never a publishable bet."""
        external_id = str(event.get("id", "")).strip()
        if not external_id or not consensus:
            return 0
        selection = max(consensus, key=consensus.get)
        prices = [
            (bookmaker, odds)
            for bookmaker, outcome, odds in best_outlier_prices(event.get("bookmakers", []))
            if outcome == selection
        ]
        if not prices:
            return 0
        bookmaker, odds = max(prices, key=lambda item: item[1])
        home = str(event.get("home_team", "")).strip()
        away = str(event.get("away_team", "")).strip()
        with closing(self._connect(settings)) as conn:
            before = conn.total_changes
            conn.execute(
                """
                INSERT OR IGNORE INTO sport_shadow_candidates
                (created_at, sport, league, external_event_id, event, home_team,
                 away_team, market, selection, bookmaker, odds, probability,
                 start_time, model_version, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'h2h', ?, ?, ?, ?, ?,
                        'market_favourite_v1', 'OPEN')
                """,
                (
                    _now_utc(), self.name, sport_key, external_id,
                    f"{home} vs {away}", home, away, selection, bookmaker,
                    float(odds), float(consensus[selection]),
                    str(event.get("commence_time", "")),
                ),
            )
            conn.commit()
            return conn.total_changes - before

    def _save_observations(
        self,
        settings: Settings,
        sport_key: str,
        event: dict,
        consensus: dict[str, float],
    ) -> int:
        observed_at = _now_utc()
        bucket = observed_at[:13]
        home = str(event.get("home_team", "")).strip()
        away = str(event.get("away_team", "")).strip()
        event_name = f"{home} vs {away}"
        external_id = str(event.get("id", "")).strip()
        start_time = str(event.get("commence_time", "")).strip()
        rows: list[tuple[Any, ...]] = []
        for bookmaker, selection, odds in best_outlier_prices(event.get("bookmakers", [])):
            probability = consensus.get(selection)
            if probability is None or odds <= 1.01:
                continue
            rows.append((
                observed_at, self.name, sport_key, external_id, event_name,
                home, away, "h2h", selection, bookmaker, float(odds),
                float(probability), start_time, "SHADOW", "OPEN",
                _hash(bucket, self.name, sport_key, external_id or event_name, selection),
            ))
        if not rows:
            return 0
        with closing(self._connect(settings)) as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO sport_learning_observations
                (observed_at, sport, league, external_event_id, event, home_team,
                 away_team, market, selection, bookmaker, odds, market_probability,
                 start_time, mode, result, source_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return conn.total_changes - before

    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)
        self._ensure_tables(settings)
        configured = [
            value.strip()
            for value in os.getenv(
                "HANDBALL_SPORT_KEYS", "handball_germany_bundesliga"
            ).split(",")
            if value.strip()
        ]
        settlement = await settle_learning_observations(
            settings,
            sport=self.name,
            sport_keys=configured,
        )
        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active = await discover_active_sport_keys(
                settings.odds_api_key, ["Handball"]
            )
            configured = filter_active_keys(configured, active)

        events_scanned = snapshots_saved = observations_saved = candidates_saved = 0
        min_books = int(os.getenv("MIN_HANDBALL_BOOKMAKERS", "2"))
        for sport_key in configured:
            data = await fetch_odds(settings.odds_api_key, sport_key, markets="h2h")
            for event in data:
                events_scanned += 1
                home = str(event.get("home_team", "")).strip()
                away = str(event.get("away_team", "")).strip()
                event_name = f"{home} vs {away}"
                bookmakers = event.get("bookmakers", [])
                snapshots_saved += self._save_snapshot_rows(
                    settings, sport_key, event_name, home, away, bookmakers
                )
                consensus = consensus_h2h(bookmakers, min_books=min_books)
                if consensus:
                    observations_saved += self._save_observations(
                        settings, sport_key, event, consensus
                    )
                    candidates_saved += self._save_baseline_candidate(
                        settings, sport_key, event, consensus
                    )

        return SportResult(
            sport=self.name,
            mode="shadow",
            bets=[],
            message=(
                "Handball Engine 2.0 SHADOW: no publishable tips. "
                f"Events scanned: {events_scanned}. "
                f"Snapshots saved: {snapshots_saved}. "
                f"Learning observations saved: {observations_saved}."
                f" New benchmark candidates: {candidates_saved}."
                f" Settled observations: {settlement.settled_rows} "
                f"({settlement.won} won, {settlement.lost} lost)."
            ),
        )

