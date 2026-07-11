from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from core.config import Settings
from core.market import consensus_h2h
from core.odds_api import fetch_odds


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def implied_probability(odds: float) -> float:
    odds = safe_float(odds, 0.0)

    if odds <= 1.0:
        return 0.0

    return 1.0 / odds


def make_source_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def normalize_probabilities(
    probabilities: Dict[str, float],
) -> Dict[str, float]:
    cleaned = {
        key: max(0.0, safe_float(value, 0.0))
        for key, value in probabilities.items()
    }

    total = sum(cleaned.values())

    if total <= 0:
        return cleaned

    return {
        key: value / total
        for key, value in cleaned.items()
    }


@dataclass
class MarketOutcome:
    selection: str
    bookmaker: str
    odds: float
    implied_probability: float


@dataclass
class FootballMarketSnapshot:
    sport_key: str
    league: str
    event: str
    home_team: str
    away_team: str
    commence_time: str
    captured_at: str

    consensus_home: float
    consensus_draw: float
    consensus_away: float

    best_home_odds: float
    best_draw_odds: float
    best_away_odds: float

    best_home_bookmaker: str
    best_draw_bookmaker: str
    best_away_bookmaker: str

    bookmaker_count: int
    overround: float

    outcomes: list[MarketOutcome]


@dataclass
class LineMovement:
    selection: str
    opening_odds: float
    latest_odds: float
    closing_odds: float | None

    opening_probability: float
    latest_probability: float
    closing_probability: float | None

    price_change_pct: float
    probability_change: float

    direction: str
    steam_move: bool


@dataclass
class CLVResult:
    selection: str
    bet_odds: float
    closing_odds: float
    clv_pct: float
    positive: bool


class FootballMarketDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    sport_key TEXT NOT NULL,
                    league TEXT NOT NULL,
                    event TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    commence_time TEXT NOT NULL,
                    captured_at TEXT NOT NULL,

                    bookmaker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    odds REAL NOT NULL,
                    implied_probability REAL NOT NULL,

                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_market_lookup
                ON football_market_snapshots (
                    sport_key,
                    event,
                    selection,
                    captured_at
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_market_closing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    sport_key TEXT NOT NULL,
                    league TEXT NOT NULL,
                    event TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    commence_time TEXT NOT NULL,

                    selection TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    closing_odds REAL NOT NULL,
                    closing_probability REAL NOT NULL,

                    captured_at TEXT NOT NULL,
                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_closing_lookup
                ON football_market_closing (
                    sport_key,
                    event,
                    selection,
                    captured_at
                )
                """
            )

            conn.commit()

    def save_event_snapshot(
        self,
        *,
        sport_key: str,
        league: str,
        event: dict,
    ) -> int:
        self.init_db()

        home_team = normalize_text(event.get("home_team"))
        away_team = normalize_text(event.get("away_team"))
        commence_time = normalize_text(event.get("commence_time"))
        event_name = f"{home_team} vs {away_team}"
        captured_at = now_utc()

        rows: list[tuple[Any, ...]] = []

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = normalize_text(bookmaker.get("title"))

            for market in bookmaker.get("markets", []):
                market_key = normalize_text(market.get("key"))

                if not market_key:
                    continue

                for outcome in market.get("outcomes", []):
                    selection = normalize_text(outcome.get("name"))
                    odds = safe_float(outcome.get("price"), 0.0)

                    if not selection or odds <= 1.01:
                        continue

                    probability = implied_probability(odds)

                    source_hash = make_source_hash(
                        captured_at,
                        sport_key,
                        event_name,
                        bookmaker_name,
                        market_key,
                        selection,
                        odds,
                    )

                    rows.append(
                        (
                            sport_key,
                            league,
                            event_name,
                            home_team,
                            away_team,
                            commence_time,
                            captured_at,
                            bookmaker_name,
                            market_key,
                            selection,
                            odds,
                            probability,
                            source_hash,
                            now_utc(),
                        )
                    )

        if not rows:
            return 0

        with self.connect() as conn:
            before = conn.total_changes

            conn.executemany(
                """
                INSERT OR IGNORE INTO football_market_snapshots (
                    sport_key,
                    league,
                    event,
                    home_team,
                    away_team,
                    commence_time,
                    captured_at,
                    bookmaker,
                    market,
                    selection,
                    odds,
                    implied_probability,
                    source_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            conn.commit()
            return conn.total_changes - before

    def save_closing_snapshot(
        self,
        *,
        sport_key: str,
        league: str,
        event: dict,
    ) -> int:
        self.init_db()

        home_team = normalize_text(event.get("home_team"))
        away_team = normalize_text(event.get("away_team"))
        commence_time = normalize_text(event.get("commence_time"))
        event_name = f"{home_team} vs {away_team}"
        captured_at = now_utc()

        rows: list[tuple[Any, ...]] = []

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = normalize_text(bookmaker.get("title"))

            for market in bookmaker.get("markets", []):
                if normalize_text(market.get("key")) != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    selection = normalize_text(outcome.get("name"))
                    odds = safe_float(outcome.get("price"), 0.0)

                    if not selection or odds <= 1.01:
                        continue

                    source_hash = make_source_hash(
                        sport_key,
                        event_name,
                        selection,
                        bookmaker_name,
                        captured_at,
                        "closing",
                    )

                    rows.append(
                        (
                            sport_key,
                            league,
                            event_name,
                            home_team,
                            away_team,
                            commence_time,
                            selection,
                            bookmaker_name,
                            odds,
                            implied_probability(odds),
                            captured_at,
                            source_hash,
                            now_utc(),
                        )
                    )

        if not rows:
            return 0

        with self.connect() as conn:
            before = conn.total_changes

            conn.executemany(
                """
                INSERT OR IGNORE INTO football_market_closing (
                    sport_key,
                    league,
                    event,
                    home_team,
                    away_team,
                    commence_time,
                    selection,
                    bookmaker,
                    closing_odds,
                    closing_probability,
                    captured_at,
                    source_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            conn.commit()
            return conn.total_changes - before

    def opening_odds(
        self,
        *,
        sport_key: str,
        event: str,
        selection: str,
    ) -> float | None:
        self.init_db()

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT odds
                FROM football_market_snapshots
                WHERE sport_key=?
                  AND event=?
                  AND selection=?
                  AND market='h2h'
                ORDER BY captured_at ASC, id ASC
                LIMIT 1
                """,
                (sport_key, event, selection),
            ).fetchone()

        if row is None:
            return None

        return safe_float(row["odds"], 0.0) or None

    def latest_odds(
        self,
        *,
        sport_key: str,
        event: str,
        selection: str,
    ) -> float | None:
        self.init_db()

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT odds
                FROM football_market_snapshots
                WHERE sport_key=?
                  AND event=?
                  AND selection=?
                  AND market='h2h'
                ORDER BY captured_at DESC, id DESC
                LIMIT 1
                """,
                (sport_key, event, selection),
            ).fetchone()

        if row is None:
            return None

        return safe_float(row["odds"], 0.0) or None

    def closing_odds(
        self,
        *,
        sport_key: str,
        event: str,
        selection: str,
    ) -> float | None:
        self.init_db()

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT closing_odds
                FROM football_market_closing
                WHERE sport_key=?
                  AND event=?
                  AND selection=?
                ORDER BY captured_at DESC, id DESC
                LIMIT 1
                """,
                (sport_key, event, selection),
            ).fetchone()

        if row is None:
            return None

        return safe_float(row["closing_odds"], 0.0) or None

    def line_movement(
        self,
        *,
        sport_key: str,
        event: str,
        selection: str,
        steam_threshold: float = 0.05,
    ) -> LineMovement | None:
        opening = self.opening_odds(
            sport_key=sport_key,
            event=event,
            selection=selection,
        )
        latest = self.latest_odds(
            sport_key=sport_key,
            event=event,
            selection=selection,
        )
        closing = self.closing_odds(
            sport_key=sport_key,
            event=event,
            selection=selection,
        )

        if opening is None or latest is None:
            return None

        opening_probability = implied_probability(opening)
        latest_probability = implied_probability(latest)
        closing_probability = (
            implied_probability(closing)
            if closing is not None
            else None
        )

        price_change_pct = (
            (latest - opening) / opening
            if opening > 0
            else 0.0
        )
        probability_change = latest_probability - opening_probability

        if latest < opening:
            direction = "SHORTENING"
        elif latest > opening:
            direction = "DRIFTING"
        else:
            direction = "FLAT"

        steam_move = (
            direction == "SHORTENING"
            and abs(price_change_pct) >= abs(steam_threshold)
        )

        return LineMovement(
            selection=selection,
            opening_odds=opening,
            latest_odds=latest,
            closing_odds=closing,
            opening_probability=opening_probability,
            latest_probability=latest_probability,
            closing_probability=closing_probability,
            price_change_pct=price_change_pct,
            probability_change=probability_change,
            direction=direction,
            steam_move=steam_move,
        )

    def calculate_clv(
        self,
        *,
        sport_key: str,
        event: str,
        selection: str,
        bet_odds: float,
    ) -> CLVResult | None:
        closing = self.closing_odds(
            sport_key=sport_key,
            event=event,
            selection=selection,
        )

        if closing is None or closing <= 1.0 or bet_odds <= 1.0:
            return None

        clv_pct = (bet_odds / closing - 1.0) * 100.0

        return CLVResult(
            selection=selection,
            bet_odds=bet_odds,
            closing_odds=closing,
            clv_pct=clv_pct,
            positive=clv_pct > 0,
        )

    def export_json(
        self,
        path: str = "exports/football_market_snapshots.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_market_snapshots
                    ORDER BY captured_at, id
                    """
                ).fetchall()
            ]

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return len(rows)


def build_market_snapshot(
    *,
    sport_key: str,
    league: str,
    event: dict,
    min_books: int = 3,
) -> FootballMarketSnapshot | None:
    home_team = normalize_text(event.get("home_team"))
    away_team = normalize_text(event.get("away_team"))
    commence_time = normalize_text(event.get("commence_time"))
    event_name = f"{home_team} vs {away_team}"
    bookmakers = event.get("bookmakers", [])

    consensus = consensus_h2h(
        bookmakers,
        min_books=max(1, int(min_books)),
    )

    if not consensus:
        return None

    normalized_consensus = normalize_probabilities(
        {
            home_team: consensus.get(home_team, 0.0),
            "Draw": consensus.get("Draw", consensus.get("draw", 0.0)),
            away_team: consensus.get(away_team, 0.0),
        }
    )

    best: dict[str, tuple[float, str]] = {
        home_team: (0.0, ""),
        "Draw": (0.0, ""),
        away_team: (0.0, ""),
    }
    outcomes: list[MarketOutcome] = []

    for bookmaker in bookmakers:
        bookmaker_name = normalize_text(bookmaker.get("title"))

        for market in bookmaker.get("markets", []):
            if normalize_text(market.get("key")) != "h2h":
                continue

            for outcome in market.get("outcomes", []):
                selection = normalize_text(outcome.get("name"))
                odds = safe_float(outcome.get("price"), 0.0)

                if odds <= 1.01:
                    continue

                outcomes.append(
                    MarketOutcome(
                        selection=selection,
                        bookmaker=bookmaker_name,
                        odds=odds,
                        implied_probability=implied_probability(odds),
                    )
                )

                canonical_selection = (
                    "Draw"
                    if selection.lower() in {"draw", "x", "remíza", "remiza"}
                    else selection
                )

                current_best = best.get(canonical_selection)

                if current_best is not None and odds > current_best[0]:
                    best[canonical_selection] = (
                        odds,
                        bookmaker_name,
                    )

    market_total = (
        implied_probability(best[home_team][0])
        + implied_probability(best["Draw"][0])
        + implied_probability(best[away_team][0])
    )

    overround = max(0.0, market_total - 1.0)

    return FootballMarketSnapshot(
        sport_key=sport_key,
        league=league,
        event=event_name,
        home_team=home_team,
        away_team=away_team,
        commence_time=commence_time,
        captured_at=now_utc(),

        consensus_home=normalized_consensus.get(home_team, 0.0),
        consensus_draw=normalized_consensus.get("Draw", 0.0),
        consensus_away=normalized_consensus.get(away_team, 0.0),

        best_home_odds=best[home_team][0],
        best_draw_odds=best["Draw"][0],
        best_away_odds=best[away_team][0],

        best_home_bookmaker=best[home_team][1],
        best_draw_bookmaker=best["Draw"][1],
        best_away_bookmaker=best[away_team][1],

        bookmaker_count=len(bookmakers),
        overround=overround,

        outcomes=outcomes,
    )


async def fetch_football_market(
    settings: Settings,
    *,
    sport_key: str,
    league: str,
    min_books: int = 3,
    save_snapshots: bool = True,
) -> list[FootballMarketSnapshot]:
    data = await fetch_odds(
        settings.odds_api_key,
        sport_key,
        markets="h2h",
    )

    if not data:
        return []

    database = FootballMarketDatabase(settings)
    results: list[FootballMarketSnapshot] = []

    for event in data:
        if save_snapshots:
            database.save_event_snapshot(
                sport_key=sport_key,
                league=league,
                event=event,
            )

        snapshot = build_market_snapshot(
            sport_key=sport_key,
            league=league,
            event=event,
            min_books=min_books,
        )

        if snapshot is not None:
            results.append(snapshot)

    return results
