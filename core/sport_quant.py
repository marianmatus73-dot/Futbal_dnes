from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from core.config import Settings


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(settings))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def norm(name: str) -> str:
    return " ".join(str(name).lower().strip().split())


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def init_sport_db(settings: Settings) -> None:
    with connect(settings) as conn:
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sport_elo_ratings (
                sport TEXT,
                entity TEXT,
                rating REAL,
                games INTEGER DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY(sport, entity)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_bets_sport ON sport_bets(sport, league, start_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_bets_result ON sport_bets(result)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_snapshots_event ON sport_odds_snapshots(sport, league, event, selection)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_snapshots_clv ON sport_odds_snapshots(sport, league, event, selection, captured_at)")


async def discover_active_sport_keys(api_key: str, groups: list[str]) -> set[str]:
    if not api_key:
        return set()

    url = "https://api.the-odds-api.com/v4/sports/"
    params = {"apiKey": api_key}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return set()
                data = await resp.json()
    except Exception:
        return set()

    wanted = {g.lower() for g in groups}
    keys: set[str] = set()

    for item in data if isinstance(data, list) else []:
        key = str(item.get("key", ""))
        group = str(item.get("group", "")).lower()
        active = bool(item.get("active", True))

        if key and active and group in wanted:
            keys.add(key)

    return keys


def filter_active_keys(configured_keys: list[str], active_keys: set[str]) -> list[str]:
    clean = [s.strip() for s in configured_keys if s.strip()]
    if not active_keys:
        return clean
    return [key for key in clean if key in active_keys]


def update_closing_lines(settings: Settings, sport: str) -> int:
    with connect(settings) as conn:
        rows = conn.execute("""
            SELECT id, league, event, selection, bookmaker, odds, start_time
            FROM sport_bets
            WHERE sport=?
              AND odds > 1.01
              AND (closing_odds IS NULL OR clv_pct IS NULL)
        """, (sport,)).fetchall()

    updates: list[tuple[float, float, int]] = []

    for bet_id, league, event, selection, bookmaker, taken_odds, start_time in rows:
        closing_odds = None

        with connect(settings) as conn:
            same_book = conn.execute("""
                SELECT odds
                FROM sport_odds_snapshots
                WHERE sport=? AND league=? AND event=? AND selection=? AND bookmaker=?
                  AND odds > 1.01
                  AND captured_at <= COALESCE(?, captured_at)
                ORDER BY captured_at DESC
                LIMIT 1
            """, (sport, league, event, selection, bookmaker, start_time)).fetchone()

            if same_book:
                closing_odds = float(same_book[0])
            else:
                market_avg = conn.execute("""
                    SELECT AVG(odds)
                    FROM (
                        SELECT bookmaker, odds, MAX(captured_at)
                        FROM sport_odds_snapshots
                        WHERE sport=? AND league=? AND event=? AND selection=?
                          AND odds > 1.01
                          AND captured_at <= COALESCE(?, captured_at)
                        GROUP BY bookmaker
                    )
                """, (sport, league, event, selection, start_time)).fetchone()

                if market_avg and market_avg[0]:
                    closing_odds = float(market_avg[0])

        if closing_odds and closing_odds > 1.01:
            clv_pct = (float(taken_odds) / closing_odds) - 1.0
            updates.append((round(closing_odds, 4), round(clv_pct, 5), int(bet_id)))

    if updates:
        with connect(settings) as conn:
            conn.executemany("""
                UPDATE sport_bets
                SET closing_odds=?, clv_pct=?
                WHERE id=?
            """, updates)

    return len(updates)


def refresh_bookmaker_stats(settings: Settings, sport: str) -> int:
    with connect(settings) as conn:
        rows = conn.execute("""
            SELECT bookmaker,
                   COUNT(*) AS bets,
                   SUM(CASE WHEN result='V' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN result='P' THEN 1 ELSE 0 END) AS losses,
                   SUM(stake) AS turnover,
                   SUM(CASE WHEN result='V' THEN stake * (odds - 1.0) ELSE -stake END) AS profit,
                   AVG(clv_pct) AS avg_clv
            FROM sport_bets
            WHERE sport=?
              AND result IN ('V','P')
              AND stake > 0
            GROUP BY bookmaker
        """, (sport,)).fetchall()

    if not rows:
        return 0

    payload = []
    for bookmaker, bets, wins, losses, turnover, profit, avg_clv in rows:
        payload.append((
            str(bookmaker), sport, int(bets or 0), int(wins or 0), int(losses or 0),
            float(turnover or 0.0), float(profit or 0.0), float(avg_clv or 0.0), now_utc()
        ))

    with connect(settings) as conn:
        conn.executemany("""
            INSERT INTO sport_bookmaker_stats
            (bookmaker, sport, bets, wins, losses, turnover, profit, avg_clv, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bookmaker, sport)
            DO UPDATE SET
                bets=excluded.bets,
                wins=excluded.wins,
                losses=excluded.losses,
                turnover=excluded.turnover,
                profit=excluded.profit,
                avg_clv=excluded.avg_clv,
                updated_at=excluded.updated_at
        """, payload)

    return len(payload)


def bookmaker_grade(settings: Settings, sport: str, bookmaker: str, min_samples: int = 20) -> float:
    with connect(settings) as conn:
        row = conn.execute("""
            SELECT bets, profit, turnover, avg_clv
            FROM sport_bookmaker_stats
            WHERE bookmaker=? AND sport=?
        """, (bookmaker, sport)).fetchone()

    if not row:
        return 1.0

    bets, profit, turnover, avg_clv = row
    if int(bets or 0) < min_samples:
        return 1.0

    yld = float(profit or 0.0) / max(float(turnover or 0.0), 1.0)
    clv = float(avg_clv or 0.0)
    grade = 1.0 + clv * 6.0 + yld * 2.0
    return round(max(0.60, min(1.40, grade)), 4)


def sport_analytics_report(settings: Settings, sport: str, days: int = 365) -> str:
    with connect(settings) as conn:
        summary = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN result='V' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN result='P' THEN 1 ELSE 0 END),
                   SUM(stake),
                   SUM(CASE WHEN result='V' THEN stake * (odds - 1.0) ELSE -stake END),
                   AVG(clv_pct)
            FROM sport_bets
            WHERE sport=?
              AND result IN ('V','P')
              AND created_at >= datetime('now', ?)
        """, (sport, f"-{int(days)} days")).fetchone()

        top_books = conn.execute("""
            SELECT bookmaker,
                   COUNT(*) AS bets,
                   SUM(CASE WHEN result='V' THEN stake * (odds - 1.0) ELSE -stake END) AS profit,
                   SUM(stake) AS turnover,
                   AVG(clv_pct) AS avg_clv
            FROM sport_bets
            WHERE sport=?
              AND result IN ('V','P')
              AND created_at >= datetime('now', ?)
            GROUP BY bookmaker
            HAVING bets >= 3
            ORDER BY profit DESC
            LIMIT 5
        """, (sport, f"-{int(days)} days")).fetchall()

        league_rows = conn.execute("""
            SELECT league,
                   COUNT(*) AS bets,
                   SUM(CASE WHEN result='V' THEN stake * (odds - 1.0) ELSE -stake END) AS profit,
                   SUM(stake) AS turnover,
                   AVG(clv_pct) AS avg_clv
            FROM sport_bets
            WHERE sport=?
              AND result IN ('V','P')
              AND created_at >= datetime('now', ?)
            GROUP BY league
            HAVING bets >= 3
            ORDER BY profit DESC
            LIMIT 5
        """, (sport, f"-{int(days)} days")).fetchall()

    total, wins, losses, turnover, profit, avg_clv = summary
    total = int(total or 0)
    wins = int(wins or 0)
    losses = int(losses or 0)
    turnover = float(turnover or 0.0)
    profit = float(profit or 0.0)
    avg_clv = float(avg_clv or 0.0)
    winrate = wins / total if total else 0.0
    yld = profit / turnover if turnover else 0.0

    lines = [
        f"{sport.upper()} ANALYTICS",
        f"Settled: {total} | W-L: {wins}-{losses} | Winrate: {winrate:.1%} | P/L: {profit:.2f} | Yield: {yld:.1%} | Avg CLV: {avg_clv:.2%}",
    ]

    if top_books:
        lines.append("Top bookmakers:")
        for bookmaker, bets, p, t, c in top_books:
            y = float(p or 0.0) / max(float(t or 0.0), 1.0)
            lines.append(f"- {bookmaker}: bets={bets}, P/L={float(p or 0.0):.2f}, yield={y:.1%}, CLV={float(c or 0.0):.2%}")

    if league_rows:
        lines.append("Top leagues/tours:")
        for league, bets, p, t, c in league_rows:
            y = float(p or 0.0) / max(float(t or 0.0), 1.0)
            lines.append(f"- {league}: bets={bets}, P/L={float(p or 0.0):.2f}, yield={y:.1%}, CLV={float(c or 0.0):.2%}")

    return "\n".join(lines)


def get_elo(settings: Settings, sport: str, entity: str, default: float = 1500.0) -> float:
    with connect(settings) as conn:
        row = conn.execute("""
            SELECT rating FROM sport_elo_ratings
            WHERE sport=? AND entity=?
        """, (sport, entity)).fetchone()
    return float(row[0]) if row else default


def set_elo(settings: Settings, sport: str, entity: str, rating: float, games_inc: int = 0) -> None:
    with connect(settings) as conn:
        conn.execute("""
            INSERT INTO sport_elo_ratings(sport, entity, rating, games, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sport, entity)
            DO UPDATE SET
                rating=excluded.rating,
                games=sport_elo_ratings.games + ?,
                updated_at=excluded.updated_at
        """, (sport, entity, float(rating), int(games_inc), now_utc(), int(games_inc)))


def elo_adjustment(settings: Settings, sport: str, home: str, away: str, selection: str) -> float:
    home_rating = get_elo(settings, sport, home)
    away_rating = get_elo(settings, sport, away)
    gap = home_rating - away_rating
    selected_gap = gap if norm(selection) == norm(home) else -gap
    return max(-0.035, min(0.035, selected_gap / 4000.0))


def update_elo_after_result(settings: Settings, sport: str, home: str, away: str, winner: str, k: float = 24.0, home_adv: float = 0.0) -> None:
    rh = get_elo(settings, sport, home)
    ra = get_elo(settings, sport, away)

    expected_home = 1.0 / (1.0 + 10.0 ** (-((rh + home_adv) - ra) / 400.0))
    actual_home = 1.0 if norm(winner) == norm(home) else 0.0
    delta = k * (actual_home - expected_home)

    set_elo(settings, sport, home, rh + delta, games_inc=1)
    set_elo(settings, sport, away, ra - delta, games_inc=1)


def poisson_hockey_adjustment(settings: Settings, home: str, away: str, selection: str) -> float:
    if os.getenv("HOCKEY_ELO_ENABLED", "1") != "1":
        return 0.0
    return elo_adjustment(settings, "hockey", home, away, selection)


def tennis_surface_adjustment(league: str) -> float:
    lower = league.lower()
    if "french_open" in lower or "monte_carlo" in lower or "madrid" in lower or "italian" in lower:
        return env_float("TENNIS_CLAY_EDGE_BONUS", 0.0)
    if "wimbledon" in lower:
        return env_float("TENNIS_GRASS_EDGE_BONUS", 0.0)
    return env_float("TENNIS_HARD_EDGE_BONUS", 0.0)
