from __future__ import annotations

"""
Football Data Collector V14

Purpose:
- prepares historical feature inputs for Team xG and Meta AI
- stores closing market snapshots
- keeps pre-match feature history independent from settlement
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any


@dataclass
class ClosingLineSnapshot:
    match_id: str
    league: str
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float
    captured_at: str


class FootballDataCollectorV14:
    def __init__(self, db_file: str = "bets.db") -> None:
        self.db_file = Path(db_file)

    def init_tables(self) -> None:
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_closing_lines_v14 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    league TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_odds REAL,
                    draw_odds REAL,
                    away_odds REAL,
                    captured_at TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_xg_history_v14 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    league TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_xg REAL,
                    away_xg REAL,
                    created_at TEXT
                )
                """
            )

    def save_closing_line(
        self,
        snapshot: ClosingLineSnapshot,
    ) -> None:
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """
                INSERT INTO football_closing_lines_v14
                (
                    match_id,
                    league,
                    home_team,
                    away_team,
                    home_odds,
                    draw_odds,
                    away_odds,
                    captured_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.match_id,
                    snapshot.league,
                    snapshot.home_team,
                    snapshot.away_team,
                    snapshot.home_odds,
                    snapshot.draw_odds,
                    snapshot.away_odds,
                    snapshot.captured_at,
                ),
            )

    def save_xg_result(
        self,
        *,
        match_id: str,
        league: str,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
    ) -> None:
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """
                INSERT INTO football_xg_history_v14
                (
                    match_id,
                    league,
                    home_team,
                    away_team,
                    home_xg,
                    away_xg,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    league,
                    home_team,
                    away_team,
                    home_xg,
                    away_xg,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def status(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_file) as conn:
            closing = conn.execute(
                "SELECT COUNT(*) FROM football_closing_lines_v14"
            ).fetchone()[0]

            xg = conn.execute(
                "SELECT COUNT(*) FROM football_xg_history_v14"
            ).fetchone()[0]

        return {
            "closing_lines": closing,
            "xg_history": xg,
            "version": "v14",
        }


if __name__ == "__main__":
    collector = FootballDataCollectorV14()
    collector.init_tables()
    print(collector.status())
