from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_HOME_ADVANTAGE_XG = 1.08
DEFAULT_HOME_ADVANTAGE_ELO = 65.0
DEFAULT_LEAGUE_AVG_XG = 1.35
DEFAULT_DIXON_RHO = -0.08


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class LeagueCalibration:
    league: str
    matches: int

    avg_home_goals: float
    avg_away_goals: float
    avg_total_goals: float

    home_win_rate: float
    draw_rate: float
    away_win_rate: float

    low_score_rate: float
    btts_rate: float
    over_25_rate: float

    home_advantage_xg: float
    home_advantage_elo: float
    league_average_xg: float
    dixon_rho: float

    reliability: float
    updated_at: str

    def normalized(self) -> "LeagueCalibration":
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"
        self.matches = max(0, safe_int(self.matches))

        self.avg_home_goals = clamp(
            safe_float(self.avg_home_goals, DEFAULT_LEAGUE_AVG_XG),
            0.0,
            6.0,
        )
        self.avg_away_goals = clamp(
            safe_float(self.avg_away_goals, DEFAULT_LEAGUE_AVG_XG),
            0.0,
            6.0,
        )
        self.avg_total_goals = clamp(
            safe_float(
                self.avg_total_goals,
                self.avg_home_goals + self.avg_away_goals,
            ),
            0.0,
            10.0,
        )

        for field_name in (
            "home_win_rate",
            "draw_rate",
            "away_win_rate",
            "low_score_rate",
            "btts_rate",
            "over_25_rate",
            "reliability",
        ):
            setattr(
                self,
                field_name,
                clamp(
                    safe_float(getattr(self, field_name), 0.0),
                    0.0,
                    1.0,
                ),
            )

        self.home_advantage_xg = clamp(
            safe_float(
                self.home_advantage_xg,
                DEFAULT_HOME_ADVANTAGE_XG,
            ),
            0.95,
            1.20,
        )
        self.home_advantage_elo = clamp(
            safe_float(
                self.home_advantage_elo,
                DEFAULT_HOME_ADVANTAGE_ELO,
            ),
            0.0,
            140.0,
        )
        self.league_average_xg = clamp(
            safe_float(
                self.league_average_xg,
                DEFAULT_LEAGUE_AVG_XG,
            ),
            0.70,
            2.20,
        )
        self.dixon_rho = clamp(
            safe_float(self.dixon_rho, DEFAULT_DIXON_RHO),
            -0.25,
            0.25,
        )

        if not self.updated_at:
            self.updated_at = now_utc()

        return self


class FootballLeagueCalibrationDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_league_calibration (
                    league TEXT PRIMARY KEY,
                    matches INTEGER NOT NULL DEFAULT 0,

                    avg_home_goals REAL NOT NULL DEFAULT 1.35,
                    avg_away_goals REAL NOT NULL DEFAULT 1.35,
                    avg_total_goals REAL NOT NULL DEFAULT 2.70,

                    home_win_rate REAL NOT NULL DEFAULT 0.37,
                    draw_rate REAL NOT NULL DEFAULT 0.26,
                    away_win_rate REAL NOT NULL DEFAULT 0.37,

                    low_score_rate REAL NOT NULL DEFAULT 0.38,
                    btts_rate REAL NOT NULL DEFAULT 0.50,
                    over_25_rate REAL NOT NULL DEFAULT 0.50,

                    home_advantage_xg REAL NOT NULL DEFAULT 1.08,
                    home_advantage_elo REAL NOT NULL DEFAULT 65.0,
                    league_average_xg REAL NOT NULL DEFAULT 1.35,
                    dixon_rho REAL NOT NULL DEFAULT -0.08,

                    reliability REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, calibration: LeagueCalibration) -> None:
        self.init_db()
        calibration = calibration.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_league_calibration (
                    league,
                    matches,
                    avg_home_goals,
                    avg_away_goals,
                    avg_total_goals,
                    home_win_rate,
                    draw_rate,
                    away_win_rate,
                    low_score_rate,
                    btts_rate,
                    over_25_rate,
                    home_advantage_xg,
                    home_advantage_elo,
                    league_average_xg,
                    dixon_rho,
                    reliability,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(league) DO UPDATE SET
                    matches=excluded.matches,
                    avg_home_goals=excluded.avg_home_goals,
                    avg_away_goals=excluded.avg_away_goals,
                    avg_total_goals=excluded.avg_total_goals,
                    home_win_rate=excluded.home_win_rate,
                    draw_rate=excluded.draw_rate,
                    away_win_rate=excluded.away_win_rate,
                    low_score_rate=excluded.low_score_rate,
                    btts_rate=excluded.btts_rate,
                    over_25_rate=excluded.over_25_rate,
                    home_advantage_xg=excluded.home_advantage_xg,
                    home_advantage_elo=excluded.home_advantage_elo,
                    league_average_xg=excluded.league_average_xg,
                    dixon_rho=excluded.dixon_rho,
                    reliability=excluded.reliability,
                    updated_at=excluded.updated_at
                """,
                (
                    calibration.league,
                    calibration.matches,
                    calibration.avg_home_goals,
                    calibration.avg_away_goals,
                    calibration.avg_total_goals,
                    calibration.home_win_rate,
                    calibration.draw_rate,
                    calibration.away_win_rate,
                    calibration.low_score_rate,
                    calibration.btts_rate,
                    calibration.over_25_rate,
                    calibration.home_advantage_xg,
                    calibration.home_advantage_elo,
                    calibration.league_average_xg,
                    calibration.dixon_rho,
                    calibration.reliability,
                    calibration.updated_at,
                ),
            )
            conn.commit()

    def load(self, league: str) -> LeagueCalibration:
        self.init_db()

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_league_calibration
                WHERE league=?
                """,
                (league,),
            ).fetchone()

        if row is None:
            return LeagueCalibration(
                league=league,
                matches=0,
                avg_home_goals=DEFAULT_LEAGUE_AVG_XG,
                avg_away_goals=DEFAULT_LEAGUE_AVG_XG,
                avg_total_goals=DEFAULT_LEAGUE_AVG_XG * 2.0,
                home_win_rate=0.37,
                draw_rate=0.26,
                away_win_rate=0.37,
                low_score_rate=0.38,
                btts_rate=0.50,
                over_25_rate=0.50,
                home_advantage_xg=DEFAULT_HOME_ADVANTAGE_XG,
                home_advantage_elo=DEFAULT_HOME_ADVANTAGE_ELO,
                league_average_xg=DEFAULT_LEAGUE_AVG_XG,
                dixon_rho=DEFAULT_DIXON_RHO,
                reliability=0.0,
                updated_at=now_utc(),
            ).normalized()

        return LeagueCalibration(**dict(row)).normalized()

    def rebuild_league(self, league: str) -> LeagueCalibration:
        self.init_db()

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    home_goals,
                    away_goals
                FROM football_form_history
                WHERE league=?
                ORDER BY played_at DESC, id DESC
                """,
                (league,),
            ).fetchall()

        matches = len(rows)

        if matches == 0:
            calibration = self.load(league)
            self.save(calibration)
            return calibration

        home_goals = sum(safe_int(row["home_goals"]) for row in rows)
        away_goals = sum(safe_int(row["away_goals"]) for row in rows)

        home_wins = sum(
            1
            for row in rows
            if safe_int(row["home_goals"]) > safe_int(row["away_goals"])
        )
        draws = sum(
            1
            for row in rows
            if safe_int(row["home_goals"]) == safe_int(row["away_goals"])
        )
        away_wins = matches - home_wins - draws

        low_scores = sum(
            1
            for row in rows
            if safe_int(row["home_goals"]) + safe_int(row["away_goals"]) <= 2
        )
        btts = sum(
            1
            for row in rows
            if safe_int(row["home_goals"]) > 0
            and safe_int(row["away_goals"]) > 0
        )
        over_25 = sum(
            1
            for row in rows
            if safe_int(row["home_goals"]) + safe_int(row["away_goals"]) >= 3
        )

        avg_home = home_goals / matches
        avg_away = away_goals / matches
        avg_total = avg_home + avg_away

        home_win_rate = home_wins / matches
        draw_rate = draws / matches
        away_win_rate = away_wins / matches
        low_score_rate = low_scores / matches
        btts_rate = btts / matches
        over_25_rate = over_25 / matches

        reliability = clamp(matches / 100.0, 0.0, 1.0)

        goal_ratio = (
            avg_home / avg_away
            if avg_away > 0.05
            else 1.0
        )

        learned_home_advantage_xg = clamp(
            1.0 + (goal_ratio - 1.0) * 0.25,
            0.98,
            1.18,
        )
        learned_home_advantage_elo = clamp(
            (home_win_rate - away_win_rate) * 220.0 + 55.0,
            0.0,
            130.0,
        )

        learned_rho = clamp(
            DEFAULT_DIXON_RHO
            - (draw_rate - 0.26) * 0.35
            - (low_score_rate - 0.38) * 0.20,
            -0.25,
            0.25,
        )

        calibration = LeagueCalibration(
            league=league,
            matches=matches,
            avg_home_goals=avg_home,
            avg_away_goals=avg_away,
            avg_total_goals=avg_total,
            home_win_rate=home_win_rate,
            draw_rate=draw_rate,
            away_win_rate=away_win_rate,
            low_score_rate=low_score_rate,
            btts_rate=btts_rate,
            over_25_rate=over_25_rate,
            home_advantage_xg=(
                learned_home_advantage_xg * reliability
                + DEFAULT_HOME_ADVANTAGE_XG * (1.0 - reliability)
            ),
            home_advantage_elo=(
                learned_home_advantage_elo * reliability
                + DEFAULT_HOME_ADVANTAGE_ELO * (1.0 - reliability)
            ),
            league_average_xg=(
                (avg_total / 2.0) * reliability
                + DEFAULT_LEAGUE_AVG_XG * (1.0 - reliability)
            ),
            dixon_rho=(
                learned_rho * reliability
                + DEFAULT_DIXON_RHO * (1.0 - reliability)
            ),
            reliability=reliability,
            updated_at=now_utc(),
        ).normalized()

        self.save(calibration)
        return calibration

    def rebuild_all(self) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT league
                FROM football_form_history
                WHERE league IS NOT NULL
                  AND TRIM(league) <> ''
                """
            ).fetchall()

        leagues = [str(row["league"]) for row in rows]

        for league in leagues:
            self.rebuild_league(league)

        return len(leagues)

    def export_json(
        self,
        path: str = "exports/football_league_calibration.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_league_calibration
                    ORDER BY league
                    """
                ).fetchall()
            ]

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return len(rows)


def get_league_calibration(
    settings: Settings,
    league: str,
) -> LeagueCalibration:
    database = FootballLeagueCalibrationDatabase(settings)
    return database.load(league)


def rebuild_football_league_calibrations(
    settings: Settings,
) -> int:
    database = FootballLeagueCalibrationDatabase(settings)
    return database.rebuild_all()


if __name__ == "__main__":
    settings = Settings.from_env()
    database = FootballLeagueCalibrationDatabase(settings)

    rebuilt = database.rebuild_all()
    exported = database.export_json()

    print(
        "Football league calibration finished: "
        f"rebuilt={rebuilt}, exported={exported}"
    )
