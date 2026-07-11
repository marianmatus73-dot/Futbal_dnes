from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.config import Settings


DEFAULT_XG = 1.35
DEFAULT_ATTACK_RATING = 1.0
DEFAULT_DEFENSE_RATING = 1.0
DEFAULT_HOME_ADVANTAGE = 1.08
DEFAULT_LEARNING_RATE = 0.12
DEFAULT_DECAY = 0.90
MIN_EXPECTED_GOALS = 0.15
MAX_EXPECTED_GOALS = 4.50


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


def normalize_team_name(team: str) -> str:
    return " ".join(str(team or "").strip().split())


@dataclass
class TeamXG:
    team: str
    league: str = "UNKNOWN"

    attack_rating: float = DEFAULT_ATTACK_RATING
    defense_rating: float = DEFAULT_DEFENSE_RATING

    rolling_xg_for_5: float = DEFAULT_XG
    rolling_xga_5: float = DEFAULT_XG
    rolling_xg_for_10: float = DEFAULT_XG
    rolling_xga_10: float = DEFAULT_XG

    home_xg_for: float = DEFAULT_XG
    home_xga: float = DEFAULT_XG
    away_xg_for: float = DEFAULT_XG
    away_xga: float = DEFAULT_XG

    matches: int = 0
    home_matches: int = 0
    away_matches: int = 0

    xg_for_sum: float = 0.0
    xga_sum: float = 0.0

    last_xg_for: float = DEFAULT_XG
    last_xga: float = DEFAULT_XG
    last_updated: str = ""

    def normalized(self) -> "TeamXG":
        self.team = normalize_team_name(self.team)
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"

        self.attack_rating = clamp(
            safe_float(self.attack_rating, DEFAULT_ATTACK_RATING),
            0.25,
            3.00,
        )
        self.defense_rating = clamp(
            safe_float(self.defense_rating, DEFAULT_DEFENSE_RATING),
            0.25,
            3.00,
        )

        for field_name in (
            "rolling_xg_for_5",
            "rolling_xga_5",
            "rolling_xg_for_10",
            "rolling_xga_10",
            "home_xg_for",
            "home_xga",
            "away_xg_for",
            "away_xga",
            "last_xg_for",
            "last_xga",
        ):
            setattr(
                self,
                field_name,
                clamp(
                    safe_float(getattr(self, field_name), DEFAULT_XG),
                    0.05,
                    6.00,
                ),
            )

        self.matches = max(0, int(self.matches or 0))
        self.home_matches = max(0, int(self.home_matches or 0))
        self.away_matches = max(0, int(self.away_matches or 0))
        self.xg_for_sum = max(0.0, safe_float(self.xg_for_sum, 0.0))
        self.xga_sum = max(0.0, safe_float(self.xga_sum, 0.0))

        if not self.last_updated:
            self.last_updated = now_utc()

        return self

    @property
    def average_xg_for(self) -> float:
        if self.matches <= 0:
            return DEFAULT_XG

        return self.xg_for_sum / self.matches

    @property
    def average_xga(self) -> float:
        if self.matches <= 0:
            return DEFAULT_XG

        return self.xga_sum / self.matches

    @property
    def sample_reliability(self) -> float:
        return clamp(self.matches / 30.0, 0.0, 1.0)


@dataclass
class MatchXGEstimate:
    home_team: str
    away_team: str
    league: str

    home_expected_goals: float
    away_expected_goals: float

    home_attack_component: float
    home_defense_component: float
    away_attack_component: float
    away_defense_component: float

    home_reliability: float
    away_reliability: float

    reason: str


class FootballXGDatabase:
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
                CREATE TABLE IF NOT EXISTS football_xg_ratings (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,

                    attack_rating REAL NOT NULL DEFAULT 1.0,
                    defense_rating REAL NOT NULL DEFAULT 1.0,

                    rolling_xg_for_5 REAL NOT NULL DEFAULT 1.35,
                    rolling_xga_5 REAL NOT NULL DEFAULT 1.35,
                    rolling_xg_for_10 REAL NOT NULL DEFAULT 1.35,
                    rolling_xga_10 REAL NOT NULL DEFAULT 1.35,

                    home_xg_for REAL NOT NULL DEFAULT 1.35,
                    home_xga REAL NOT NULL DEFAULT 1.35,
                    away_xg_for REAL NOT NULL DEFAULT 1.35,
                    away_xga REAL NOT NULL DEFAULT 1.35,

                    matches INTEGER NOT NULL DEFAULT 0,
                    home_matches INTEGER NOT NULL DEFAULT 0,
                    away_matches INTEGER NOT NULL DEFAULT 0,

                    xg_for_sum REAL NOT NULL DEFAULT 0.0,
                    xga_sum REAL NOT NULL DEFAULT 0.0,

                    last_xg_for REAL NOT NULL DEFAULT 1.35,
                    last_xga REAL NOT NULL DEFAULT 1.35,
                    last_updated TEXT NOT NULL,

                    PRIMARY KEY (team, league)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_xg_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    played_at TEXT NOT NULL,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_xg REAL NOT NULL,
                    away_xg REAL NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    source TEXT NOT NULL DEFAULT 'manual',
                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_xg_history_teams
                ON football_xg_history (
                    league,
                    home_team,
                    away_team,
                    played_at
                )
                """
            )

            conn.commit()

    def load_team(self, team: str, league: str = "UNKNOWN") -> TeamXG:
        self.init_db()

        team = normalize_team_name(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_xg_ratings
                WHERE team=? AND league=?
                """,
                (team, league),
            ).fetchone()

        if row is None:
            return TeamXG(
                team=team,
                league=league,
                last_updated=now_utc(),
            ).normalized()

        return TeamXG(**dict(row)).normalized()

    def save_team(self, rating: TeamXG) -> None:
        self.init_db()
        rating = rating.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_xg_ratings (
                    team,
                    league,
                    attack_rating,
                    defense_rating,
                    rolling_xg_for_5,
                    rolling_xga_5,
                    rolling_xg_for_10,
                    rolling_xga_10,
                    home_xg_for,
                    home_xga,
                    away_xg_for,
                    away_xga,
                    matches,
                    home_matches,
                    away_matches,
                    xg_for_sum,
                    xga_sum,
                    last_xg_for,
                    last_xga,
                    last_updated
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(team, league) DO UPDATE SET
                    attack_rating=excluded.attack_rating,
                    defense_rating=excluded.defense_rating,
                    rolling_xg_for_5=excluded.rolling_xg_for_5,
                    rolling_xga_5=excluded.rolling_xga_5,
                    rolling_xg_for_10=excluded.rolling_xg_for_10,
                    rolling_xga_10=excluded.rolling_xga_10,
                    home_xg_for=excluded.home_xg_for,
                    home_xga=excluded.home_xga,
                    away_xg_for=excluded.away_xg_for,
                    away_xga=excluded.away_xga,
                    matches=excluded.matches,
                    home_matches=excluded.home_matches,
                    away_matches=excluded.away_matches,
                    xg_for_sum=excluded.xg_for_sum,
                    xga_sum=excluded.xga_sum,
                    last_xg_for=excluded.last_xg_for,
                    last_xga=excluded.last_xga,
                    last_updated=excluded.last_updated
                """,
                (
                    rating.team,
                    rating.league,
                    rating.attack_rating,
                    rating.defense_rating,
                    rating.rolling_xg_for_5,
                    rating.rolling_xga_5,
                    rating.rolling_xg_for_10,
                    rating.rolling_xga_10,
                    rating.home_xg_for,
                    rating.home_xga,
                    rating.away_xg_for,
                    rating.away_xga,
                    rating.matches,
                    rating.home_matches,
                    rating.away_matches,
                    rating.xg_for_sum,
                    rating.xga_sum,
                    rating.last_xg_for,
                    rating.last_xga,
                    rating.last_updated,
                ),
            )
            conn.commit()

    def estimate_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        league_average_xg: float = DEFAULT_XG,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    ) -> MatchXGEstimate:
        home = self.load_team(home_team, league)
        away = self.load_team(away_team, league)

        league_average_xg = clamp(
            safe_float(league_average_xg, DEFAULT_XG),
            0.50,
            3.50,
        )
        home_advantage = clamp(
            safe_float(home_advantage, DEFAULT_HOME_ADVANTAGE),
            0.90,
            1.30,
        )

        home_attack_component = (
            home.attack_rating * 0.40
            + (home.rolling_xg_for_5 / league_average_xg) * 0.25
            + (home.rolling_xg_for_10 / league_average_xg) * 0.20
            + (home.home_xg_for / league_average_xg) * 0.15
        )

        away_defense_component = (
            away.defense_rating * 0.45
            + (away.rolling_xga_5 / league_average_xg) * 0.25
            + (away.rolling_xga_10 / league_average_xg) * 0.20
            + (away.away_xga / league_average_xg) * 0.10
        )

        away_attack_component = (
            away.attack_rating * 0.40
            + (away.rolling_xg_for_5 / league_average_xg) * 0.25
            + (away.rolling_xg_for_10 / league_average_xg) * 0.20
            + (away.away_xg_for / league_average_xg) * 0.15
        )

        home_defense_component = (
            home.defense_rating * 0.45
            + (home.rolling_xga_5 / league_average_xg) * 0.25
            + (home.rolling_xga_10 / league_average_xg) * 0.20
            + (home.home_xga / league_average_xg) * 0.10
        )

        raw_home_xg = (
            league_average_xg
            * home_attack_component
            * away_defense_component
            * home_advantage
        )

        raw_away_xg = (
            league_average_xg
            * away_attack_component
            * home_defense_component
            / max(0.95, home_advantage)
        )

        home_reliability = home.sample_reliability
        away_reliability = away.sample_reliability
        combined_reliability = (home_reliability + away_reliability) / 2.0

        shrink = 0.35 + combined_reliability * 0.55

        home_expected_goals = (
            raw_home_xg * shrink
            + league_average_xg * home_advantage * (1.0 - shrink)
        )
        away_expected_goals = (
            raw_away_xg * shrink
            + league_average_xg / home_advantage * (1.0 - shrink)
        )

        home_expected_goals = clamp(
            home_expected_goals,
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS,
        )
        away_expected_goals = clamp(
            away_expected_goals,
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS,
        )

        reason = (
            f"xG estimate: home={home_expected_goals:.2f}, "
            f"away={away_expected_goals:.2f}; "
            f"home_attack={home_attack_component:.3f}; "
            f"away_defense={away_defense_component:.3f}; "
            f"away_attack={away_attack_component:.3f}; "
            f"home_defense={home_defense_component:.3f}; "
            f"reliability={combined_reliability:.2f}"
        )

        return MatchXGEstimate(
            home_team=home.team,
            away_team=away.team,
            league=league,
            home_expected_goals=home_expected_goals,
            away_expected_goals=away_expected_goals,
            home_attack_component=home_attack_component,
            home_defense_component=home_defense_component,
            away_attack_component=away_attack_component,
            away_defense_component=away_defense_component,
            home_reliability=home_reliability,
            away_reliability=away_reliability,
            reason=reason,
        )

    def update_after_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        home_goals: int | None = None,
        away_goals: int | None = None,
        played_at: str | None = None,
        source: str = "manual",
        source_hash: str | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        decay_5: float = 0.70,
        decay_10: float = DEFAULT_DECAY,
    ) -> bool:
        self.init_db()

        league = str(league or "UNKNOWN").strip() or "UNKNOWN"
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        home_xg = clamp(safe_float(home_xg, DEFAULT_XG), 0.0, 8.0)
        away_xg = clamp(safe_float(away_xg, DEFAULT_XG), 0.0, 8.0)
        learning_rate = clamp(
            safe_float(learning_rate, DEFAULT_LEARNING_RATE),
            0.01,
            0.50,
        )
        decay_5 = clamp(safe_float(decay_5, 0.70), 0.40, 0.95)
        decay_10 = clamp(safe_float(decay_10, DEFAULT_DECAY), 0.70, 0.98)
        played_at = played_at or now_utc()

        if source_hash:
            with self.connect() as conn:
                exists = conn.execute(
                    """
                    SELECT 1
                    FROM football_xg_history
                    WHERE source_hash=?
                    """,
                    (source_hash,),
                ).fetchone()

            if exists:
                return False

        home = self.load_team(home_team, league)
        away = self.load_team(away_team, league)

        league_baseline = DEFAULT_XG

        home_attack_observed = home_xg / league_baseline
        home_defense_observed = away_xg / league_baseline
        away_attack_observed = away_xg / league_baseline
        away_defense_observed = home_xg / league_baseline

        home.attack_rating = (
            home.attack_rating * (1.0 - learning_rate)
            + home_attack_observed * learning_rate
        )
        home.defense_rating = (
            home.defense_rating * (1.0 - learning_rate)
            + home_defense_observed * learning_rate
        )
        away.attack_rating = (
            away.attack_rating * (1.0 - learning_rate)
            + away_attack_observed * learning_rate
        )
        away.defense_rating = (
            away.defense_rating * (1.0 - learning_rate)
            + away_defense_observed * learning_rate
        )

        home.rolling_xg_for_5 = (
            home.rolling_xg_for_5 * decay_5
            + home_xg * (1.0 - decay_5)
        )
        home.rolling_xga_5 = (
            home.rolling_xga_5 * decay_5
            + away_xg * (1.0 - decay_5)
        )
        home.rolling_xg_for_10 = (
            home.rolling_xg_for_10 * decay_10
            + home_xg * (1.0 - decay_10)
        )
        home.rolling_xga_10 = (
            home.rolling_xga_10 * decay_10
            + away_xg * (1.0 - decay_10)
        )

        away.rolling_xg_for_5 = (
            away.rolling_xg_for_5 * decay_5
            + away_xg * (1.0 - decay_5)
        )
        away.rolling_xga_5 = (
            away.rolling_xga_5 * decay_5
            + home_xg * (1.0 - decay_5)
        )
        away.rolling_xg_for_10 = (
            away.rolling_xg_for_10 * decay_10
            + away_xg * (1.0 - decay_10)
        )
        away.rolling_xga_10 = (
            away.rolling_xga_10 * decay_10
            + home_xg * (1.0 - decay_10)
        )

        home.home_xg_for = (
            home.home_xg_for * decay_10
            + home_xg * (1.0 - decay_10)
        )
        home.home_xga = (
            home.home_xga * decay_10
            + away_xg * (1.0 - decay_10)
        )
        away.away_xg_for = (
            away.away_xg_for * decay_10
            + away_xg * (1.0 - decay_10)
        )
        away.away_xga = (
            away.away_xga * decay_10
            + home_xg * (1.0 - decay_10)
        )

        home.matches += 1
        home.home_matches += 1
        home.xg_for_sum += home_xg
        home.xga_sum += away_xg
        home.last_xg_for = home_xg
        home.last_xga = away_xg
        home.last_updated = now_utc()

        away.matches += 1
        away.away_matches += 1
        away.xg_for_sum += away_xg
        away.xga_sum += home_xg
        away.last_xg_for = away_xg
        away.last_xga = home_xg
        away.last_updated = now_utc()

        self.save_team(home)
        self.save_team(away)

        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO football_xg_history (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_xg,
                    away_xg,
                    home_goals,
                    away_goals,
                    source,
                    source_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    played_at,
                    league,
                    home_team,
                    away_team,
                    home_xg,
                    away_xg,
                    home_goals,
                    away_goals,
                    source,
                    source_hash,
                    now_utc(),
                ),
            )
            conn.commit()

        return True

    def export_json(self, path: str = "exports/football_xg_ratings.json") -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_xg_ratings
                    ORDER BY league, team
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

    def import_json(self, path: str = "exports/football_xg_ratings.json") -> int:
        file_path = Path(path)

        if not file_path.exists():
            return 0

        data = json.loads(file_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            return 0

        imported = 0

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                rating = TeamXG(**item).normalized()
                self.save_team(rating)
                imported += 1
            except Exception:
                continue

        return imported

    def league_table(self, league: str) -> list[TeamXG]:
        self.init_db()

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM football_xg_ratings
                WHERE league=?
                ORDER BY attack_rating DESC, defense_rating ASC
                """,
                (league,),
            ).fetchall()

        return [
            TeamXG(**dict(row)).normalized()
            for row in rows
        ]


def estimate_match_xg(
    settings: Settings,
    home_team: str,
    away_team: str,
    league: str,
    league_average_xg: float = DEFAULT_XG,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> MatchXGEstimate:
    database = FootballXGDatabase(settings)

    return database.estimate_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
        league_average_xg=league_average_xg,
        home_advantage=home_advantage,
    )


def update_xg_after_match(
    settings: Settings,
    **kwargs: Any,
) -> bool:
    database = FootballXGDatabase(settings)
    return database.update_after_match(**kwargs)
