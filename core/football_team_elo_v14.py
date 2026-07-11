from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_ELO = 1500.0
DEFAULT_HOME_ADVANTAGE = 65.0
DEFAULT_K_FACTOR = 24.0
MAX_HISTORY_MATCHES = 100


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


def normalize_team(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


@dataclass
class TeamEloV14:
    team: str
    league: str

    rating: float = DEFAULT_ELO
    home_rating: float = DEFAULT_ELO
    away_rating: float = DEFAULT_ELO

    matches: int = 0
    home_matches: int = 0
    away_matches: int = 0

    wins: int = 0
    draws: int = 0
    losses: int = 0

    goals_for: int = 0
    goals_against: int = 0

    recent_form: float = 0.50
    reliability: float = 0.0
    uncertainty: float = 1.0

    last_match_at: str = ""
    last_updated: str = ""

    def normalized(self) -> "TeamEloV14":
        self.team = normalize_team(self.team)
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"

        self.rating = clamp(
            safe_float(self.rating, DEFAULT_ELO),
            900.0,
            2500.0,
        )
        self.home_rating = clamp(
            safe_float(self.home_rating, self.rating),
            900.0,
            2500.0,
        )
        self.away_rating = clamp(
            safe_float(self.away_rating, self.rating),
            900.0,
            2500.0,
        )

        for field_name in (
            "matches",
            "home_matches",
            "away_matches",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
        ):
            setattr(
                self,
                field_name,
                max(0, safe_int(getattr(self, field_name))),
            )

        self.recent_form = clamp(
            safe_float(self.recent_form, 0.50),
            0.0,
            1.0,
        )
        self.reliability = clamp(
            safe_float(self.reliability, 0.0),
            0.0,
            1.0,
        )
        self.uncertainty = clamp(
            safe_float(self.uncertainty, 1.0),
            0.0,
            1.0,
        )

        if not self.last_updated:
            self.last_updated = now_utc()

        return self


@dataclass
class EloV14Prediction:
    league: str
    home_team: str
    away_team: str

    home_probability: float
    draw_probability: float
    away_probability: float

    home_rating: float
    away_rating: float
    rating_difference: float

    home_reliability: float
    away_reliability: float
    combined_reliability: float

    uncertainty_penalty: float
    reason: str


@dataclass
class EloV14Update:
    inserted: bool

    home_old_rating: float
    away_old_rating: float
    home_new_rating: float
    away_new_rating: float

    home_change: float
    away_change: float

    k_factor: float
    expected_home: float
    actual_home: float


class FootballTeamEloV14Database:
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
                CREATE TABLE IF NOT EXISTS football_team_elo_v14 (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,

                    rating REAL NOT NULL DEFAULT 1500.0,
                    home_rating REAL NOT NULL DEFAULT 1500.0,
                    away_rating REAL NOT NULL DEFAULT 1500.0,

                    matches INTEGER NOT NULL DEFAULT 0,
                    home_matches INTEGER NOT NULL DEFAULT 0,
                    away_matches INTEGER NOT NULL DEFAULT 0,

                    wins INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,

                    goals_for INTEGER NOT NULL DEFAULT 0,
                    goals_against INTEGER NOT NULL DEFAULT 0,

                    recent_form REAL NOT NULL DEFAULT 0.5,
                    reliability REAL NOT NULL DEFAULT 0.0,
                    uncertainty REAL NOT NULL DEFAULT 1.0,

                    last_match_at TEXT,
                    last_updated TEXT NOT NULL,

                    PRIMARY KEY (team, league)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_team_elo_v14_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,

                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,

                    home_old_rating REAL NOT NULL,
                    away_old_rating REAL NOT NULL,
                    home_new_rating REAL NOT NULL,
                    away_new_rating REAL NOT NULL,

                    home_change REAL NOT NULL,
                    away_change REAL NOT NULL,

                    k_factor REAL NOT NULL,
                    expected_home REAL NOT NULL,
                    actual_home REAL NOT NULL,

                    played_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    source_hash TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_team_elo_v14_league
                ON football_team_elo_v14 (
                    league,
                    rating DESC
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_team_elo_v14_history
                ON football_team_elo_v14_history (
                    league,
                    played_at
                )
                """
            )

            conn.commit()

    def load(self, team: str, league: str) -> TeamEloV14:
        self.init_db()

        team = normalize_team(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_team_elo_v14
                WHERE team=? AND league=?
                """,
                (team, league),
            ).fetchone()

        if row is None:
            return TeamEloV14(
                team=team,
                league=league,
                last_updated=now_utc(),
            ).normalized()

        return TeamEloV14(**dict(row)).normalized()

    def save(self, rating: TeamEloV14) -> None:
        self.init_db()
        rating = rating.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_team_elo_v14 (
                    team,
                    league,
                    rating,
                    home_rating,
                    away_rating,
                    matches,
                    home_matches,
                    away_matches,
                    wins,
                    draws,
                    losses,
                    goals_for,
                    goals_against,
                    recent_form,
                    reliability,
                    uncertainty,
                    last_match_at,
                    last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, league) DO UPDATE SET
                    rating=excluded.rating,
                    home_rating=excluded.home_rating,
                    away_rating=excluded.away_rating,
                    matches=excluded.matches,
                    home_matches=excluded.home_matches,
                    away_matches=excluded.away_matches,
                    wins=excluded.wins,
                    draws=excluded.draws,
                    losses=excluded.losses,
                    goals_for=excluded.goals_for,
                    goals_against=excluded.goals_against,
                    recent_form=excluded.recent_form,
                    reliability=excluded.reliability,
                    uncertainty=excluded.uncertainty,
                    last_match_at=excluded.last_match_at,
                    last_updated=excluded.last_updated
                """,
                (
                    rating.team,
                    rating.league,
                    rating.rating,
                    rating.home_rating,
                    rating.away_rating,
                    rating.matches,
                    rating.home_matches,
                    rating.away_matches,
                    rating.wins,
                    rating.draws,
                    rating.losses,
                    rating.goals_for,
                    rating.goals_against,
                    rating.recent_form,
                    rating.reliability,
                    rating.uncertainty,
                    rating.last_match_at,
                    rating.last_updated,
                ),
            )
            conn.commit()

    @staticmethod
    def expected_score(
        rating_a: float,
        rating_b: float,
    ) -> float:
        return 1.0 / (
            1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
        )

    @staticmethod
    def actual_score(
        home_goals: int,
        away_goals: int,
    ) -> float:
        if home_goals > away_goals:
            return 1.0

        if home_goals < away_goals:
            return 0.0

        return 0.5

    @staticmethod
    def goal_margin_multiplier(
        home_goals: int,
        away_goals: int,
        rating_difference: float,
    ) -> float:
        margin = abs(home_goals - away_goals)

        if margin <= 1:
            return 1.0

        multiplier = math.log(margin + 1.0) * (
            2.2 / (0.001 * abs(rating_difference) + 2.2)
        )

        return clamp(multiplier, 1.0, 2.5)

    @staticmethod
    def dynamic_k_factor(
        *,
        home_matches: int,
        away_matches: int,
        importance: float,
        reliability: float,
    ) -> float:
        experience = min(home_matches, away_matches)

        if experience < 10:
            base = 34.0
        elif experience < 30:
            base = 28.0
        else:
            base = DEFAULT_K_FACTOR

        importance_multiplier = clamp(
            importance,
            0.70,
            1.60,
        )

        reliability_multiplier = (
            1.15 - reliability * 0.25
        )

        return clamp(
            base
            * importance_multiplier
            * reliability_multiplier,
            12.0,
            55.0,
        )

    @staticmethod
    def _recent_form_update(
        current: float,
        actual_score: float,
        decay: float = 0.82,
    ) -> float:
        return clamp(
            current * decay + actual_score * (1.0 - decay),
            0.0,
            1.0,
        )

    def update_after_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        played_at: str,
        source: str = "",
        source_hash: str | None = None,
        home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE,
        importance: float = 1.0,
    ) -> EloV14Update:
        self.init_db()

        home = self.load(home_team, league)
        away = self.load(away_team, league)

        if source_hash:
            with self.connect() as conn:
                duplicate = conn.execute(
                    """
                    SELECT 1
                    FROM football_team_elo_v14_history
                    WHERE source_hash=?
                    """,
                    (source_hash,),
                ).fetchone()

            if duplicate is not None:
                return EloV14Update(
                    inserted=False,
                    home_old_rating=home.rating,
                    away_old_rating=away.rating,
                    home_new_rating=home.rating,
                    away_new_rating=away.rating,
                    home_change=0.0,
                    away_change=0.0,
                    k_factor=0.0,
                    expected_home=0.5,
                    actual_home=0.5,
                )

        effective_home_rating = (
            home.home_rating + home_advantage_elo
        )
        effective_away_rating = away.away_rating

        expected_home = self.expected_score(
            effective_home_rating,
            effective_away_rating,
        )
        actual_home = self.actual_score(
            home_goals,
            away_goals,
        )

        combined_reliability = (
            home.reliability + away.reliability
        ) / 2.0

        k_factor = self.dynamic_k_factor(
            home_matches=home.matches,
            away_matches=away.matches,
            importance=importance,
            reliability=combined_reliability,
        )

        margin_multiplier = self.goal_margin_multiplier(
            home_goals,
            away_goals,
            effective_home_rating - effective_away_rating,
        )

        change = (
            k_factor
            * margin_multiplier
            * (actual_home - expected_home)
        )

        home_old_rating = home.rating
        away_old_rating = away.rating

        home.rating += change
        away.rating -= change

        home.home_rating += change * 0.70
        home.away_rating += change * 0.30

        away.away_rating -= change * 0.70
        away.home_rating -= change * 0.30

        home.matches += 1
        away.matches += 1
        home.home_matches += 1
        away.away_matches += 1

        home.goals_for += int(home_goals)
        home.goals_against += int(away_goals)
        away.goals_for += int(away_goals)
        away.goals_against += int(home_goals)

        away_actual = 1.0 - actual_home

        if actual_home == 1.0:
            home.wins += 1
            away.losses += 1
        elif actual_home == 0.0:
            home.losses += 1
            away.wins += 1
        else:
            home.draws += 1
            away.draws += 1

        home.recent_form = self._recent_form_update(
            home.recent_form,
            actual_home,
        )
        away.recent_form = self._recent_form_update(
            away.recent_form,
            away_actual,
        )

        home.reliability = clamp(
            home.matches / 30.0,
            0.0,
            1.0,
        )
        away.reliability = clamp(
            away.matches / 30.0,
            0.0,
            1.0,
        )

        home.uncertainty = clamp(
            1.0 - home.reliability * 0.85,
            0.05,
            1.0,
        )
        away.uncertainty = clamp(
            1.0 - away.reliability * 0.85,
            0.05,
            1.0,
        )

        home.last_match_at = played_at
        away.last_match_at = played_at
        home.last_updated = now_utc()
        away.last_updated = now_utc()

        self.save(home)
        self.save(away)

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_team_elo_v14_history (
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    home_old_rating,
                    away_old_rating,
                    home_new_rating,
                    away_new_rating,
                    home_change,
                    away_change,
                    k_factor,
                    expected_home,
                    actual_home,
                    played_at,
                    source,
                    source_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    league,
                    home.team,
                    away.team,
                    int(home_goals),
                    int(away_goals),
                    home_old_rating,
                    away_old_rating,
                    home.rating,
                    away.rating,
                    home.rating - home_old_rating,
                    away.rating - away_old_rating,
                    k_factor,
                    expected_home,
                    actual_home,
                    played_at,
                    source,
                    source_hash,
                    now_utc(),
                ),
            )
            conn.commit()

        return EloV14Update(
            inserted=True,
            home_old_rating=home_old_rating,
            away_old_rating=away_old_rating,
            home_new_rating=home.rating,
            away_new_rating=away.rating,
            home_change=home.rating - home_old_rating,
            away_change=away.rating - away_old_rating,
            k_factor=k_factor,
            expected_home=expected_home,
            actual_home=actual_home,
        )

    def predict_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE,
    ) -> EloV14Prediction:
        home = self.load(home_team, league)
        away = self.load(away_team, league)

        effective_home = home.home_rating + home_advantage_elo
        effective_away = away.away_rating

        base_home = self.expected_score(
            effective_home,
            effective_away,
        )

        draw_base = 0.26

        reliability = (
            home.reliability + away.reliability
        ) / 2.0

        form_gap = home.recent_form - away.recent_form
        form_adjustment = clamp(
            form_gap * 0.08 * reliability,
            -0.05,
            0.05,
        )

        home_probability = clamp(
            base_home + form_adjustment,
            0.05,
            0.90,
        )

        draw_probability = clamp(
            draw_base
            + (1.0 - abs(home_probability - 0.50) * 2.0)
            * 0.05
            - reliability * 0.02,
            0.12,
            0.34,
        )

        away_probability = max(
            0.01,
            1.0 - home_probability - draw_probability,
        )

        total = (
            home_probability
            + draw_probability
            + away_probability
        )

        home_probability /= total
        draw_probability /= total
        away_probability /= total

        uncertainty_penalty = (
            home.uncertainty + away.uncertainty
        ) / 2.0

        reason = (
            f"v14 team ELO: home={effective_home:.1f}; "
            f"away={effective_away:.1f}; "
            f"diff={effective_home - effective_away:.1f}; "
            f"form_gap={form_gap:.3f}; "
            f"reliability={reliability:.3f}; "
            f"uncertainty={uncertainty_penalty:.3f}"
        )

        return EloV14Prediction(
            league=league,
            home_team=home.team,
            away_team=away.team,
            home_probability=home_probability,
            draw_probability=draw_probability,
            away_probability=away_probability,
            home_rating=effective_home,
            away_rating=effective_away,
            rating_difference=effective_home - effective_away,
            home_reliability=home.reliability,
            away_reliability=away.reliability,
            combined_reliability=reliability,
            uncertainty_penalty=uncertainty_penalty,
            reason=reason,
        )

    def rebuild_from_history(self) -> int:
        self.init_db()

        with self.connect() as conn:
            table_exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type='table'
                  AND name='football_form_history'
                """
            ).fetchone()

            if table_exists is None:
                return 0

            rows = conn.execute(
                """
                SELECT
                    league,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    played_at,
                    source,
                    source_hash
                FROM football_form_history
                ORDER BY played_at, id
                """
            ).fetchall()

        processed = 0

        for row in rows:
            source_hash = (
                f"{row['source_hash']}:elo_v14"
                if row["source_hash"]
                else None
            )

            result = self.update_after_match(
                league=row["league"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                played_at=row["played_at"],
                source=row["source"] or "football_form_history",
                source_hash=source_hash,
            )

            if result.inserted:
                processed += 1

        return processed

    def export_json(
        self,
        path: str = "exports/football_team_elo_v14.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_team_elo_v14
                    ORDER BY league, rating DESC
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


def predict_team_elo_v14(
    settings: Settings,
    *,
    league: str,
    home_team: str,
    away_team: str,
    home_advantage_elo: float = DEFAULT_HOME_ADVANTAGE,
) -> EloV14Prediction:
    database = FootballTeamEloV14Database(settings)

    return database.predict_match(
        league=league,
        home_team=home_team,
        away_team=away_team,
        home_advantage_elo=home_advantage_elo,
    )


if __name__ == "__main__":
    settings = Settings.from_env()
    database = FootballTeamEloV14Database(settings)

    rebuilt = database.rebuild_from_history()
    exported = database.export_json()

    print(
        "Football Team ELO v14 finished: "
        f"rebuilt={rebuilt}, exported={exported}"
    )
