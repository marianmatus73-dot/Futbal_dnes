from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_LEAGUE_XG = 1.35
PRIOR_MATCHES = 8.0
RECENCY_DECAY = 0.88


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
class TeamXGV14:
    team: str
    league: str

    matches: int = 0
    effective_matches: float = 0.0

    attack_xg: float = 1.0
    defense_xg: float = 1.0

    home_attack_xg: float = 1.0
    home_defense_xg: float = 1.0
    away_attack_xg: float = 1.0
    away_defense_xg: float = 1.0

    recent_xg_for: float = DEFAULT_LEAGUE_XG
    recent_xg_against: float = DEFAULT_LEAGUE_XG

    xg_for_variance: float = 0.0
    xg_against_variance: float = 0.0

    attack_uncertainty: float = 1.0
    defense_uncertainty: float = 1.0
    reliability: float = 0.0

    last_updated: str = ""

    def normalized(self) -> "TeamXGV14":
        self.team = normalize_team(self.team)
        self.league = str(self.league or "UNKNOWN").strip() or "UNKNOWN"

        self.matches = max(0, safe_int(self.matches))
        self.effective_matches = clamp(
            safe_float(self.effective_matches, 0.0),
            0.0,
            500.0,
        )

        for name in (
            "attack_xg",
            "defense_xg",
            "home_attack_xg",
            "home_defense_xg",
            "away_attack_xg",
            "away_defense_xg",
        ):
            setattr(
                self,
                name,
                clamp(safe_float(getattr(self, name), 1.0), 0.25, 3.50),
            )

        self.recent_xg_for = clamp(
            safe_float(self.recent_xg_for, DEFAULT_LEAGUE_XG),
            0.05,
            6.0,
        )
        self.recent_xg_against = clamp(
            safe_float(self.recent_xg_against, DEFAULT_LEAGUE_XG),
            0.05,
            6.0,
        )

        self.xg_for_variance = clamp(
            safe_float(self.xg_for_variance, 0.0),
            0.0,
            10.0,
        )
        self.xg_against_variance = clamp(
            safe_float(self.xg_against_variance, 0.0),
            0.0,
            10.0,
        )

        self.attack_uncertainty = clamp(
            safe_float(self.attack_uncertainty, 1.0),
            0.0,
            1.0,
        )
        self.defense_uncertainty = clamp(
            safe_float(self.defense_uncertainty, 1.0),
            0.0,
            1.0,
        )
        self.reliability = clamp(
            safe_float(self.reliability, 0.0),
            0.0,
            1.0,
        )

        if not self.last_updated:
            self.last_updated = now_utc()

        return self


@dataclass
class MatchXGV14Prediction:
    league: str
    home_team: str
    away_team: str

    home_xg: float
    away_xg: float

    home_attack_strength: float
    home_defense_strength: float
    away_attack_strength: float
    away_defense_strength: float

    home_reliability: float
    away_reliability: float
    combined_reliability: float

    uncertainty_penalty: float
    reason: str


class FootballTeamXGV14Database:
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
                CREATE TABLE IF NOT EXISTS football_team_xg_v14 (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,

                    matches INTEGER NOT NULL DEFAULT 0,
                    effective_matches REAL NOT NULL DEFAULT 0.0,

                    attack_xg REAL NOT NULL DEFAULT 1.0,
                    defense_xg REAL NOT NULL DEFAULT 1.0,

                    home_attack_xg REAL NOT NULL DEFAULT 1.0,
                    home_defense_xg REAL NOT NULL DEFAULT 1.0,
                    away_attack_xg REAL NOT NULL DEFAULT 1.0,
                    away_defense_xg REAL NOT NULL DEFAULT 1.0,

                    recent_xg_for REAL NOT NULL DEFAULT 1.35,
                    recent_xg_against REAL NOT NULL DEFAULT 1.35,

                    xg_for_variance REAL NOT NULL DEFAULT 0.0,
                    xg_against_variance REAL NOT NULL DEFAULT 0.0,

                    attack_uncertainty REAL NOT NULL DEFAULT 1.0,
                    defense_uncertainty REAL NOT NULL DEFAULT 1.0,
                    reliability REAL NOT NULL DEFAULT 0.0,

                    last_updated TEXT NOT NULL,

                    PRIMARY KEY (team, league)
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_team_xg_v14_league
                ON football_team_xg_v14 (league, reliability DESC)
                """
            )

            conn.commit()

    def load(self, team: str, league: str) -> TeamXGV14:
        self.init_db()

        team = normalize_team(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM football_team_xg_v14
                WHERE team=? AND league=?
                """,
                (team, league),
            ).fetchone()

        if row is None:
            return TeamXGV14(
                team=team,
                league=league,
                last_updated=now_utc(),
            ).normalized()

        return TeamXGV14(**dict(row)).normalized()

    def save(self, rating: TeamXGV14) -> None:
        self.init_db()
        rating = rating.normalized()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_team_xg_v14 (
                    team,
                    league,
                    matches,
                    effective_matches,
                    attack_xg,
                    defense_xg,
                    home_attack_xg,
                    home_defense_xg,
                    away_attack_xg,
                    away_defense_xg,
                    recent_xg_for,
                    recent_xg_against,
                    xg_for_variance,
                    xg_against_variance,
                    attack_uncertainty,
                    defense_uncertainty,
                    reliability,
                    last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, league) DO UPDATE SET
                    matches=excluded.matches,
                    effective_matches=excluded.effective_matches,
                    attack_xg=excluded.attack_xg,
                    defense_xg=excluded.defense_xg,
                    home_attack_xg=excluded.home_attack_xg,
                    home_defense_xg=excluded.home_defense_xg,
                    away_attack_xg=excluded.away_attack_xg,
                    away_defense_xg=excluded.away_defense_xg,
                    recent_xg_for=excluded.recent_xg_for,
                    recent_xg_against=excluded.recent_xg_against,
                    xg_for_variance=excluded.xg_for_variance,
                    xg_against_variance=excluded.xg_against_variance,
                    attack_uncertainty=excluded.attack_uncertainty,
                    defense_uncertainty=excluded.defense_uncertainty,
                    reliability=excluded.reliability,
                    last_updated=excluded.last_updated
                """,
                (
                    rating.team,
                    rating.league,
                    rating.matches,
                    rating.effective_matches,
                    rating.attack_xg,
                    rating.defense_xg,
                    rating.home_attack_xg,
                    rating.home_defense_xg,
                    rating.away_attack_xg,
                    rating.away_defense_xg,
                    rating.recent_xg_for,
                    rating.recent_xg_against,
                    rating.xg_for_variance,
                    rating.xg_against_variance,
                    rating.attack_uncertainty,
                    rating.defense_uncertainty,
                    rating.reliability,
                    rating.last_updated,
                ),
            )
            conn.commit()

    def _league_average(self, league: str) -> tuple[float, float]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    AVG(home_xg) AS avg_home_xg,
                    AVG(away_xg) AS avg_away_xg
                FROM football_xg_history
                WHERE league=?
                """,
                (league,),
            ).fetchone()

        if row is None:
            return DEFAULT_LEAGUE_XG, DEFAULT_LEAGUE_XG

        return (
            clamp(
                safe_float(row["avg_home_xg"], DEFAULT_LEAGUE_XG),
                0.50,
                2.50,
            ),
            clamp(
                safe_float(row["avg_away_xg"], DEFAULT_LEAGUE_XG),
                0.50,
                2.50,
            ),
        )

    def rebuild_team(self, team: str, league: str) -> TeamXGV14:
        self.init_db()

        team = normalize_team(team)
        league = str(league or "UNKNOWN").strip() or "UNKNOWN"

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM football_xg_history
                WHERE league=?
                  AND (home_team=? OR away_team=?)
                ORDER BY played_at DESC, id DESC
                LIMIT 50
                """,
                (league, team, team),
            ).fetchall()

        league_home_xg, league_away_xg = self._league_average(league)

        if not rows:
            rating = TeamXGV14(
                team=team,
                league=league,
                last_updated=now_utc(),
            )
            self.save(rating)
            return rating

        weighted_for = 0.0
        weighted_against = 0.0
        weighted_for_sq = 0.0
        weighted_against_sq = 0.0
        total_weight = 0.0

        home_for = home_against = home_weight = 0.0
        away_for = away_against = away_weight = 0.0

        for index, row in enumerate(rows):
            weight = RECENCY_DECAY ** index
            is_home = row["home_team"] == team

            xg_for = safe_float(
                row["home_xg"] if is_home else row["away_xg"],
                DEFAULT_LEAGUE_XG,
            )
            xg_against = safe_float(
                row["away_xg"] if is_home else row["home_xg"],
                DEFAULT_LEAGUE_XG,
            )

            weighted_for += xg_for * weight
            weighted_against += xg_against * weight
            weighted_for_sq += (xg_for ** 2) * weight
            weighted_against_sq += (xg_against ** 2) * weight
            total_weight += weight

            if is_home:
                home_for += xg_for * weight
                home_against += xg_against * weight
                home_weight += weight
            else:
                away_for += xg_for * weight
                away_against += xg_against * weight
                away_weight += weight

        recent_for = weighted_for / total_weight
        recent_against = weighted_against / total_weight

        variance_for = max(
            0.0,
            weighted_for_sq / total_weight - recent_for ** 2,
        )
        variance_against = max(
            0.0,
            weighted_against_sq / total_weight - recent_against ** 2,
        )

        effective_matches = total_weight
        prior_weight = PRIOR_MATCHES

        league_attack_baseline = (
            league_home_xg + league_away_xg
        ) / 2.0

        posterior_for = (
            recent_for * effective_matches
            + league_attack_baseline * prior_weight
        ) / (effective_matches + prior_weight)

        posterior_against = (
            recent_against * effective_matches
            + league_attack_baseline * prior_weight
        ) / (effective_matches + prior_weight)

        attack_strength = posterior_for / league_attack_baseline
        defense_strength = posterior_against / league_attack_baseline

        home_attack = (
            (home_for / home_weight) / league_home_xg
            if home_weight > 0
            else attack_strength
        )
        home_defense = (
            (home_against / home_weight) / league_away_xg
            if home_weight > 0
            else defense_strength
        )
        away_attack = (
            (away_for / away_weight) / league_away_xg
            if away_weight > 0
            else attack_strength
        )
        away_defense = (
            (away_against / away_weight) / league_home_xg
            if away_weight > 0
            else defense_strength
        )

        reliability = clamp(
            effective_matches / 20.0,
            0.0,
            1.0,
        )

        attack_uncertainty = clamp(
            (math.sqrt(variance_for) / max(recent_for, 0.20))
            * (1.0 - reliability * 0.65),
            0.0,
            1.0,
        )
        defense_uncertainty = clamp(
            (math.sqrt(variance_against) / max(recent_against, 0.20))
            * (1.0 - reliability * 0.65),
            0.0,
            1.0,
        )

        rating = TeamXGV14(
            team=team,
            league=league,
            matches=len(rows),
            effective_matches=effective_matches,
            attack_xg=attack_strength,
            defense_xg=defense_strength,
            home_attack_xg=home_attack,
            home_defense_xg=home_defense,
            away_attack_xg=away_attack,
            away_defense_xg=away_defense,
            recent_xg_for=recent_for,
            recent_xg_against=recent_against,
            xg_for_variance=variance_for,
            xg_against_variance=variance_against,
            attack_uncertainty=attack_uncertainty,
            defense_uncertainty=defense_uncertainty,
            reliability=reliability,
            last_updated=now_utc(),
        ).normalized()

        self.save(rating)
        return rating

    def rebuild_all(self) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT league, home_team AS team
                FROM football_xg_history
                UNION
                SELECT league, away_team AS team
                FROM football_xg_history
                """
            ).fetchall()

        teams = {
            (normalize_team(row["team"]), str(row["league"]))
            for row in rows
            if normalize_team(row["team"])
        }

        for team, league in sorted(teams):
            self.rebuild_team(team, league)

        return len(teams)

    def predict_match(
        self,
        *,
        league: str,
        home_team: str,
        away_team: str,
        league_average_xg: float = DEFAULT_LEAGUE_XG,
        home_advantage: float = 1.08,
    ) -> MatchXGV14Prediction:
        home = self.load(home_team, league)
        away = self.load(away_team, league)

        home_attack = home.home_attack_xg
        away_defense = away.away_defense_xg
        away_attack = away.away_attack_xg
        home_defense = home.home_defense_xg

        home_xg = (
            league_average_xg
            * home_advantage
            * math.sqrt(home_attack * away_defense)
        )
        away_xg = (
            league_average_xg
            * math.sqrt(away_attack * home_defense)
        )

        combined_reliability = (
            home.reliability + away.reliability
        ) / 2.0

        uncertainty_penalty = clamp(
            (
                home.attack_uncertainty
                + home.defense_uncertainty
                + away.attack_uncertainty
                + away.defense_uncertainty
            )
            / 4.0,
            0.0,
            1.0,
        )

        shrink = clamp(
            (1.0 - combined_reliability) * 0.55
            + uncertainty_penalty * 0.20,
            0.0,
            0.75,
        )

        home_xg = (
            home_xg * (1.0 - shrink)
            + league_average_xg * home_advantage * shrink
        )
        away_xg = (
            away_xg * (1.0 - shrink)
            + league_average_xg * shrink
        )

        home_xg = clamp(home_xg, 0.15, 4.50)
        away_xg = clamp(away_xg, 0.15, 4.50)

        reason = (
            f"v14 team xG: home={home_xg:.3f}, away={away_xg:.3f}; "
            f"home_attack={home_attack:.3f}; "
            f"away_defense={away_defense:.3f}; "
            f"away_attack={away_attack:.3f}; "
            f"home_defense={home_defense:.3f}; "
            f"reliability={combined_reliability:.3f}; "
            f"uncertainty={uncertainty_penalty:.3f}; "
            f"shrink={shrink:.3f}"
        )

        return MatchXGV14Prediction(
            league=league,
            home_team=home.team,
            away_team=away.team,
            home_xg=home_xg,
            away_xg=away_xg,
            home_attack_strength=home_attack,
            home_defense_strength=home_defense,
            away_attack_strength=away_attack,
            away_defense_strength=away_defense,
            home_reliability=home.reliability,
            away_reliability=away.reliability,
            combined_reliability=combined_reliability,
            uncertainty_penalty=uncertainty_penalty,
            reason=reason,
        )

    def export_json(
        self,
        path: str = "exports/football_team_xg_v14.json",
    ) -> int:
        self.init_db()

        with self.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM football_team_xg_v14
                    ORDER BY league, reliability DESC, team
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


def predict_team_xg_v14(
    settings: Settings,
    *,
    league: str,
    home_team: str,
    away_team: str,
    league_average_xg: float = DEFAULT_LEAGUE_XG,
    home_advantage: float = 1.08,
) -> MatchXGV14Prediction:
    database = FootballTeamXGV14Database(settings)

    return database.predict_match(
        league=league,
        home_team=home_team,
        away_team=away_team,
        league_average_xg=league_average_xg,
        home_advantage=home_advantage,
    )


if __name__ == "__main__":
    settings = Settings.from_env()
    database = FootballTeamXGV14Database(settings)

    rebuilt = database.rebuild_all()
    exported = database.export_json()

    print(
        "Football Team xG v14 finished: "
        f"rebuilt={rebuilt}, exported={exported}"
    )
