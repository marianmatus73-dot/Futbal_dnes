from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.sport_quant import norm
from core.sport_settlement import fetch_scores


@dataclass(frozen=True)
class ObservationSettlementSummary:
    score_events: int = 0
    matched_events: int = 0
    settled_rows: int = 0
    won: int = 0
    lost: int = 0
    void: int = 0


def _db_path(settings: Settings) -> Path:
    return Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_score(event: dict[str, Any]) -> tuple[int, int] | None:
    """Return the score only when both named teams are explicitly present."""
    home = norm(str(event.get("home_team", "")))
    away = norm(str(event.get("away_team", "")))
    if not home or not away:
        return None

    values: dict[str, int] = {}
    for item in event.get("scores") or []:
        try:
            values[norm(str(item.get("name", "")))] = int(float(item.get("score")))
        except (TypeError, ValueError):
            return None
    if home not in values or away not in values:
        return None
    return values[home], values[away]


def _h2h_result(
    selection: str,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> str | None:
    choice = norm(selection)
    if choice in {"draw", "x", "remiza"}:
        return "WON" if home_score == away_score else "LOST"
    if choice == norm(home):
        return "WON" if home_score > away_score else "LOST"
    if choice == norm(away):
        return "WON" if away_score > home_score else "LOST"
    return None


async def settle_learning_observations(
    settings: Settings,
    *,
    sport: str,
    sport_keys: list[str],
) -> ObservationSettlementSummary:
    """Settle shadow observations using the provider's immutable event ID.

    Team-name similarity is intentionally not a fallback. An unmatched row is
    safer left OPEN than assigned a result from a different event.
    """
    api_key = settings.odds_api_key or os.getenv("ODDS_API_KEY", "")
    database = _db_path(settings)
    if not api_key or not database.exists():
        return ObservationSettlementSummary()

    with closing(sqlite3.connect(database)) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='sport_learning_observations'"
        ).fetchone()
        if not table:
            return ObservationSettlementSummary()
        open_ids = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT external_event_id
                FROM sport_learning_observations
                WHERE sport=? AND result='OPEN'
                  AND TRIM(COALESCE(external_event_id, '')) <> ''
                """,
                (sport,),
            )
        }
    if not open_ids:
        return ObservationSettlementSummary()

    completed: dict[str, dict[str, Any]] = {}
    for sport_key in dict.fromkeys(key.strip() for key in sport_keys if key.strip()):
        for event in await fetch_scores(api_key, sport_key, days_from=3):
            event_id = str(event.get("id", "")).strip()
            if event_id in open_ids and event.get("completed") is True:
                completed[event_id] = event

    settled_rows = won = lost = void = matched_events = 0
    settled_at = _now_utc()
    with closing(sqlite3.connect(database)) as conn:
        for event_id, event in completed.items():
            score = _strict_score(event)
            if score is None:
                continue
            home_score, away_score = score
            home = str(event.get("home_team", ""))
            away = str(event.get("away_team", ""))
            rows = conn.execute(
                """
                SELECT id, market, selection
                FROM sport_learning_observations
                WHERE sport=? AND external_event_id=? AND result='OPEN'
                """,
                (sport, event_id),
            ).fetchall()
            event_updates = 0
            for row_id, market, selection in rows:
                result = None
                if str(market) == "h2h":
                    result = _h2h_result(
                        str(selection), home, away, home_score, away_score
                    )
                if result is None:
                    continue
                conn.execute(
                    """
                    UPDATE sport_learning_observations
                    SET result=?, final_score=?, settled_at=?
                    WHERE id=? AND result='OPEN'
                    """,
                    (result, f"{home_score}-{away_score}", settled_at, row_id),
                )
                event_updates += 1
                settled_rows += 1
                won += result == "WON"
                lost += result == "LOST"
                void += result == "VOID"
            matched_events += event_updates > 0
        conn.commit()

    return ObservationSettlementSummary(
        score_events=len(completed),
        matched_events=matched_events,
        settled_rows=settled_rows,
        won=won,
        lost=lost,
        void=void,
    )

