from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings
from core.sport_context import SportContextDatabase
from core.types import Bet, SportResult


@dataclass(frozen=True)
class ReleaseSummary:
    early: int = 0
    final: int = 0
    awaiting_lineup: int = 0


def _parse_time(value: str) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ensure_release_columns(settings: Settings) -> None:
    database = Path(settings.db_file or "bets.db")
    if not database.exists():
        return
    with sqlite3.connect(database) as conn:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")
        }
        for name, definition in {
            "release_stage": "TEXT NOT NULL DEFAULT ''",
            "opening_odds": "REAL",
            "final_odds": "REAL",
            "early_released_at": "TEXT",
            "final_confirmed_at": "TEXT",
            "lineup_verified": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE sport_bets ADD COLUMN {name} {definition}")


def classify_football_release(
    bet: Bet,
    context_database: SportContextDatabase,
    *,
    now: datetime | None = None,
    final_window_minutes: int = 60,
) -> str:
    current = now or datetime.now(timezone.utc)
    start = _parse_time(bet.start_time)
    context = context_database.latest(
        "football", bet.event, bet.external_event_id, bet.start_time
    )
    bet.lineup_verified = bool(context.verified and context.lineup_confirmed)
    if start is None:
        return "EARLY"
    minutes = (start - current).total_seconds() / 60.0
    if minutes > final_window_minutes:
        return "EARLY"
    if minutes >= 0 and bet.lineup_verified:
        return "FINAL"
    return "AWAITING_LINEUP"


def apply_football_release_policy(
    module_outputs: list[dict],
    settings: Settings,
    *,
    now: datetime | None = None,
) -> ReleaseSummary:
    ensure_release_columns(settings)
    context_database = SportContextDatabase(settings)
    current = now or datetime.now(timezone.utc)
    early = final = awaiting = 0
    database = Path(settings.db_file or "bets.db")

    for output in module_outputs:
        result = output.get("result")
        if not isinstance(result, SportResult) or result.sport != "football":
            continue
        published: list[Bet] = []
        for bet in result.bets:
            stage = classify_football_release(
                bet, context_database, now=current,
            )
            bet.release_stage = stage
            if stage == "EARLY":
                early += 1
            elif stage == "FINAL":
                final += 1
            else:
                awaiting += 1

            with sqlite3.connect(database) as conn:
                previous = conn.execute(
                    """
                    SELECT opening_odds, final_odds
                    FROM sport_bets
                    WHERE sport='football' AND league=? AND event=?
                      AND selection=? AND start_time=?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (bet.league, bet.event, bet.selection, bet.start_time),
                ).fetchone()
                bet.opening_odds = (
                    float(previous[0])
                    if previous and previous[0] not in (None, "")
                    else float(bet.odds)
                )
                if stage == "FINAL":
                    bet.final_odds = float(bet.odds)
                elif previous and previous[1] not in (None, ""):
                    bet.final_odds = float(previous[1])
                conn.execute(
                    """
                    UPDATE sport_bets
                    SET release_stage=?,
                        opening_odds=COALESCE(opening_odds, odds),
                        final_odds=CASE WHEN ?='FINAL' THEN ? ELSE final_odds END,
                        early_released_at=CASE
                            WHEN ?='EARLY' THEN COALESCE(early_released_at, ?)
                            ELSE early_released_at END,
                        final_confirmed_at=CASE
                            WHEN ?='FINAL' THEN COALESCE(final_confirmed_at, ?)
                            ELSE final_confirmed_at END,
                        lineup_verified=?
                    WHERE sport='football' AND league=? AND event=?
                      AND selection=? AND start_time=?
                    """,
                    (
                        stage, stage, bet.odds, stage, current.isoformat(),
                        stage, current.isoformat(), int(bet.lineup_verified),
                        bet.league, bet.event, bet.selection, bet.start_time,
                    ),
                )
            if stage != "AWAITING_LINEUP":
                published.append(bet)
        result.bets = published

    return ReleaseSummary(early=early, final=final, awaiting_lineup=awaiting)
