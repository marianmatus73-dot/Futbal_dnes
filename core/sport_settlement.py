from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import aiohttp

from core.config import Settings
from core.sport_quant import (
    connect,
    norm,
    update_closing_lines,
    refresh_bookmaker_stats,
    update_elo_after_result,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_result(event: dict[str, Any]) -> tuple[str, int, int] | None:
    scores = event.get("scores") or []

    if len(scores) < 2:
        return None

    parsed = []

    for item in scores:
        name = str(item.get("name", ""))
        score_raw = item.get("score")

        try:
            score = int(score_raw)
        except Exception:
            try:
                score = int(float(score_raw))
            except Exception:
                return None

        parsed.append((name, score))

    home_name = str(event.get("home_team", ""))
    away_name = str(event.get("away_team", ""))
    score_map = {norm(name): score for name, score in parsed}
    home_score = score_map.get(norm(home_name))
    away_score = score_map.get(norm(away_name))
    if home_score is None or away_score is None:
        home_score, away_score = parsed[0][1], parsed[1][1]
        home_name, away_name = parsed[0][0], parsed[1][0]
    winner = home_name if home_score > away_score else away_name if away_score > home_score else "DRAW"
    return winner, home_score, away_score


def _winner_from_scores(event: dict[str, Any]) -> str | None:
    result = _score_result(event)
    return result[0] if result else None


def _profit_for_result(result: str, odds: float, stake: float) -> float:
    if result == "WON":
        return round((odds - 1.0) * stake, 4)

    if result == "LOST":
        return round(-stake, 4)

    return 0.0


async def fetch_scores(
    api_key: str,
    sport_key: str,
    days_from: int = 3,
) -> list[dict[str, Any]]:
    if not api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": max(1, min(int(days_from), 3)),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(
                        f"WARNING | Scores API {resp.status} "
                        f"for {sport_key}: {body[:200]}"
                    )
                    return []

                data = await resp.json()
                return data if isinstance(data, list) else []

    except Exception as exc:
        print(f"WARNING | Scores API error for {sport_key}: {exc}")
        return []


def ensure_settlement_columns(settings: Settings) -> None:
    with connect(settings) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(sport_bets)").fetchall()
        }

        if "settled_at" not in columns:
            conn.execute("ALTER TABLE sport_bets ADD COLUMN settled_at TEXT")

        if "profit" not in columns:
            conn.execute("ALTER TABLE sport_bets ADD COLUMN profit REAL DEFAULT 0")

        if "profit_units" not in columns:
            conn.execute(
                "ALTER TABLE sport_bets ADD COLUMN profit_units REAL DEFAULT 0"
            )

        for name, definition in {
            "home_goals": "INTEGER",
            "away_goals": "INTEGER",
            "final_score": "TEXT",
            "settlement_source": "TEXT",
        }.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE sport_bets ADD COLUMN {name} {definition}")


def backfill_settled_profit(settings: Settings) -> int:
    """Repair legacy settled rows whose accounting fields were never written."""
    ensure_settlement_columns(settings)

    with connect(settings) as conn:
        before = conn.total_changes
        conn.execute(
            """
            UPDATE sport_bets
            SET profit = CASE UPPER(TRIM(result))
                    WHEN 'WON' THEN ROUND((CAST(odds AS REAL) - 1.0)
                                          * CAST(stake AS REAL), 4)
                    WHEN 'LOST' THEN ROUND(-CAST(stake AS REAL), 4)
                    ELSE 0.0
                END,
                profit_units = CASE UPPER(TRIM(result))
                    WHEN 'WON' THEN ROUND(CAST(odds AS REAL) - 1.0, 4)
                    WHEN 'LOST' THEN -1.0
                    ELSE 0.0
                END
            WHERE UPPER(TRIM(COALESCE(result, ''))) IN
                  ('WON', 'LOST', 'VOID')
              AND CAST(COALESCE(stake, 0) AS REAL) > 0
              AND CAST(COALESCE(odds, 0) AS REAL) > 1.0
              AND (
                    profit IS NULL
                    OR profit_units IS NULL
                    OR (CAST(profit AS REAL) = 0.0
                        AND UPPER(TRIM(result)) IN ('WON', 'LOST'))
                  )
            """
        )
        return conn.total_changes - before


async def settle_sport_bets(
    settings: Settings,
    sport: str,
    sport_keys: list[str],
) -> int:
    api_key = settings.odds_api_key or os.getenv("ODDS_API_KEY", "")

    if not api_key:
        return 0

    ensure_settlement_columns(settings)
    update_closing_lines(settings, sport)

    with connect(settings) as conn:
        open_rows = conn.execute(
            """
            SELECT id, league, event, home_team, away_team,
                   selection, odds, stake, market
            FROM sport_bets
            WHERE sport=?
              AND market IN ('h2h', 'totals_2.5')
              AND (
                    result IS NULL
                    OR result=''
                    OR result='OPEN'
                    OR result='V'
                    OR result='P'
                  )
            """,
            (sport,),
        ).fetchall()

    if not open_rows:
        refresh_bookmaker_stats(settings, sport)
        return 0

    rows_by_league: dict[str, list[tuple[Any, ...]]] = {}

    for row in open_rows:
        rows_by_league.setdefault(str(row[1]), []).append(row)

    settled = 0

    for sport_key in [s.strip() for s in sport_keys if s.strip()]:
        scores = await fetch_scores(api_key, sport_key, days_from=3)

        if not scores:
            continue

        score_events = []

        for event in scores:
            if not event.get("completed"):
                continue

            home = str(event.get("home_team", ""))
            away = str(event.get("away_team", ""))
            score_result = _score_result(event)

            if not score_result:
                continue

            winner, home_score, away_score = score_result

            score_events.append(
                {
                    "home": home,
                    "away": away,
                    "event": f"{home} vs {away}",
                    "winner": winner,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

        if not score_events:
            continue

        updates: list[tuple[Any, ...]] = []
        elo_updates: list[tuple[str, str, str]] = []

        for row in rows_by_league.get(sport_key, []):
            (
                bet_id,
                league,
                event_name,
                home_team,
                away_team,
                selection,
                odds,
                stake,
                market,
            ) = row

            bet_home = norm(home_team or "")
            bet_away = norm(away_team or "")
            bet_event = norm(event_name or "")
            bet_selection = norm(selection or "")

            matched = None

            for score_event in score_events:
                score_home = norm(score_event["home"])
                score_away = norm(score_event["away"])
                score_event_name = norm(score_event["event"])

                if (
                    bet_home
                    and bet_away
                    and bet_home == score_home
                    and bet_away == score_away
                ):
                    matched = score_event
                    break

                if bet_event and bet_event == score_event_name:
                    matched = score_event
                    break

            if not matched:
                continue

            winner = str(matched["winner"])

            if str(market) == "totals_2.5":
                total = int(matched["home_score"]) + int(matched["away_score"])
                is_over = "over" in bet_selection
                is_under = "under" in bet_selection
                result = "WON" if (is_over and total > 2.5) or (is_under and total < 2.5) else "LOST"
            elif winner == "DRAW":
                result = "WON" if bet_selection in {"draw", "x", "remiza"} else "LOST"
            else:
                result = "WON" if norm(winner) == bet_selection else "LOST"

            odds_float = float(odds or 0)
            stake_float = float(stake or 0)

            profit = _profit_for_result(
                result=result,
                odds=odds_float,
                stake=stake_float,
            )

            profit_units = round(
                profit / stake_float,
                4,
            ) if stake_float > 0 else 0.0

            updates.append(
                (
                    result,
                    now_utc(),
                    profit,
                    profit_units,
                    int(matched["home_score"]),
                    int(matched["away_score"]),
                    f'{matched["home_score"]}-{matched["away_score"]}',
                    "the_odds_api_scores",
                    int(bet_id),
                )
            )

            if str(market) == "h2h" and result in {"WON", "LOST"}:
                elo_updates.append(
                    (
                        str(matched["home"]),
                        str(matched["away"]),
                        winner,
                    )
                )

        if updates:
            with connect(settings) as conn:
                conn.executemany(
                    """
                    UPDATE sport_bets
                    SET result=?,
                        settled_at=?,
                        profit=?,
                        profit_units=?
                        ,home_goals=?
                        ,away_goals=?
                        ,final_score=?
                        ,settlement_source=?
                    WHERE id=?
                    """,
                    updates,
                )

            settled += len(updates)

            for home, away, winner in elo_updates:
                k = 20.0 if sport == "tennis" else 24.0
                home_adv = (
                    0.0
                    if sport == "tennis"
                    else float(os.getenv("HOCKEY_HOME_ELO_ADV", "35"))
                )

                update_elo_after_result(
                    settings,
                    sport,
                    home,
                    away,
                    winner,
                    k=k,
                    home_adv=home_adv,
                )

    update_closing_lines(settings, sport)
    refresh_bookmaker_stats(settings, sport)

    return settled
