from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_team_aliases import (
    FootballTeamAliasEngine,
)


log = logging.getLogger(__name__)

API_HOST = "https://api.the-odds-api.com"
DEFAULT_DAYS_FROM = 3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_team(value: Any) -> str:
    return normalize_text(value).casefold()


def parse_datetime(value: Any) -> datetime | None:
    text = normalize_text(value)

    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def split_event(event: str) -> tuple[str, str] | None:
    text = normalize_text(event)

    for separator in (" vs ", " v ", " - "):
        if separator in text:
            home, away = text.split(separator, 1)
            home = normalize_text(home)
            away = normalize_text(away)

            if home and away:
                return home, away

    return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class OpenFootballBet:
    bet_id: int
    source_hash: str
    sport_key: str
    league: str
    event: str
    market: str
    selection: str
    start_time: str
    home_team: str
    away_team: str


@dataclass
class CompletedFootballGame:
    event_id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: str
    home_goals: int
    away_goals: int
    last_update: str


@dataclass
class FootballSettlementSummary:
    open_bets: int
    sport_keys: int
    score_events: int
    matched_bets: int
    settled_won: int
    settled_lost: int
    settled_void: int
    unmatched_bets: int
    api_errors: int
    diagnostics_file: str = ""


class FootballSettlementEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        days_from: int = DEFAULT_DAYS_FROM,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.api_key = str(settings.odds_api_key or "").strip()
        self.days_from = max(1, min(3, int(days_from)))
        self.alias_engine = FootballTeamAliasEngine()

        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _table_exists(
        conn: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone()

        return row is not None

    @staticmethod
    def _columns(
        conn: sqlite3.Connection,
        table_name: str,
    ) -> set[str]:
        return {
            str(row[1])
            for row in conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        }

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            if not self._table_exists(conn, "sport_bets"):
                return

            columns = self._columns(conn, "sport_bets")

            additions = {
                "home_goals": "INTEGER",
                "away_goals": "INTEGER",
                "final_score": "TEXT",
                "settled_at": "TEXT",
                "settlement_source": "TEXT",
                "external_event_id": "TEXT",
            }

            for column, sql_type in additions.items():
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE sport_bets "
                        f"ADD COLUMN {column} {sql_type}"
                    )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_settlement_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bet_id INTEGER NOT NULL,
                    source_hash TEXT,
                    sport_key TEXT,
                    event TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    external_event_id TEXT,
                    settled_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_settlement_bet
                ON football_settlement_audit (bet_id, settled_at)
                """
            )

            conn.commit()

    def load_open_bets(self) -> list[OpenFootballBet]:
        with self.connect() as conn:
            if not self._table_exists(conn, "sport_bets"):
                return []

            bet_columns = self._columns(conn, "sport_bets")

            if not {
                "id",
                "sport",
                "league",
                "event",
                "market",
                "selection",
                "start_time",
                "result",
            }.issubset(bet_columns):
                return []

            feature_exists = self._table_exists(
                conn,
                "football_feature_history",
            )

            feature_columns = (
                self._columns(conn, "football_feature_history")
                if feature_exists
                else set()
            )

            rows = conn.execute(
                """
                SELECT *
                FROM sport_bets
                WHERE sport='football'
                  AND UPPER(COALESCE(result, 'OPEN'))='OPEN'
                ORDER BY id
                """
            ).fetchall()

            open_bets: list[OpenFootballBet] = []

            for row in rows:
                teams = split_event(row["event"])

                if teams is None:
                    continue

                source_hash = (
                    normalize_text(row["source_hash"])
                    if "source_hash" in bet_columns
                    else ""
                )

                sport_key = ""

                if feature_exists:
                    feature_row = None

                    if source_hash and "source_hash" in feature_columns:
                        feature_row = conn.execute(
                            """
                            SELECT sport_key
                            FROM football_feature_history
                            WHERE source_hash=?
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (source_hash,),
                        ).fetchone()

                    if feature_row is None:
                        feature_row = conn.execute(
                            """
                            SELECT sport_key
                            FROM football_feature_history
                            WHERE league=?
                              AND event=?
                              AND selection=?
                              AND commence_time=?
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (
                                row["league"],
                                row["event"],
                                row["selection"],
                                row["start_time"],
                            ),
                        ).fetchone()

                    if feature_row is not None:
                        sport_key = normalize_text(
                            feature_row["sport_key"]
                        )

                if not sport_key:
                    continue

                home_team, away_team = teams

                open_bets.append(
                    OpenFootballBet(
                        bet_id=int(row["id"]),
                        source_hash=source_hash,
                        sport_key=sport_key,
                        league=normalize_text(row["league"]),
                        event=normalize_text(row["event"]),
                        market=normalize_text(row["market"]),
                        selection=normalize_text(row["selection"]),
                        start_time=normalize_text(row["start_time"]),
                        home_team=home_team,
                        away_team=away_team,
                    )
                )

        return open_bets

    def _fetch_scores_sync(
        self,
        sport_key: str,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is missing")

        query = urllib.parse.urlencode(
            {
                "apiKey": self.api_key,
                "daysFrom": self.days_from,
                "dateFormat": "iso",
            }
        )
        url = (
            f"{API_HOST}/v4/sports/"
            f"{urllib.parse.quote(sport_key)}/scores/?{query}"
        )

        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Football-v13-settlement/1.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=30,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Scores API {exc.code} for {sport_key}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Scores API connection failed for "
                f"{sport_key}: {exc}"
            ) from exc

        return payload if isinstance(payload, list) else []

    async def fetch_scores(
        self,
        sport_key: str,
    ) -> list[CompletedFootballGame]:
        payload = await asyncio.to_thread(
            self._fetch_scores_sync,
            sport_key,
        )

        games: list[CompletedFootballGame] = []

        for item in payload:
            if not item.get("completed"):
                continue

            scores = item.get("scores")

            if not isinstance(scores, list):
                continue

            home_team = normalize_text(item.get("home_team"))
            away_team = normalize_text(item.get("away_team"))
            score_map = {
                normalize_team(score.get("name")): safe_int(
                    score.get("score")
                )
                for score in scores
                if isinstance(score, dict)
            }

            home_goals = score_map.get(normalize_team(home_team))
            away_goals = score_map.get(normalize_team(away_team))

            if home_goals is None or away_goals is None:
                continue

            games.append(
                CompletedFootballGame(
                    event_id=normalize_text(item.get("id")),
                    sport_key=normalize_text(
                        item.get("sport_key")
                    )
                    or sport_key,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=normalize_text(
                        item.get("commence_time")
                    ),
                    home_goals=home_goals,
                    away_goals=away_goals,
                    last_update=normalize_text(
                        item.get("last_update")
                    )
                    or now_utc(),
                )
            )

        return games

    def _team_match(self, left: str, right: str) -> bool:
        return self.alias_engine.is_match(
            left,
            right,
            threshold=0.84,
        )

    def _match_game(
        self,
        bet: OpenFootballBet,
        games: list[CompletedFootballGame],
    ) -> CompletedFootballGame | None:
        exact_candidates = [
            game
            for game in games
            if self._team_match(game.home_team, bet.home_team)
            and self._team_match(game.away_team, bet.away_team)
        ]

        if not exact_candidates:
            reversed_candidates = [
                game
                for game in games
                if self._team_match(game.home_team, bet.away_team)
                and self._team_match(game.away_team, bet.home_team)
            ]

            # Do not silently settle reversed fixtures. This only helps
            # identify provider naming/order issues during diagnostics.
            if reversed_candidates:
                return None

            return None

        if len(exact_candidates) == 1:
            return exact_candidates[0]

        bet_time = parse_datetime(bet.start_time)

        if bet_time is None:
            return exact_candidates[0]

        timed = []

        for game in exact_candidates:
            game_time = parse_datetime(game.commence_time)

            if game_time is None:
                continue

            timed.append(
                (
                    abs(
                        (game_time - bet_time).total_seconds()
                    ),
                    game,
                )
            )

        if not timed:
            return exact_candidates[0]

        timed.sort(key=lambda item: item[0])

        # Reject a same-team match if start times differ by more than 12h.
        if timed[0][0] > 12 * 60 * 60:
            return None

        return timed[0][1]

    @staticmethod
    def _settle_result(
        bet: OpenFootballBet,
        game: CompletedFootballGame,
    ) -> str:
        if bet.market.casefold() not in {"h2h", "1x2"}:
            return "VOID"

        selection = normalize_team(bet.selection)
        home = normalize_team(game.home_team)
        away = normalize_team(game.away_team)

        if game.home_goals > game.away_goals:
            winner = home
        elif game.away_goals > game.home_goals:
            winner = away
        else:
            winner = "draw"

        if selection in {"draw", "x", "remíza", "remiza"}:
            selected = "draw"
        elif selection == home:
            selected = home
        elif selection == away:
            selected = away
        else:
            return "VOID"

        return "WON" if selected == winner else "LOST"

    def _save_settlement(
        self,
        bet: OpenFootballBet,
        game: CompletedFootballGame,
        result: str,
    ) -> bool:
        settled_at = game.last_update or now_utc()

        with self.connect() as conn:
            before = conn.total_changes

            conn.execute(
                """
                UPDATE sport_bets
                SET
                    result=?,
                    home_goals=?,
                    away_goals=?,
                    final_score=?,
                    settled_at=?,
                    settlement_source='the_odds_api_scores',
                    external_event_id=?
                WHERE id=?
                  AND UPPER(COALESCE(result, 'OPEN'))='OPEN'
                """,
                (
                    result,
                    game.home_goals,
                    game.away_goals,
                    f"{game.home_goals}-{game.away_goals}",
                    settled_at,
                    game.event_id,
                    bet.bet_id,
                ),
            )

            updated = conn.total_changes > before

            if updated:
                conn.execute(
                    """
                    INSERT INTO football_settlement_audit (
                        bet_id,
                        source_hash,
                        sport_key,
                        event,
                        selection,
                        home_goals,
                        away_goals,
                        result,
                        external_event_id,
                        settled_at,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bet.bet_id,
                        bet.source_hash,
                        bet.sport_key,
                        bet.event,
                        bet.selection,
                        game.home_goals,
                        game.away_goals,
                        result,
                        game.event_id,
                        settled_at,
                        now_utc(),
                    ),
                )

                if self._table_exists(
                    conn,
                    "football_feature_history",
                ):
                    if bet.source_hash:
                        conn.execute(
                            """
                            UPDATE football_feature_history
                            SET result=?, settled_at=?
                            WHERE source_hash=?
                              AND result='OPEN'
                            """,
                            (
                                result,
                                settled_at,
                                bet.source_hash,
                            ),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE football_feature_history
                            SET result=?, settled_at=?
                            WHERE league=?
                              AND event=?
                              AND selection=?
                              AND commence_time=?
                              AND result='OPEN'
                            """,
                            (
                                result,
                                settled_at,
                                bet.league,
                                bet.event,
                                bet.selection,
                                bet.start_time,
                            ),
                        )

            conn.commit()

        return updated

    def _write_diagnostics(
        self,
        open_bets: list[OpenFootballBet],
        scores_by_key: dict[str, list[CompletedFootballGame]],
    ) -> str:
        if os.getenv("FOOTBALL_SETTLEMENT_DIAGNOSTICS", "1") != "1":
            return ""

        output = Path(
            os.getenv(
                "FOOTBALL_SETTLEMENT_DIAGNOSTICS_FILE",
                "exports/football_settlement_diagnostics.txt",
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "FOOTBALL SETTLEMENT DIAGNOSTICS",
            "=" * 60,
            "",
            f"Open bets: {len(open_bets)}",
            f"Score events: {sum(len(v) for v in scores_by_key.values())}",
            "",
            "=== OPEN BETS ===",
        ]

        for bet in open_bets:
            lines.append(
                f"{bet.sport_key} | {bet.start_time} | "
                f"{bet.home_team} vs {bet.away_team} | "
                f"pick={bet.selection}"
            )

        lines.extend(["", "=== SCORE EVENTS ==="])

        for sport_key, games in sorted(scores_by_key.items()):
            for game in games:
                lines.append(
                    f"{sport_key} | {game.commence_time} | "
                    f"{game.home_team} vs {game.away_team} | "
                    f"{game.home_goals}-{game.away_goals}"
                )

        lines.extend(["", "=== CLOSEST CANDIDATES ==="])

        for bet in open_bets:
            candidates: list[tuple[float, float, CompletedFootballGame]] = []

            for game in scores_by_key.get(bet.sport_key, []):
                home_similarity = self.alias_engine.similarity(
                    game.home_team,
                    bet.home_team,
                )
                away_similarity = self.alias_engine.similarity(
                    game.away_team,
                    bet.away_team,
                )
                combined = (home_similarity + away_similarity) / 2.0

                if combined > 0:
                    candidates.append(
                        (
                            combined,
                            min(home_similarity, away_similarity),
                            game,
                        )
                    )

            candidates.sort(
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )

            if not candidates:
                lines.append(
                    f"{bet.sport_key} | {bet.event} | no score candidates"
                )
                continue

            combined, minimum, game = candidates[0]
            lines.append(
                f"{bet.sport_key} | {bet.event} -> "
                f"{game.home_team} vs {game.away_team} | "
                f"combined={combined:.3f} min={minimum:.3f} | "
                f"bet_time={bet.start_time} score_time={game.commence_time}"
            )

        output.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        return str(output)

    async def run(self) -> FootballSettlementSummary:
        open_bets = self.load_open_bets()
        sport_keys = sorted(
            {bet.sport_key for bet in open_bets}
        )

        scores_by_key: dict[str, list[CompletedFootballGame]] = {}
        api_errors = 0

        for sport_key in sport_keys:
            try:
                scores_by_key[sport_key] = await self.fetch_scores(
                    sport_key
                )
            except Exception as exc:
                api_errors += 1
                log.warning(
                    "Football scores fetch failed for %s: %s",
                    sport_key,
                    exc,
                )
                scores_by_key[sport_key] = []

        diagnostics_file = self._write_diagnostics(
            open_bets,
            scores_by_key,
        )

        matched_bets = 0
        won = 0
        lost = 0
        void = 0
        unmatched = 0

        for bet in open_bets:
            game = self._match_game(
                bet,
                scores_by_key.get(bet.sport_key, []),
            )

            if game is None:
                unmatched += 1
                continue

            result = self._settle_result(bet, game)

            if not self._save_settlement(
                bet,
                game,
                result,
            ):
                continue

            matched_bets += 1

            if result == "WON":
                won += 1
            elif result == "LOST":
                lost += 1
            else:
                void += 1

        return FootballSettlementSummary(
            open_bets=len(open_bets),
            sport_keys=len(sport_keys),
            score_events=sum(
                len(games)
                for games in scores_by_key.values()
            ),
            matched_bets=matched_bets,
            settled_won=won,
            settled_lost=lost,
            settled_void=void,
            unmatched_bets=unmatched,
            api_errors=api_errors,
            diagnostics_file=diagnostics_file,
        )


async def settle_football_bets(
    settings: Settings,
    *,
    days_from: int = DEFAULT_DAYS_FROM,
) -> FootballSettlementSummary:
    engine = FootballSettlementEngine(
        settings,
        days_from=days_from,
    )
    return await engine.run()


if __name__ == "__main__":
    async def _main() -> None:
        settings = Settings.from_env()
        summary = await settle_football_bets(settings)

        print(
            "Football settlement: "
            f"open={summary.open_bets}, "
            f"sport_keys={summary.sport_keys}, "
            f"scores={summary.score_events}, "
            f"matched={summary.matched_bets}, "
            f"won={summary.settled_won}, "
            f"lost={summary.settled_lost}, "
            f"void={summary.settled_void}, "
            f"unmatched={summary.unmatched_bets}, "
            f"api_errors={summary.api_errors}, "
            f"diagnostics={summary.diagnostics_file or 'disabled'}"
        )

    asyncio.run(_main())
