from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SNAPSHOT_SOURCE = Path("exports/history_football_market_snapshots_v14.csv")
POSTMATCH_SOURCE = Path("exports/history_football_postmatch_dataset_v14.csv")
REPORT_CSV = Path("exports/football_match_diagnostics_v15_31.csv")
REPORT_JSON = Path("exports/football_match_diagnostics_v15_31.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _detect_column(
    headers: list[str],
    exact_aliases: Iterable[str],
    required_tokens: Iterable[str] = (),
    forbidden_tokens: Iterable[str] = (),
) -> str:
    normalized = {header: _header_key(header) for header in headers}
    aliases = {_header_key(alias) for alias in exact_aliases}

    for header, key in normalized.items():
        if key in aliases:
            return header

    required = tuple(_header_key(token) for token in required_tokens)
    forbidden = tuple(_header_key(token) for token in forbidden_tokens)

    candidates: list[tuple[int, str]] = []

    for header, key in normalized.items():
        if required and not all(token in key for token in required):
            continue
        if any(token in key for token in forbidden):
            continue

        score = sum(len(token) for token in required)
        if "name" in key:
            score += 3
        if "team" in key:
            score += 3

        candidates.append((score, header))

    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else ""


def _detect_schema(headers: list[str]) -> dict[str, str]:
    return {
        "match_id": _detect_column(
            headers,
            ("match_id", "event_id", "fixture_id", "game_id", "id"),
            required_tokens=("id",),
        ),
        "league": _detect_column(
            headers,
            ("league", "league_name", "competition", "competition_name",
             "sport_title", "sport_key", "tournament"),
            required_tokens=("league",),
        ),
        "home_team": _detect_column(
            headers,
            (
                "home_team", "home", "team_home", "home_name",
                "home_team_name", "hometeam", "hometeamname",
                "home_club", "host_team", "team1",
            ),
            required_tokens=("home",),
            forbidden_tokens=("score", "goal", "odds", "probability", "xg", "elo"),
        ),
        "away_team": _detect_column(
            headers,
            (
                "away_team", "away", "team_away", "away_name",
                "away_team_name", "awayteam", "awayteamname",
                "away_club", "visitor_team", "team2",
            ),
            required_tokens=("away",),
            forbidden_tokens=("score", "goal", "odds", "probability", "xg", "elo"),
        ),
        "kickoff": _detect_column(
            headers,
            (
                "commence_time", "kickoff_time", "start_time", "event_time",
                "match_time", "fixture_time", "datetime", "date",
            ),
            required_tokens=("time",),
            forbidden_tokens=("snapshot", "captured", "created", "updated"),
        ),
    }


def _value(row: dict[str, Any], column: str) -> str:
    if not column:
        return ""

    value = row.get(column)
    if value is None:
        return ""

    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _similarity(a: str, b: str) -> float:
    left = _normalize_text(a)
    right = _normalize_text(b)

    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    return round(SequenceMatcher(None, left, right).ratio(), 4)


def _parse_datetime(value: str) -> datetime | None:
    raw = (value or "").strip()

    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _time_difference_minutes(a: str, b: str) -> float | None:
    first = _parse_datetime(a)
    second = _parse_datetime(b)

    if first is None or second is None:
        return None

    return round(abs((first - second).total_seconds()) / 60.0, 1)


def run_match_diagnostics_v15_31(
    snapshot_source: str = str(SNAPSHOT_SOURCE),
    postmatch_source: str = str(POSTMATCH_SOURCE),
    top_candidates: int = 3,
) -> dict[str, Any]:
    snapshot_rows, snapshot_headers = _read_csv(Path(snapshot_source))
    postmatch_rows, postmatch_headers = _read_csv(Path(postmatch_source))

    snapshot_schema = _detect_schema(snapshot_headers)
    postmatch_schema = _detect_schema(postmatch_headers)

    snapshots = [
        {
            "match_id": _value(row, snapshot_schema["match_id"]),
            "league": _value(row, snapshot_schema["league"]),
            "home_team": _value(row, snapshot_schema["home_team"]),
            "away_team": _value(row, snapshot_schema["away_team"]),
            "kickoff": _value(row, snapshot_schema["kickoff"]),
        }
        for row in snapshot_rows
    ]

    postmatches = [
        {
            "match_id": _value(row, postmatch_schema["match_id"]),
            "league": _value(row, postmatch_schema["league"]),
            "home_team": _value(row, postmatch_schema["home_team"]),
            "away_team": _value(row, postmatch_schema["away_team"]),
            "kickoff": _value(row, postmatch_schema["kickoff"]),
        }
        for row in postmatch_rows
    ]

    output_rows: list[dict[str, Any]] = []
    likely_matches = 0

    for postmatch_index, postmatch in enumerate(postmatches):
        candidates: list[dict[str, Any]] = []

        for snapshot_index, snapshot in enumerate(snapshots):
            direct_home = _similarity(postmatch["home_team"], snapshot["home_team"])
            direct_away = _similarity(postmatch["away_team"], snapshot["away_team"])
            reverse_home = _similarity(postmatch["home_team"], snapshot["away_team"])
            reverse_away = _similarity(postmatch["away_team"], snapshot["home_team"])

            direct_score = (direct_home + direct_away) / 2
            reverse_score = (reverse_home + reverse_away) / 2
            reversed_order = reverse_score > direct_score

            if reversed_order:
                home_similarity = reverse_home
                away_similarity = reverse_away
                team_score = reverse_score
            else:
                home_similarity = direct_home
                away_similarity = direct_away
                team_score = direct_score

            kickoff_diff = _time_difference_minutes(
                postmatch["kickoff"],
                snapshot["kickoff"],
            )

            if kickoff_diff is None:
                time_score = 0.0
            elif kickoff_diff <= 5:
                time_score = 1.0
            elif kickoff_diff <= 60:
                time_score = 0.8
            elif kickoff_diff <= 180:
                time_score = 0.5
            else:
                time_score = 0.0

            league_similarity = _similarity(
                postmatch["league"],
                snapshot["league"],
            )

            match_score = round(
                team_score * 0.75
                + time_score * 0.20
                + league_similarity * 0.05,
                4,
            )

            candidates.append(
                {
                    "snapshot_index": snapshot_index,
                    "snapshot": snapshot,
                    "match_score": match_score,
                    "home_similarity": home_similarity,
                    "away_similarity": away_similarity,
                    "league_similarity": league_similarity,
                    "kickoff_diff_min": kickoff_diff,
                    "reversed_home_away": reversed_order,
                }
            )

        candidates.sort(key=lambda item: item["match_score"], reverse=True)

        for rank, candidate in enumerate(
            candidates[:max(1, top_candidates)],
            start=1,
        ):
            team_min = min(
                candidate["home_similarity"],
                candidate["away_similarity"],
            )

            if team_min >= 0.90 and (
                candidate["kickoff_diff_min"] is None
                or candidate["kickoff_diff_min"] <= 180
            ):
                reason = "likely_match"
            elif not postmatch["home_team"] or not postmatch["away_team"]:
                reason = "postmatch_team_columns_unresolved"
            elif team_min < 0.55:
                reason = "team_names_different"
            elif (
                candidate["kickoff_diff_min"] is not None
                and candidate["kickoff_diff_min"] > 180
            ):
                reason = "kickoff_time_mismatch"
            else:
                reason = "manual_review"

            if rank == 1 and reason == "likely_match":
                likely_matches += 1

            snapshot = candidate["snapshot"]

            output_rows.append(
                {
                    "postmatch_index": postmatch_index,
                    "postmatch_match_id": postmatch["match_id"],
                    "postmatch_league": postmatch["league"],
                    "postmatch_home": postmatch["home_team"],
                    "postmatch_away": postmatch["away_team"],
                    "postmatch_kickoff": postmatch["kickoff"],
                    "candidate_rank": rank,
                    "snapshot_index": candidate["snapshot_index"],
                    "snapshot_match_id": snapshot["match_id"],
                    "snapshot_league": snapshot["league"],
                    "snapshot_home": snapshot["home_team"],
                    "snapshot_away": snapshot["away_team"],
                    "snapshot_kickoff": snapshot["kickoff"],
                    "match_score": candidate["match_score"],
                    "home_similarity": candidate["home_similarity"],
                    "away_similarity": candidate["away_similarity"],
                    "league_similarity": candidate["league_similarity"],
                    "kickoff_diff_min": candidate["kickoff_diff_min"],
                    "reversed_home_away": candidate["reversed_home_away"],
                    "reason": reason,
                }
            )

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "postmatch_index",
        "reason",
    ]

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "version": "v15.31",
        "created_at": _now(),
        "snapshots": len(snapshots),
        "postmatch_rows": len(postmatches),
        "diagnostic_rows": len(output_rows),
        "likely_matches": likely_matches,
        "snapshot_headers": snapshot_headers,
        "postmatch_headers": postmatch_headers,
        "snapshot_schema": snapshot_schema,
        "postmatch_schema": postmatch_schema,
        "postmatch_team_rows_ready": sum(
            1 for row in postmatches
            if row["home_team"] and row["away_team"]
        ),
        "csv_report": str(REPORT_CSV),
        "status": "READY" if snapshots and postmatches else "BUILDING",
    }

    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report
