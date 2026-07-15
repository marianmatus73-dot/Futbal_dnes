from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SNAPSHOT_SOURCE = Path("exports/history_football_market_snapshots_v14.csv")
POSTMATCH_SOURCE = Path("exports/history_football_postmatch_dataset_v14.csv")
REPORT_CSV = Path("exports/football_match_diagnostics_v15_30.csv")
REPORT_JSON = Path("exports/football_match_diagnostics_v15_30.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()

    for token in (
        "football club",
        "association football club",
        "athletic football club",
        " fc ",
        " afc ",
        " cf ",
        " sc ",
    ):
        value = value.replace(token, " ")

    return re.sub(r"[^a-z0-9]+", "", value)


def _similarity(a: str, b: str) -> float:
    a_norm = _normalize(a)
    b_norm = _normalize(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    return round(SequenceMatcher(None, a_norm, b_norm).ratio(), 4)


def _parse_datetime(value: str) -> datetime | None:
    raw = (value or "").strip()

    if not raw:
        return None

    candidates = [
        raw,
        raw.replace("Z", "+00:00"),
    ]

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    )

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                continue

    return None


def _time_difference_minutes(a: str, b: str) -> float | None:
    first = _parse_datetime(a)
    second = _parse_datetime(b)

    if first is None or second is None:
        return None

    return round(abs((first - second).total_seconds()) / 60.0, 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _snapshot_fields(row: dict[str, Any]) -> dict[str, str]:
    return {
        "event_id": _first(row, ("event_id", "id", "match_id")),
        "league": _first(row, ("league", "league_name", "sport_title", "sport_key")),
        "home_team": _first(row, ("home_team", "home", "team_home", "home_name")),
        "away_team": _first(row, ("away_team", "away", "team_away", "away_name")),
        "kickoff": _first(
            row,
            (
                "commence_time",
                "kickoff_time",
                "start_time",
                "event_time",
                "match_time",
                "date",
            ),
        ),
        "snapshot_time": _first(
            row,
            (
                "snapshot_time",
                "captured_at",
                "created_at",
                "timestamp",
                "updated_at",
            ),
        ),
        "bookmaker": _first(row, ("bookmaker", "bookmaker_name", "sportsbook")),
        "odds": _first(row, ("odds", "price", "decimal_odds", "best_odds")),
    }


def _postmatch_fields(row: dict[str, Any]) -> dict[str, str]:
    return {
        "match_id": _first(row, ("match_id", "event_id", "id")),
        "league": _first(row, ("league", "league_name", "sport_title", "sport_key")),
        "home_team": _first(row, ("home_team", "home", "team_home", "home_name")),
        "away_team": _first(row, ("away_team", "away", "team_away", "away_name")),
        "kickoff": _first(
            row,
            (
                "commence_time",
                "kickoff_time",
                "start_time",
                "event_time",
                "match_time",
                "date",
            ),
        ),
    }


@dataclass
class Candidate:
    snapshot_index: int
    score: float
    direct: bool
    reverse: bool
    home_similarity: float
    away_similarity: float
    league_similarity: float
    kickoff_diff_min: float | None
    snapshot: dict[str, str]


def _candidate_score(
    postmatch: dict[str, str],
    snapshot: dict[str, str],
) -> Candidate:
    direct_home = _similarity(postmatch["home_team"], snapshot["home_team"])
    direct_away = _similarity(postmatch["away_team"], snapshot["away_team"])
    reverse_home = _similarity(postmatch["home_team"], snapshot["away_team"])
    reverse_away = _similarity(postmatch["away_team"], snapshot["home_team"])

    direct_team_score = (direct_home + direct_away) / 2
    reverse_team_score = (reverse_home + reverse_away) / 2
    reverse = reverse_team_score > direct_team_score

    if reverse:
        home_similarity = reverse_home
        away_similarity = reverse_away
        team_score = reverse_team_score
    else:
        home_similarity = direct_home
        away_similarity = direct_away
        team_score = direct_team_score

    league_similarity = _similarity(postmatch["league"], snapshot["league"])
    kickoff_diff = _time_difference_minutes(postmatch["kickoff"], snapshot["kickoff"])

    if kickoff_diff is None:
        time_score = 0.0
    elif kickoff_diff <= 5:
        time_score = 1.0
    elif kickoff_diff <= 60:
        time_score = 0.8
    elif kickoff_diff <= 180:
        time_score = 0.55
    elif kickoff_diff <= 1440:
        time_score = 0.2
    else:
        time_score = 0.0

    score = round(
        (team_score * 0.72)
        + (league_similarity * 0.10)
        + (time_score * 0.18),
        4,
    )

    return Candidate(
        snapshot_index=-1,
        score=score,
        direct=not reverse,
        reverse=reverse,
        home_similarity=home_similarity,
        away_similarity=away_similarity,
        league_similarity=league_similarity,
        kickoff_diff_min=kickoff_diff,
        snapshot=snapshot,
    )


def _reason(candidate: Candidate) -> str:
    team_min = min(candidate.home_similarity, candidate.away_similarity)

    if team_min >= 0.95 and (
        candidate.kickoff_diff_min is None
        or candidate.kickoff_diff_min <= 60
    ):
        return "likely_match"

    if team_min >= 0.80 and candidate.reverse:
        return "likely_reversed_home_away"

    if team_min < 0.55:
        return "team_names_different"

    if (
        candidate.kickoff_diff_min is not None
        and candidate.kickoff_diff_min > 180
    ):
        return "kickoff_time_mismatch"

    if candidate.league_similarity < 0.45:
        return "league_mismatch"

    return "manual_review"


def run_match_diagnostics_v15_30(
    snapshot_source: str = str(SNAPSHOT_SOURCE),
    postmatch_source: str = str(POSTMATCH_SOURCE),
    top_candidates: int = 3,
) -> dict[str, Any]:
    snapshot_rows = _read_csv(Path(snapshot_source))
    postmatch_rows = _read_csv(Path(postmatch_source))

    snapshots = [_snapshot_fields(row) for row in snapshot_rows]
    postmatches = [_postmatch_fields(row) for row in postmatch_rows]

    output_rows: list[dict[str, Any]] = []
    summary_counts: dict[str, int] = {}

    for postmatch_index, postmatch in enumerate(postmatches):
        candidates: list[Candidate] = []

        for snapshot_index, snapshot in enumerate(snapshots):
            candidate = _candidate_score(postmatch, snapshot)
            candidate.snapshot_index = snapshot_index
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)

        for rank, candidate in enumerate(candidates[:max(1, top_candidates)], start=1):
            reason = _reason(candidate)
            summary_counts[reason] = summary_counts.get(reason, 0) + 1

            output_rows.append(
                {
                    "postmatch_index": postmatch_index,
                    "postmatch_match_id": postmatch["match_id"],
                    "postmatch_league": postmatch["league"],
                    "postmatch_home": postmatch["home_team"],
                    "postmatch_away": postmatch["away_team"],
                    "postmatch_kickoff": postmatch["kickoff"],
                    "candidate_rank": rank,
                    "snapshot_index": candidate.snapshot_index,
                    "snapshot_event_id": candidate.snapshot["event_id"],
                    "snapshot_league": candidate.snapshot["league"],
                    "snapshot_home": candidate.snapshot["home_team"],
                    "snapshot_away": candidate.snapshot["away_team"],
                    "snapshot_kickoff": candidate.snapshot["kickoff"],
                    "snapshot_time": candidate.snapshot["snapshot_time"],
                    "bookmaker": candidate.snapshot["bookmaker"],
                    "odds": candidate.snapshot["odds"],
                    "match_score": candidate.score,
                    "home_similarity": candidate.home_similarity,
                    "away_similarity": candidate.away_similarity,
                    "league_similarity": candidate.league_similarity,
                    "kickoff_diff_min": candidate.kickoff_diff_min,
                    "reversed_home_away": candidate.reverse,
                    "reason": reason,
                }
            )

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "postmatch_index",
        "postmatch_match_id",
        "postmatch_league",
        "postmatch_home",
        "postmatch_away",
        "postmatch_kickoff",
        "candidate_rank",
        "snapshot_index",
        "snapshot_event_id",
        "snapshot_league",
        "snapshot_home",
        "snapshot_away",
        "snapshot_kickoff",
        "snapshot_time",
        "bookmaker",
        "odds",
        "match_score",
        "home_similarity",
        "away_similarity",
        "league_similarity",
        "kickoff_diff_min",
        "reversed_home_away",
        "reason",
    ]

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    likely_matches = sum(
        1
        for row in output_rows
        if row["candidate_rank"] == 1
        and row["reason"] in ("likely_match", "likely_reversed_home_away")
    )

    report = {
        "version": "v15.30",
        "created_at": _now(),
        "snapshot_source": snapshot_source,
        "postmatch_source": postmatch_source,
        "snapshots": len(snapshots),
        "postmatch_rows": len(postmatches),
        "diagnostic_rows": len(output_rows),
        "top_candidates_per_match": top_candidates,
        "likely_matches": likely_matches,
        "reason_counts": summary_counts,
        "csv_report": str(REPORT_CSV),
        "status": "READY" if snapshots and postmatches else "BUILDING",
    }

    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report
