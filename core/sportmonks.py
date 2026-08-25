from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from core.sport_context import SportContextDatabase


API_BASE = "https://api.sportmonks.com/v3/football"


class SportmonksError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, detail: str = ""):
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class SyncSummary:
    fixtures_received: int = 0
    snapshots_added: int = 0
    context_rows_added: int = 0
    confirmed_lineups: int = 0
    sidelined_records: int = 0
    xg_records: int = 0
    home_absence_impact: float = 0.0
    away_absence_impact: float = 0.0


def _explicit_lineup_confirmation(metadata: Any) -> bool:
    """Accept only an explicit provider confirmation, never infer it from a lineup."""
    if isinstance(metadata, list):
        return any(_explicit_lineup_confirmation(item) for item in metadata)
    if not isinstance(metadata, dict):
        return False

    label_parts = []
    for key in ("key", "name", "developer_name", "code", "type"):
        value = metadata.get(key)
        if isinstance(value, str):
            label_parts.append(value.lower().replace("-", "_").replace(" ", "_"))
        elif isinstance(value, dict):
            label_parts.extend(
                str(value.get(n, "")).lower().replace("-", "_").replace(" ", "_")
                for n in ("name", "developer_name", "code")
            )
    is_confirmation = any("lineup_confirmed" in label for label in label_parts)
    if is_confirmation:
        for key in ("value", "values", "data"):
            value = metadata.get(key)
            if value is True or value == 1 or str(value).strip().lower() == "true":
                return True
            if isinstance(value, dict) and any(
                child is True or child == 1 or str(child).strip().lower() == "true"
                for child in value.values()
            ):
                return True
    return any(_explicit_lineup_confirmation(value) for value in metadata.values())


def _complete_starting_lineups(fixture: dict[str, Any]) -> bool:
    """Recognize two complete official starting XIs from the lineup feed."""
    _, _, home_id, away_id = _fixture_teams(fixture)
    if not home_id or not away_id:
        return False
    starters = {home_id: set(), away_id: set()}
    for item in fixture.get("lineups") or []:
        if not isinstance(item, dict):
            continue
        team_id = str(item.get("participant_id") or item.get("team_id") or "")
        try:
            is_starter = int(item.get("type_id") or 0) == 11
        except (TypeError, ValueError):
            is_starter = False
        if team_id in starters and is_starter:
            player_id = str(item.get("player_id") or item.get("id") or "")
            if player_id:
                starters[team_id].add(player_id)
    return all(len(starters[team_id]) >= 11 for team_id in (home_id, away_id))


def _fixture_teams(fixture: dict[str, Any]) -> tuple[str, str, str, str]:
    home_name = away_name = home_id = away_id = ""
    for participant in fixture.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        location = str((participant.get("meta") or {}).get("location", "")).lower()
        if location == "home":
            home_name, home_id = str(participant.get("name", "")), str(participant.get("id", ""))
        elif location == "away":
            away_name, away_id = str(participant.get("name", "")), str(participant.get("id", ""))
    if not home_name or not away_name:
        name = str(fixture.get("name", ""))
        if " vs " in name:
            fallback_home, fallback_away = name.split(" vs ", 1)
            home_name = home_name or fallback_home.strip()
            away_name = away_name or fallback_away.strip()
    return home_name, away_name, home_id, away_id


def _player_weight(item: dict[str, Any], unit: float) -> float:
    player = item.get("player") if isinstance(item.get("player"), dict) else item
    position = str(
        player.get("position_name") or player.get("position")
        or (player.get("position") or {}).get("name", "")
    ).lower()
    position_factor = 1.0
    if "goal" in position:
        position_factor = 1.35
    elif any(label in position for label in ("forward", "attack", "striker")):
        position_factor = 1.25
    elif "mid" in position:
        position_factor = 1.10
    minutes = float(player.get("minutes_played") or player.get("minutes") or 0)
    minutes_factor = min(1.4, max(.75, minutes / 900.0)) if minutes else 1.0
    rating = float(player.get("rating") or player.get("average_rating") or 0)
    rating_factor = min(1.25, max(.85, rating / 6.5)) if rating else 1.0
    return unit * position_factor * minutes_factor * rating_factor


def _absence_impacts(fixture: dict[str, Any]) -> dict[str, float]:
    _, _, home_id, away_id = _fixture_teams(fixture)
    unit = max(0.0, min(.01, float(os.getenv("SPORTMONKS_ABSENCE_UNIT", ".002"))))
    impacts = {home_id: 0.0, away_id: 0.0}
    injuries = {home_id: 0.0, away_id: 0.0}
    suspensions = {home_id: 0.0, away_id: 0.0}
    seen: set[str] = set()
    for item in fixture.get("sidelined") or []:
        if not isinstance(item, dict):
            continue
        identity = str(item.get("sideline_id") or item.get("id") or "")
        if identity and identity in seen:
            continue
        seen.add(identity)
        team_id = str(item.get("participant_id") or item.get("team_id") or "")
        sideline = item.get("sideline") if isinstance(item.get("sideline"), dict) else item
        games_missed = float(sideline.get("games_missed") or 0)
        category = str(sideline.get("category") or "").lower()
        weight = _player_weight(item, unit) + min(unit, max(0.0, games_missed) * unit * .20)
        if "susp" in category:
            weight *= .80
            suspensions[team_id] = suspensions.get(team_id, 0.0) + weight
        else:
            injuries[team_id] = injuries.get(team_id, 0.0) + weight
        if team_id in impacts:
            impacts[team_id] += weight
    return {
        "home": min(.05, impacts.get(home_id, 0.0)),
        "away": min(.05, impacts.get(away_id, 0.0)),
        "home_injury": min(.05, injuries.get(home_id, 0.0)),
        "away_injury": min(.05, injuries.get(away_id, 0.0)),
        "home_suspension": min(.05, suspensions.get(home_id, 0.0)),
        "away_suspension": min(.05, suspensions.get(away_id, 0.0)),
    }


def _lineup_strengths(fixture: dict[str, Any]) -> tuple[float | None, float | None]:
    _, _, home_id, away_id = _fixture_teams(fixture)
    values: dict[str, list[float]] = {home_id: [], away_id: []}
    for item in fixture.get("lineups") or []:
        if not isinstance(item, dict):
            continue
        team_id = str(item.get("participant_id") or item.get("team_id") or "")
        if team_id not in values:
            continue
        values[team_id].append(_player_weight(item, 1.0))
    def score(team_id: str) -> float | None:
        weights = values.get(team_id, [])
        return round(sum(weights) / len(weights), 4) if weights else None
    return score(home_id), score(away_id)


def _actual_xg(fixture: dict[str, Any]) -> tuple[float | None, float | None]:
    _, _, home_id, away_id = _fixture_teams(fixture)
    values: dict[str, float] = {}
    payload = fixture.get("xgfixture") or fixture.get("xGFixture") or []
    if isinstance(payload, dict):
        payload = [payload]
    for item in payload:
        if not isinstance(item, dict):
            continue
        team_id = str(item.get("participant_id") or item.get("team_id") or "")
        raw = item.get("value", item.get("xg"))
        try:
            values[team_id] = float(raw)
        except (TypeError, ValueError):
            continue
    return values.get(home_id), values.get(away_id)


def _fixture_time(fixture: dict[str, Any]) -> datetime | None:
    raw = str(fixture.get("starting_at", "")).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _schedule_index(fixtures: list[dict[str, Any]]) -> dict[str, list[datetime]]:
    index: dict[str, list[datetime]] = {}
    for fixture in fixtures:
        kickoff = _fixture_time(fixture)
        if kickoff is None:
            continue
        _, _, home_id, away_id = _fixture_teams(fixture)
        for team_id in (home_id, away_id):
            if team_id:
                index.setdefault(team_id, []).append(kickoff)
    for dates in index.values():
        dates.sort()
    return index


def _schedule_load(
    fixture: dict[str, Any],
    index: dict[str, list[datetime]],
) -> tuple[float | None, float | None, float | None]:
    kickoff = _fixture_time(fixture)
    if kickoff is None:
        return None, None, None
    _, _, home_id, away_id = _fixture_teams(fixture)

    def team_load(team_id: str) -> tuple[float | None, int]:
        dates = index.get(team_id, [])
        previous = [date for date in dates if date < kickoff]
        rest = (kickoff - previous[-1]).total_seconds() / 86400.0 if previous else None
        congestion = sum(
            1 for date in dates
            if date != kickoff and abs((date - kickoff).total_seconds()) <= 7 * 86400
        )
        return rest, congestion

    home_rest, home_load = team_load(home_id)
    away_rest, away_load = team_load(away_id)
    return home_rest, away_rest, float(max(home_load, away_load))


def _explicit_metric(payload: Any, names: set[str]) -> float | None:
    """Read a provider-supplied metric; never estimate a missing value."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in names:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            found = _explicit_metric(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _explicit_metric(item, names)
            if found is not None:
                return found
    return None


class SportmonksClient:
    def __init__(self, token: str, timeout: float = 30.0, include_xg: bool = False):
        token = token.strip()
        if not token:
            raise ValueError("SPORTMONKS_API_TOKEN is missing")
        self._token = token
        self.timeout = timeout
        self.include_xg = include_xg

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({**params, "api_token": self._token})
        request = urllib.request.Request(
            f"{API_BASE}/{path}?{query}",
            headers={"Accept": "application/json", "User-Agent": "Futbal-dnes/2.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SportmonksError(
                f"Sportmonks HTTP {exc.code}: {detail}",
                status=exc.code,
                detail=detail,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SportmonksError(f"Sportmonks request failed: {exc}") from exc

    def fixtures_by_date(self, fixture_date: date, max_pages: int = 10) -> list[dict[str, Any]]:
        fixtures: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            base_includes = "participants;lineups;sidelined.sideline;metadata"
            includes = f"{base_includes};xGFixture" if self.include_xg else base_includes
            try:
                payload = self._get(
                    f"fixtures/date/{fixture_date.isoformat()}",
                    {"include": includes, "page": page},
                )
            except SportmonksError as exc:
                # xG is a paid add-on. A valid free-plan token must still be able
                # to collect the context fields that are included in its plan.
                xg_denied = exc.status == 403 and "xgfixture" in exc.detail.lower()
                if not self.include_xg or not xg_denied:
                    raise
                self.include_xg = False
                payload = self._get(
                    f"fixtures/date/{fixture_date.isoformat()}",
                    {"include": base_includes, "page": page},
                )
            data = payload.get("data", [])
            if isinstance(data, dict):
                data = [data]
            fixtures.extend(item for item in data if isinstance(item, dict))
            pagination = payload.get("pagination") or payload.get("meta", {}).get("pagination") or {}
            if not pagination.get("has_more") and not pagination.get("has_more_pages"):
                break
        return fixtures


def sync_upcoming_context(
    client: SportmonksClient,
    database: SportContextDatabase,
    *,
    start_date: date | None = None,
    days: int = 1,
) -> SyncSummary:
    start = start_date or datetime.now(timezone.utc).date()
    totals = {field: 0 for field in SyncSummary.__dataclass_fields__}
    captured_at = datetime.now(timezone.utc).isoformat()

    fixtures: list[dict[str, Any]] = []
    for offset in range(max(1, days)):
        fixtures.extend(client.fixtures_by_date(start + timedelta(days=offset)))
    schedule_index = _schedule_index(fixtures)

    for fixture in fixtures:
            totals["fixtures_received"] += 1
            fixture_id = str(fixture.get("id", "")).strip()
            event = str(fixture.get("name", "")).strip()
            if not fixture_id or not event:
                continue
            start_time = str(fixture.get("starting_at", "")).strip()
            if database.store_provider_snapshot(
                provider="sportmonks-v3",
                sport="football",
                external_event_id=fixture_id,
                event=event,
                start_time=start_time,
                payload=fixture,
                captured_at=captured_at,
            ):
                totals["snapshots_added"] += 1

            lineup_confirmed = (
                _explicit_lineup_confirmation(fixture.get("metadata", []))
                or _complete_starting_lineups(fixture)
            )
            home_team, away_team, _, _ = _fixture_teams(fixture)
            absence = _absence_impacts(fixture)
            home_impact, away_impact = absence["home"], absence["away"]
            home_lineup_strength, away_lineup_strength = _lineup_strengths(fixture)
            home_xg, away_xg = _actual_xg(fixture)
            home_rest, away_rest, schedule_congestion = _schedule_load(
                fixture, schedule_index
            )
            travel_km = _explicit_metric(
                fixture, {"travel_km", "travel_distance_km"}
            )
            sidelined = fixture.get("sidelined") or []
            xg = fixture.get("xgfixture") or fixture.get("xGFixture") or []
            totals["confirmed_lineups"] += int(lineup_confirmed)
            totals["sidelined_records"] += len(sidelined) if isinstance(sidelined, list) else 0
            totals["xg_records"] += len(xg) if isinstance(xg, list) else 0
            totals["home_absence_impact"] += home_impact
            totals["away_absence_impact"] += away_impact

            source_hash = f"sportmonks-v3:{fixture_id}:{captured_at}"
            with database.connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO sport_context_features (
                        sport, league, event, external_event_id, start_time,
                        lineup_confirmed, injury_impact, suspension_impact,
                        rest_days, travel_km,
                        home_team, away_team, home_absence_impact, away_absence_impact,
                        home_injury_impact, away_injury_impact,
                        home_suspension_impact, away_suspension_impact,
                        home_lineup_strength, away_lineup_strength,
                        home_xg, away_xg,
                        home_rest_days, away_rest_days, schedule_congestion,
                        source, captured_at, source_hash
                    ) VALUES ('football', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(fixture.get("league_id", "")), event, fixture_id, start_time,
                        int(lineup_confirmed),
                        min(
                            value for value in (home_rest, away_rest)
                            if value is not None
                        ) if home_rest is not None or away_rest is not None else None,
                        travel_km,
                        home_team, away_team,
                        home_impact, away_impact,
                        absence["home_injury"], absence["away_injury"],
                        absence["home_suspension"], absence["away_suspension"],
                        home_lineup_strength, away_lineup_strength,
                        home_xg, away_xg,
                        home_rest, away_rest, schedule_congestion,
                        "sportmonks-v3", captured_at, source_hash,
                    ),
                )
            totals["context_rows_added"] += int(cursor.rowcount == 1)

    return SyncSummary(**totals)


