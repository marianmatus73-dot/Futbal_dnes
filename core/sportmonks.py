from __future__ import annotations

import json
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
            base_includes = "participants;lineups;sidelined.player;metadata"
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

    for offset in range(max(1, days)):
        for fixture in client.fixtures_by_date(start + timedelta(days=offset)):
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

            lineup_confirmed = _explicit_lineup_confirmation(fixture.get("metadata", []))
            sidelined = fixture.get("sidelined") or []
            xg = fixture.get("xgfixture") or fixture.get("xGFixture") or []
            totals["confirmed_lineups"] += int(lineup_confirmed)
            totals["sidelined_records"] += len(sidelined) if isinstance(sidelined, list) else 0
            totals["xg_records"] += len(xg) if isinstance(xg, list) else 0

            source_hash = f"sportmonks-v3:{fixture_id}:{captured_at}"
            with database.connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO sport_context_features (
                        sport, league, event, external_event_id, start_time,
                        lineup_confirmed, injury_impact, suspension_impact,
                        source, captured_at, source_hash
                    ) VALUES ('football', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    """,
                    (
                        str(fixture.get("league_id", "")), event, fixture_id, start_time,
                        int(lineup_confirmed), "sportmonks-v3", captured_at, source_hash,
                    ),
                )
            totals["context_rows_added"] += int(cursor.rowcount == 1)

    return SyncSummary(**totals)


