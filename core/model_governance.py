from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIN_SAMPLES = {"sport": 100, "market": 50, "league": 50, "challenger": 75}
SUPPORTED_SPORTS = {"football", "tennis", "basketball", "hockey", "baseball", "handball", "nfl"}


def _mode(metrics: dict[str, Any], minimum: int) -> str:
    settled = int(metrics.get("settled", 0) or 0)
    clv = metrics.get("average_clv_pct")
    yield_pct = float(metrics.get("yield_pct", 0) or 0)
    if settled >= 30 and ((clv is not None and float(clv) <= -2.5) or yield_pct <= -10):
        return "BLOCKED"
    if settled < minimum:
        return "SHADOW"
    if yield_pct < 0 or (clv is not None and float(clv) < 0):
        return "SHADOW"
    return "LIVE"


def _coach(name: str, metrics: dict[str, Any], minimum: int) -> dict[str, Any]:
    settled = int(metrics.get("settled", 0) or 0)
    ready = settled >= minimum
    suggestions: list[str] = []
    if not ready:
        suggestions.append(f"Zbierať dáta: {settled}/{minimum} uzavretých tipov.")
    else:
        if float(metrics.get("yield_pct", 0) or 0) < 0:
            suggestions.append("Sprísniť minimálny edge o 1 percentuálny bod.")
        if metrics.get("average_clv_pct") is not None and float(metrics["average_clv_pct"]) < 0:
            suggestions.append("Preveriť načasovanie tipu a kvalitu opening kurzu.")
        if metrics.get("calibration_error") is not None and float(metrics["calibration_error"]) > .08:
            suggestions.append("Prekalibrovať pravdepodobnosti na časovo oddelenej vzorke.")
        if not suggestions:
            suggestions.append("Ponechať prahy bez zmeny.")
    return {"segment": name, "status": "READY" if ready else "HOLD", "auto_apply": False, "samples": settled, "minimum_samples": minimum, "recommendations": suggestions}


def _source_quality(health: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in health.get("sports", []):
        snapshots = int(row.get("snapshots_24h", 0) or 0)
        missing = int(row.get("missing_event_id", 0) or 0)
        score = 100
        if snapshots == 0:
            score -= 35
        if missing:
            score -= min(40, missing * 5)
        if int(row.get("settled_bets", 0) or 0) == 0:
            score -= 20
        result.append({"sport": row.get("sport"), "score": max(0, score), "status": "GOOD" if score >= 80 else "PARTIAL" if score >= 50 else "MISSING", "odds_feed": snapshots > 0, "stable_event_ids": missing == 0, "settlement_samples": int(row.get("settled_bets", 0) or 0)})
    return result


def build_model_governance(professional_table: dict[str, Any], operational_health: dict[str, Any], export_dir: str | Path = "exports") -> dict[str, Any]:
    sports = professional_table.get("sports", {})
    sport_modes = []
    coach = []
    for sport, data in sorted(sports.items()):
        if sport not in SUPPORTED_SPORTS:
            continue
        metrics = data.get("all_time", {})
        sport_modes.append({"sport": sport, "mode": _mode(metrics, MIN_SAMPLES["sport"]), "metrics": metrics})
        coach.append(_coach(sport, metrics, MIN_SAMPLES["sport"]))

    football = sports.get("football", {})
    market_aliases = {
        "1X2": {"h2h", "1x2"}, "DOUBLE_CHANCE": {"double_chance", "double chance"},
        "OVER_UNDER": {"totals", "totals_2.5", "over_under", "goals"}, "BTTS": {"btts", "both_teams_to_score"},
    }
    raw_markets = football.get("by_market", {})
    market_modes = []
    for label, aliases in market_aliases.items():
        rows = [value for key, value in raw_markets.items() if str(key).casefold() in aliases]
        metrics = max(rows, key=lambda item: int(item.get("settled", 0) or 0)) if rows else {"settled": 0, "yield_pct": 0}
        market_modes.append({"market": label, "mode": _mode(metrics, MIN_SAMPLES["market"]), "metrics": metrics})
        coach.append(_coach(f"football/{label}", metrics, MIN_SAMPLES["market"]))

    leagues = []
    for name, metrics in football.get("by_league", {}).items():
        leagues.append({"league": name, "mode": _mode(metrics, MIN_SAMPLES["league"]), "metrics": metrics})
    leagues.sort(key=lambda item: int(item["metrics"].get("settled", 0) or 0), reverse=True)

    comparisons = []
    for sport, data in sorted(sports.items()):
        if sport not in SUPPORTED_SPORTS:
            continue
        champion = data.get("all_time", {})
        samples = int(champion.get("settled", 0) or 0)
        comparisons.append({
            "sport": sport, "champion": champion,
            "challenger": {"name": "calibrated-conservative-v1", "mode": "SHADOW", "evaluated_samples": samples, "minimum_samples": MIN_SAMPLES["challenger"]},
            "decision": "EVALUATE" if samples >= MIN_SAMPLES["challenger"] else "COLLECTING",
            "auto_promote": False,
        })

    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "principle": "Odporúčania sa nikdy neaplikujú automaticky.",
        "sport_modes": sport_modes, "football_market_modes": market_modes,
        "football_league_trust": leagues, "weekly_model_coach": coach,
        "champion_challenger": comparisons, "source_quality": _source_quality(operational_health),
    }
    target = Path(export_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model_governance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def model_governance_report(payload: dict[str, Any]) -> str:
    markets = ", ".join(f"{row['market']}={row['mode']}" for row in payload.get("football_market_modes", []))
    return "\n=== MODEL GOVERNANCE ===\n" + f"Football markets: {markets}\n" + f"Coach recommendations: {len(payload.get('weekly_model_coach', []))} (auto-apply OFF)\n"
