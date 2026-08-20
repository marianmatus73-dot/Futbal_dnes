from __future__ import annotations

import csv
import html
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


@dataclass(frozen=True)
class ModelMetrics:
    settled: int = 0
    stake: float = 0.0
    profit: float = 0.0
    yield_pct: float = 0.0
    average_clv_pct: float | None = None
    clv_samples: int = 0
    brier_score: float | None = None
    calibration_error: float | None = None
    probability_samples: int = 0
    max_drawdown: float = 0.0


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _target(result: Any) -> int | None:
    value = str(result or "").strip().upper()
    if value in {"WON", "WIN", "V"}:
        return 1
    if value in {"LOST", "LOSS", "P"}:
        return 0
    return None


def _parse_time(value: Any) -> datetime | None:
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


def _metrics(rows: list[sqlite3.Row]) -> ModelMetrics:
    stake = sum(_as_float(row["stake"]) or 0.0 for row in rows)
    profit = sum(_as_float(row["profit"]) or 0.0 for row in rows)
    clv = [
        value for row in rows
        if (value := _as_float(row["clv_pct"])) is not None
    ]
    probabilities = []
    for row in rows:
        probability = _as_float(row["prob_final"])
        target = _target(row["result"])
        if probability is not None and target is not None and 0.0 < probability < 1.0:
            probabilities.append((probability, target))

    brier = None
    calibration_error = None
    if probabilities:
        brier = sum((p - y) ** 2 for p, y in probabilities) / len(probabilities)
        bins: dict[int, list[tuple[float, int]]] = {}
        for probability, target in probabilities:
            bins.setdefault(min(9, int(probability * 10)), []).append((probability, target))
        calibration_error = sum(
            len(values) / len(probabilities)
            * abs(
                sum(p for p, _ in values) / len(values)
                - sum(y for _, y in values) / len(values)
            )
            for values in bins.values()
        )

    balance = peak = max_drawdown = 0.0
    for row in sorted(
        rows,
        key=lambda item: _parse_time(item["settled_at"] or item["start_time"])
        or datetime.min.replace(tzinfo=timezone.utc),
    ):
        balance += _as_float(row["profit"]) or 0.0
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)

    return ModelMetrics(
        settled=len(rows),
        stake=round(stake, 4),
        profit=round(profit, 4),
        yield_pct=round(profit / stake * 100.0, 3) if stake > 0 else 0.0,
        average_clv_pct=round(sum(clv) / len(clv) * 100.0, 3) if clv else None,
        clv_samples=len(clv),
        brier_score=round(brier, 6) if brier is not None else None,
        calibration_error=(
            round(calibration_error, 6) if calibration_error is not None else None
        ),
        probability_samples=len(probabilities),
        max_drawdown=round(max_drawdown, 4),
    )


def _odds_bucket(value: Any) -> str:
    odds = _as_float(value) or 0.0
    if odds < 1.50:
        return "<1.50"
    if odds < 2.00:
        return "1.50-1.99"
    if odds < 3.00:
        return "2.00-2.99"
    return "3.00+"


def _group(rows: list[sqlite3.Row], key) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(key(row) or "UNKNOWN"), []).append(row)
    return {name: asdict(_metrics(values)) for name, values in sorted(grouped.items())}


def build_professional_model_table(
    settings: Settings,
    export_dir: str | Path = "exports",
) -> dict[str, Any]:
    database = Path(settings.db_file or "bets.db")
    if not database.exists():
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "sports": {}}

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone()
        if exists is None:
            return {"generated_at": datetime.now(timezone.utc).isoformat(), "sports": {}}
        rows = conn.execute(
            """
            SELECT sport, league, market, odds, stake, profit, clv_pct,
                   prob_final, result, start_time, settled_at
            FROM sport_bets
            WHERE UPPER(TRIM(COALESCE(result, ''))) IN
                  ('WON', 'WIN', 'LOST', 'LOSS', 'V', 'P')
            """
        ).fetchall()

    now = datetime.now(timezone.utc)
    sports: dict[str, Any] = {}
    for sport in sorted({str(row["sport"] or "UNKNOWN") for row in rows}):
        sport_rows = [row for row in rows if str(row["sport"] or "UNKNOWN") == sport]
        periods = {}
        for days in (30, 90, 365):
            cutoff = now - timedelta(days=days)
            period_rows = []
            for row in sport_rows:
                timestamp = _parse_time(row["settled_at"] or row["start_time"])
                if timestamp is not None and timestamp >= cutoff:
                    period_rows.append(row)
            periods[str(days)] = asdict(_metrics(period_rows))
        sports[sport] = {
            "all_time": asdict(_metrics(sport_rows)),
            "periods": periods,
            "by_league": _group(sport_rows, lambda row: row["league"]),
            "by_odds": _group(sport_rows, lambda row: _odds_bucket(row["odds"])),
            "by_market": _group(sport_rows, lambda row: row["market"]),
        }

    payload = {"generated_at": now.isoformat(), "sports": sports}
    output = Path(export_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "professional_model_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "professional_model_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sport", "period_days", "settled", "stake", "profit", "yield_pct",
            "average_clv_pct", "clv_samples", "brier_score", "calibration_error",
            "probability_samples", "max_drawdown",
        ])
        for sport, data in sports.items():
            for period, metrics in [("all", data["all_time"]), *data["periods"].items()]:
                writer.writerow([sport, period, *metrics.values()])

    headers = ["Sport", "Settled", "Yield", "Profit", "Avg CLV", "Brier", "Calibration", "Max DD"]
    body = []
    for sport, data in sports.items():
        metric = data["all_time"]
        body.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            sport, metric["settled"], f'{metric["yield_pct"]:.2f}%',
            f'{metric["profit"]:.2f}', metric["average_clv_pct"],
            metric["brier_score"], metric["calibration_error"],
            f'{metric["max_drawdown"]:.2f}',
        )) + "</tr>")
    (output / "professional_model_table.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Professional Model Table</title>"
        "<h1>Professional Model Table</h1><table border='1' cellpadding='6'>"
        "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table>",
        encoding="utf-8",
    )
    return payload


def professional_model_report(payload: dict[str, Any]) -> str:
    lines = ["\n=== PROFESSIONAL MODEL TABLE ==="]
    for sport, data in payload.get("sports", {}).items():
        metric = data["all_time"]
        clv = "n/a" if metric["average_clv_pct"] is None else f'{metric["average_clv_pct"]:.2f}%'
        brier = "n/a" if metric["brier_score"] is None else f'{metric["brier_score"]:.4f}'
        lines.append(
            f"- {sport}: settled={metric['settled']} | yield={metric['yield_pct']:.2f}% | "
            f"profit={metric['profit']:.2f} | CLV={clv} ({metric['clv_samples']}) | "
            f"Brier={brier} | max DD={metric['max_drawdown']:.2f}"
        )
    return "\n".join(lines) + "\n"
