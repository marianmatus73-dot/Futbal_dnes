from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint
from typing import Any


SUPPORTED_SPORTS = (
    "baseball",
    "basketball",
    "tennis",
    "hockey",
    "mma",
    "nfl",
)

ALIASES = {
    "sport_name": "sport",
    "category": "sport",
    "competition": "league",
    "match": "event",
    "pick": "selection",
    "sportsbook": "bookmaker",
    "price": "odds",
    "amount": "stake",
    "stake_amount": "stake",
    "bet_result": "result",
    "outcome": "result",
    "profit_loss": "profit",
    "pnl": "profit",
    "probability": "model_probability",
    "predicted_probability": "model_probability",
    "implied_probability": "market_probability",
    "confidence_score": "confidence",
}


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema(database: Path) -> dict[str, Any]:
    with connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sport_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                league TEXT,
                event TEXT,
                selection TEXT,
                bookmaker TEXT,
                odds REAL,
                stake REAL,
                model_probability REAL,
                market_probability REAL,
                confidence REAL,
                result TEXT NOT NULL DEFAULT 'OPEN',
                profit REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                settled_at TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sport_bets_sport "
            "ON sport_bets(sport)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sport_bets_result "
            "ON sport_bets(result)"
        )
        conn.commit()

        rows = int(
            conn.execute("SELECT COUNT(*) FROM sport_bets").fetchone()[0]
        )

    return {
        "database": str(database),
        "rows": rows,
        "status": "READY",
    }


def normalize_result(value: Any) -> str:
    raw = str(value or "OPEN").strip().upper()
    if raw in {"WON", "WIN", "SUCCESS"}:
        return "WIN"
    if raw in {"LOST", "LOSS", "FAIL"}:
        return "LOSS"
    return "OPEN"


def restore_history(database: Path, csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "csv": str(csv_path),
            "imported": 0,
            "status": "CSV_NOT_FOUND",
        }

    with connect(database) as conn:
        available = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(sport_bets)"
            ).fetchall()
        }

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            imported = 0

            for raw_row in reader:
                normalized: dict[str, Any] = {}

                for key, value in raw_row.items():
                    if key is None:
                        continue
                    normalized[ALIASES.get(key, key)] = value

                normalized["result"] = normalize_result(
                    normalized.get("result")
                )

                filtered = {
                    key: value
                    for key, value in normalized.items()
                    if key in available and key != "id"
                }

                if not filtered.get("sport"):
                    continue

                columns = list(filtered)
                placeholders = ",".join("?" for _ in columns)
                names = ",".join(f'"{column}"' for column in columns)

                conn.execute(
                    f"INSERT INTO sport_bets ({names}) "
                    f"VALUES ({placeholders})",
                    [filtered[column] for column in columns],
                )
                imported += 1

            conn.commit()

    return {
        "csv": str(csv_path),
        "imported": imported,
        "status": "READY",
    }


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sport_metrics(database: Path, sport: str) -> dict[str, Any]:
    with connect(database) as conn:
        rows = conn.execute(
            """
            SELECT result, stake, odds, profit,
                   model_probability, confidence,
                   league, bookmaker
            FROM sport_bets
            WHERE LOWER(sport)=?
            """,
            (sport.lower(),),
        ).fetchall()

    wins = losses = opened = 0
    profit = stake_sum = 0.0
    brier = log_loss = 0.0
    probability_samples = 0
    confidences: list[float] = []
    leagues: set[str] = set()
    bookmakers: set[str] = set()

    for row in rows:
        result = normalize_result(row["result"])

        if result == "WIN":
            wins += 1
        elif result == "LOSS":
            losses += 1
        else:
            opened += 1

        stake = to_float(row["stake"])
        odds = to_float(row["odds"])
        explicit_profit = to_float(row["profit"])

        if stake is not None:
            stake_sum += stake

        if explicit_profit is not None:
            profit += explicit_profit
        elif result == "WIN" and stake is not None and odds is not None:
            profit += stake * max(odds - 1.0, 0.0)
        elif result == "LOSS" and stake is not None:
            profit -= stake

        probability = to_float(row["model_probability"])
        if probability is not None and result in {"WIN", "LOSS"}:
            probability = min(max(probability, 1e-6), 1 - 1e-6)
            target = 1.0 if result == "WIN" else 0.0
            brier += (probability - target) ** 2
            log_loss += -(
                target * math.log(probability)
                + (1 - target) * math.log(1 - probability)
            )
            probability_samples += 1

        confidence = to_float(row["confidence"])
        if confidence is not None:
            confidences.append(
                confidence / 100 if confidence > 1 else confidence
            )

        if row["league"]:
            leagues.add(str(row["league"]))
        if row["bookmaker"]:
            bookmakers.add(str(row["bookmaker"]))

    settled = wins + losses
    win_rate = wins / settled if settled else None
    yield_value = profit / stake_sum if stake_sum else None

    maturity = (
        "READY"
        if settled >= 150
        else "DEVELOPING"
        if settled >= 50
        else "EARLY"
        if settled > 0
        else "EMPTY"
    )

    quality = round(
        (
            min(settled / 150, 1.0)
            + (1.0 if probability_samples else 0.0)
            + (1.0 if stake_sum else 0.0)
            + min(len(leagues) / 5, 1.0)
        )
        / 4,
        4,
    )

    return {
        "total_bets": len(rows),
        "settled_bets": settled,
        "open_bets": opened,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit": round(profit, 4),
        "stake_sum": round(stake_sum, 4),
        "yield": round(yield_value, 4) if yield_value is not None else None,
        "probability_samples": probability_samples,
        "brier_score": (
            round(brier / probability_samples, 6)
            if probability_samples else None
        ),
        "log_loss": (
            round(log_loss / probability_samples, 6)
            if probability_samples else None
        ),
        "average_confidence": (
            round(sum(confidences) / len(confidences), 4)
            if confidences else None
        ),
        "leagues": len(leagues),
        "bookmakers": len(bookmakers),
        "data_quality": quality,
        "maturity": maturity,
        "ai_health": (
            "GOOD"
            if maturity == "READY" and quality >= 0.75
            else "BUILDING"
            if settled
            else "EMPTY"
        ),
        "status": "READY",
    }


def adaptive_weights(sports: dict[str, dict[str, Any]]) -> dict[str, float]:
    raw: dict[str, float] = {}

    for sport, metrics in sports.items():
        settled = float(metrics.get("settled_bets") or 0)
        quality = float(metrics.get("data_quality") or 0)
        yield_value = float(metrics.get("yield") or 0)

        score = (
            min(settled / 200, 1.0) * 0.45
            + quality * 0.35
            + (max(min(yield_value, 0.25), -0.25) + 0.25) * 0.20
        )
        raw[sport] = max(score, 0.01)

    total = sum(raw.values())
    return {
        sport: round(score / total, 4)
        for sport, score in raw.items()
    } if total else {}


def export_dashboard(
    result: dict[str, Any],
    export_dir: Path,
) -> dict[str, str]:
    export_dir.mkdir(parents=True, exist_ok=True)

    json_path = export_dir / "multisport_learning_v2_1_report.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = [
        {"sport": sport, **metrics}
        for sport, metrics in result["sports"].items()
    ]
    fieldnames = sorted({key for row in rows for key in row})

    csv_path = export_dir / "multisport_learning_v2_1_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(name, '')))}</td>"
            for name in fieldnames
        )
        + "</tr>"
        for row in rows
    )

    html_path = export_dir / "multisport_learning_v2_1_dashboard.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="sk">
<head><meta charset="utf-8"><title>Multisport Learning V2.1</title></head>
<body>
<h1>Multisport Learning V2.1</h1>
<p>Status: {html.escape(str(result["status"]))}</p>
<table border="1" cellspacing="0" cellpadding="6">
<thead><tr>{headers}</tr></thead>
<tbody>{body}</tbody>
</table>
</body>
</html>""",
        encoding="utf-8",
    )

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        default=os.getenv("DB_FILE", "bets.db"),
    )
    parser.add_argument(
        "--history-csv",
        default=os.getenv(
            "MULTISPORT_V2_HISTORY_CSV",
            "exports/history_sport_bets.csv",
        ),
    )
    parser.add_argument(
        "--export-dir",
        default=os.getenv("EXPORT_DIR", "exports"),
    )
    args = parser.parse_args()

    database = Path(args.database)
    history_csv = Path(args.history_csv)
    export_dir = Path(args.export_dir)

    print("=== MULTISPORT LEARNING V2.1 STANDALONE ===")
    pprint(ensure_schema(database))
    pprint(restore_history(database, history_csv))

    sports = {
        sport: sport_metrics(database, sport)
        for sport in SUPPORTED_SPORTS
    }

    result = {
        "version": "V2.1-standalone",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sports": sports,
        "adaptive_weights": adaptive_weights(sports),
        "sports_completed": len(sports),
        "sports_ready": len(sports),
        "errors": [],
        "status": "READY",
    }
    result["artifacts"] = export_dashboard(result, export_dir)

    pprint(result)


if __name__ == "__main__":
    main()
