from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_JSON_PATH = "exports/football_evaluation_dashboard_v15.json"
DEFAULT_CSV_PATH = "exports/football_evaluation_dashboard_v15.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


@dataclass
class FootballEvaluationSummary:
    total_rows: int
    training_ready_rows: int
    wins: int
    losses: int
    hit_rate: float | None
    brier_score: float | None
    log_loss: float | None
    average_clv_probability: float | None
    average_consensus_safety: float | None
    json_path: str
    csv_path: str


class FootballEvaluationDashboardV15:
    def __init__(
        self,
        settings: Settings,
        *,
        json_path: str = DEFAULT_JSON_PATH,
        csv_path: str = DEFAULT_CSV_PATH,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.json_path = Path(json_path)
        self.csv_path = Path(csv_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _table_exists(
        conn: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        return conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone() is not None

    @staticmethod
    def _metrics(
        rows: list[sqlite3.Row],
    ) -> dict[str, float | None]:
        valid = []

        for row in rows:
            probability = safe_float(
                row["model_probability"],
                None,
            )

            if probability is None:
                continue

            probability = min(
                0.999,
                max(0.001, probability),
            )

            valid.append(
                (
                    probability,
                    int(row["target"]),
                )
            )

        if not valid:
            return {
                "hit_rate": None,
                "brier_score": None,
                "log_loss": None,
            }

        hit_rate = sum(
            target for _, target in valid
        ) / len(valid)

        brier = sum(
            (probability - target) ** 2
            for probability, target in valid
        ) / len(valid)

        log_loss = -sum(
            target * math.log(probability)
            + (1 - target) * math.log(
                1.0 - probability
            )
            for probability, target in valid
        ) / len(valid)

        return {
            "hit_rate": hit_rate,
            "brier_score": brier,
            "log_loss": log_loss,
        }

    @staticmethod
    def _bucket(
        value: float | None,
        boundaries: list[float],
        labels: list[str],
    ) -> str:
        if value is None:
            return "missing"

        for index, boundary in enumerate(boundaries):
            if value < boundary:
                return labels[index]

        return labels[-1]

    @staticmethod
    def _aggregate(
        rows: list[sqlite3.Row],
        key_name: str,
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[sqlite3.Row]] = {}

        for row in rows:
            key = str(row[key_name] or "UNKNOWN")
            groups.setdefault(key, []).append(row)

        result = []

        for key, selected in groups.items():
            metrics = FootballEvaluationDashboardV15._metrics(
                selected
            )

            clv_values = [
                value
                for value in (
                    safe_float(
                        row["clv_probability"],
                        None,
                    )
                    for row in selected
                )
                if value is not None
            ]

            result.append(
                {
                    key_name: key,
                    "samples": len(selected),
                    "wins": sum(
                        int(row["target"])
                        for row in selected
                    ),
                    "losses": sum(
                        1 - int(row["target"])
                        for row in selected
                    ),
                    **metrics,
                    "average_clv_probability": (
                        sum(clv_values) / len(clv_values)
                        if clv_values
                        else None
                    ),
                }
            )

        result.sort(
            key=lambda item: (
                -int(item["samples"]),
                str(item[key_name]),
            )
        )

        return result

    def build_report(self) -> dict[str, Any]:
        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_dataset_v15",
            ):
                return {
                    "generated_at": now_utc(),
                    "version": "v15",
                    "warning": (
                        "football_dataset_v15 table does not exist"
                    ),
                    "total_rows": 0,
                    "training_ready_rows": 0,
                }

            rows = conn.execute(
                """
                SELECT *
                FROM football_dataset_v15
                ORDER BY commence_time
                """
            ).fetchall()

        training_rows = [
            row
            for row in rows
            if int(row["training_ready"] or 0) == 1
        ]

        metrics = self._metrics(training_rows)

        clv_values = [
            value
            for value in (
                safe_float(
                    row["clv_probability"],
                    None,
                )
                for row in training_rows
            )
            if value is not None
        ]

        safety_values = [
            value
            for value in (
                safe_float(
                    row["consensus_safety"],
                    None,
                )
                for row in training_rows
            )
            if value is not None
        ]

        confidence_buckets: dict[str, list[sqlite3.Row]] = {}
        safety_buckets: dict[str, list[sqlite3.Row]] = {}
        probability_buckets: dict[str, list[sqlite3.Row]] = {}

        for row in training_rows:
            confidence_bucket = self._bucket(
                safe_float(row["confidence"], None),
                [55.0, 65.0, 75.0, 85.0],
                ["<55", "55-64", "65-74", "75-84", "85+"],
            )
            confidence_buckets.setdefault(
                confidence_bucket,
                [],
            ).append(row)

            safety_bucket = self._bucket(
                safe_float(
                    row["consensus_safety"],
                    None,
                ),
                [0.50, 0.70, 0.85, 0.95],
                ["<0.50", "0.50-0.69", "0.70-0.84", "0.85-0.94", "0.95+"],
            )
            safety_buckets.setdefault(
                safety_bucket,
                [],
            ).append(row)

            probability_bucket = self._bucket(
                safe_float(
                    row["model_probability"],
                    None,
                ),
                [0.40, 0.50, 0.60, 0.70, 0.80],
                ["<0.40", "0.40-0.49", "0.50-0.59", "0.60-0.69", "0.70-0.79", "0.80+"],
            )
            probability_buckets.setdefault(
                probability_bucket,
                [],
            ).append(row)

        def bucket_report(
            groups: dict[str, list[sqlite3.Row]],
        ) -> list[dict[str, Any]]:
            output = []

            for label, selected in groups.items():
                output.append(
                    {
                        "bucket": label,
                        "samples": len(selected),
                        "wins": sum(
                            int(row["target"])
                            for row in selected
                        ),
                        "losses": sum(
                            1 - int(row["target"])
                            for row in selected
                        ),
                        **self._metrics(selected),
                    }
                )

            return sorted(
                output,
                key=lambda item: item["bucket"],
            )

        report = {
            "generated_at": now_utc(),
            "version": "v15",
            "total_rows": len(rows),
            "training_ready_rows": len(training_rows),
            "wins": sum(
                int(row["target"])
                for row in training_rows
            ),
            "losses": sum(
                1 - int(row["target"])
                for row in training_rows
            ),
            **metrics,
            "average_clv_probability": (
                sum(clv_values) / len(clv_values)
                if clv_values
                else None
            ),
            "average_consensus_safety": (
                sum(safety_values) / len(safety_values)
                if safety_values
                else None
            ),
            "by_league": self._aggregate(
                training_rows,
                "league",
            ),
            "by_bookmaker": self._aggregate(
                training_rows,
                "bookmaker",
            ),
            "by_confidence": bucket_report(
                confidence_buckets
            ),
            "by_consensus_safety": bucket_report(
                safety_buckets
            ),
            "calibration": bucket_report(
                probability_buckets
            ),
            "metrics_reliable": (
                len(training_rows) >= 20
                and sum(
                    int(row["target"])
                    for row in training_rows
                ) >= 5
                and sum(
                    1 - int(row["target"])
                    for row in training_rows
                ) >= 5
            ),
        }

        return report

    def write_outputs(
        self,
        report: dict[str, Any],
    ) -> None:
        self.json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.json_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        rows = []

        for section in (
            "by_league",
            "by_bookmaker",
            "by_confidence",
            "by_consensus_safety",
            "calibration",
        ):
            for item in report.get(section, []):
                rows.append(
                    {
                        "section": section,
                        **item,
                    }
                )

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = sorted(
            {
                key
                for row in rows
                for key in row.keys()
            }
        )

        with self.csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

    def run(self) -> FootballEvaluationSummary:
        report = self.build_report()
        self.write_outputs(report)

        return FootballEvaluationSummary(
            total_rows=int(
                report.get("total_rows", 0)
            ),
            training_ready_rows=int(
                report.get(
                    "training_ready_rows",
                    0,
                )
            ),
            wins=int(report.get("wins", 0)),
            losses=int(report.get("losses", 0)),
            hit_rate=report.get("hit_rate"),
            brier_score=report.get("brier_score"),
            log_loss=report.get("log_loss"),
            average_clv_probability=report.get(
                "average_clv_probability"
            ),
            average_consensus_safety=report.get(
                "average_consensus_safety"
            ),
            json_path=str(self.json_path),
            csv_path=str(self.csv_path),
        )


def run_football_evaluation_dashboard_v15(
    settings: Settings,
) -> FootballEvaluationSummary:
    return FootballEvaluationDashboardV15(
        settings
    ).run()


if __name__ == "__main__":
    settings = Settings.from_env()
    summary = run_football_evaluation_dashboard_v15(
        settings
    )

    def fmt(
        value: float | None,
        digits: int = 4,
    ) -> str:
        return (
            "n/a"
            if value is None
            else f"{value:.{digits}f}"
        )

    print(
        "Football Evaluation Dashboard v15: "
        f"total={summary.total_rows}, "
        f"training_ready={summary.training_ready_rows}, "
        f"wins={summary.wins}, "
        f"losses={summary.losses}, "
        f"hit_rate={fmt(summary.hit_rate, 3)}, "
        f"brier={fmt(summary.brier_score)}, "
        f"log_loss={fmt(summary.log_loss)}, "
        f"avg_clv={fmt(summary.average_clv_probability)}, "
        f"avg_consensus_safety="
        f"{fmt(summary.average_consensus_safety, 3)}"
    )
