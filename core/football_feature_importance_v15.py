from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


FEATURES = {
    "consensus_safety": "Consensus safety",
    "raw_edge": "Raw edge",
    "market_model_gap": "Market-model gap",
    "elo_difference": "ELO separation",
    "form_difference": "Form separation",
    "competition_importance": "Competition importance",
    "context_reliability": "Context reliability",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class FootballFeatureImportanceV15:
    """
    Analytical layer only.

    Does not modify prediction weights.
    Works with partial Dataset V15 schemas.
    """

    def __init__(
        self,
        db_file: str = "bets.db",
    ) -> None:
        self.db_file = Path(db_file)
        self.missing_features: list[str] = []
        self.used_features: list[str] = []

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def load_rows(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type='table'
                AND name='football_dataset_v15'
                """
            ).fetchone()

            if exists is None:
                return []

            return conn.execute(
                """
                SELECT *
                FROM football_dataset_v15
                WHERE training_ready = 1
                """
            ).fetchall()

    def analyse_feature(
        self,
        rows: list[sqlite3.Row],
        column: str,
        label: str,
    ) -> dict[str, Any]:

        values: list[float] = []
        wins = 0
        losses = 0

        for row in rows:
            if column not in row.keys():
                continue

            value = _safe_float(row[column])

            if value is None:
                continue

            values.append(value)

            if int(row["target"]) == 1:
                wins += 1
            else:
                losses += 1

        samples = wins + losses

        return {
            "feature": label,
            "column": column,
            "samples": samples,
            "wins": wins,
            "losses": losses,
            "winrate": (
                round(wins / samples, 4)
                if samples
                else None
            ),
            "average_value": (
                round(sum(values) / len(values), 4)
                if values
                else None
            ),
        }

    def build_report(self) -> dict[str, Any]:
        rows = self.load_rows()

        available_columns = (
            set(rows[0].keys())
            if rows
            else set()
        )

        ranking = []

        for column, label in FEATURES.items():
            if column not in available_columns:
                self.missing_features.append(column)
                continue

            self.used_features.append(column)

            ranking.append(
                self.analyse_feature(
                    rows,
                    column,
                    label,
                )
            )

        ranking.sort(
            key=lambda item: (
                -(item["winrate"] or 0),
                -item["samples"],
            )
        )

        return {
            "version": "v15",
            "training_samples": len(rows),
            "features_used": self.used_features,
            "missing_features": self.missing_features,
            "feature_ranking": ranking,
            "warning": (
                "Small sample size - information only"
                if len(rows) < 50
                else None
            ),
            "weight_update_allowed": (
                len(rows) >= 100
            ),
        }

    def export(
        self,
        report: dict[str, Any],
    ) -> None:

        export_dir = Path("exports")
        export_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            export_dir /
            "football_feature_importance_v15.json"
        )

        csv_path = (
            export_dir /
            "football_feature_importance_v15.csv"
        )

        json_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            fields = [
                "feature",
                "column",
                "samples",
                "wins",
                "losses",
                "winrate",
                "average_value",
            ]

            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                report["feature_ranking"]
            )

    def run(self) -> dict[str, Any]:
        report = self.build_report()
        self.export(report)
        return report


def run_feature_importance_v15(
    db_file: str = "bets.db",
) -> dict[str, Any]:
    return FootballFeatureImportanceV15(
        db_file
    ).run()
