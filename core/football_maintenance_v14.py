from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


DEFAULT_REPORT_PATH = "exports/football_model_health_v14.json"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


@dataclass
class FootballMaintenanceSummary:
    deleted_market_snapshots: int
    deleted_diagnostics_rows: int
    settled_samples: int
    open_samples: int
    wins: int
    losses: int
    accuracy: float
    brier_score: float
    average_clv: float
    report_path: str


class FootballMaintenanceV14:
    def __init__(
        self,
        settings: Settings,
        *,
        snapshot_retention_days: int = 45,
        diagnostics_retention_days: int = 14,
        report_path: str = DEFAULT_REPORT_PATH,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.snapshot_retention_days = max(
            7,
            min(365, int(snapshot_retention_days)),
        )
        self.diagnostics_retention_days = max(
            3,
            min(90, int(diagnostics_retention_days)),
        )
        self.report_path = Path(report_path)

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
        return conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone() is not None

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

    def cleanup_old_rows(self) -> tuple[int, int]:
        market_deleted = 0
        diagnostics_deleted = 0

        market_cutoff = (
            now_utc() - timedelta(days=self.snapshot_retention_days)
        ).isoformat(timespec="seconds")
        diagnostics_cutoff = (
            now_utc() - timedelta(days=self.diagnostics_retention_days)
        ).isoformat(timespec="seconds")

        with self.connect() as conn:
            if self._table_exists(
                conn,
                "football_market_snapshots_v14",
            ):
                before = conn.total_changes
                conn.execute(
                    """
                    DELETE FROM football_market_snapshots_v14
                    WHERE captured_at < ?
                      AND is_closing_window = 0
                    """,
                    (market_cutoff,),
                )
                market_deleted = conn.total_changes - before

            if self._table_exists(
                conn,
                "football_settlement_audit",
            ):
                columns = self._columns(
                    conn,
                    "football_settlement_audit",
                )
                timestamp_column = (
                    "created_at"
                    if "created_at" in columns
                    else "settled_at"
                    if "settled_at" in columns
                    else ""
                )

                if timestamp_column:
                    before = conn.total_changes
                    conn.execute(
                        f"""
                        DELETE FROM football_settlement_audit
                        WHERE {timestamp_column} < ?
                        """,
                        (diagnostics_cutoff,),
                    )
                    diagnostics_deleted = (
                        conn.total_changes - before
                    )

            conn.commit()

        return market_deleted, diagnostics_deleted

    def build_health_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "generated_at": iso_now(),
            "version": "v14",
            "settled_samples": 0,
            "open_samples": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": 0.0,
            "brier_score": 0.0,
            "average_clv": 0.0,
            "meta_model_exists": Path(
                "models/football_meta_model.pkl"
            ).exists(),
        }

        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_feature_history",
            ):
                return report

            columns = self._columns(
                conn,
                "football_feature_history",
            )
            rows = conn.execute(
                "SELECT * FROM football_feature_history"
            ).fetchall()

        settled: list[dict[str, float]] = []
        open_count = 0

        for row in rows:
            result = str(row["result"] or "OPEN").upper()

            if result == "OPEN":
                open_count += 1
                continue

            if result not in {"WON", "LOST"}:
                continue

            probability = safe_float(
                row["final_probability"]
                if "final_probability" in columns
                else row["model_probability"]
                if "model_probability" in columns
                else 0.5,
                0.5,
            )
            actual = 1.0 if result == "WON" else 0.0
            clv = safe_float(
                row["clv"] if "clv" in columns else 0.0,
                0.0,
            )

            settled.append(
                {
                    "actual": actual,
                    "probability": min(
                        0.999,
                        max(0.001, probability),
                    ),
                    "clv": clv,
                }
            )

        wins = sum(
            int(item["actual"] == 1.0)
            for item in settled
        )
        losses = len(settled) - wins

        if settled:
            accuracy = sum(
                int(
                    (item["probability"] >= 0.5)
                    == bool(item["actual"])
                )
                for item in settled
            ) / len(settled)

            brier_score = sum(
                (
                    item["probability"]
                    - item["actual"]
                ) ** 2
                for item in settled
            ) / len(settled)

            average_clv = sum(
                item["clv"] for item in settled
            ) / len(settled)
        else:
            accuracy = 0.0
            brier_score = 0.0
            average_clv = 0.0

        report.update(
            {
                "settled_samples": len(settled),
                "open_samples": open_count,
                "wins": wins,
                "losses": losses,
                "accuracy": accuracy,
                "brier_score": brier_score,
                "average_clv": average_clv,
            }
        )

        return report

    def write_report(
        self,
        report: dict[str, Any],
    ) -> None:
        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def run(self) -> FootballMaintenanceSummary:
        deleted_market, deleted_diagnostics = (
            self.cleanup_old_rows()
        )
        report = self.build_health_report()
        self.write_report(report)

        return FootballMaintenanceSummary(
            deleted_market_snapshots=deleted_market,
            deleted_diagnostics_rows=deleted_diagnostics,
            settled_samples=int(report["settled_samples"]),
            open_samples=int(report["open_samples"]),
            wins=int(report["wins"]),
            losses=int(report["losses"]),
            accuracy=float(report["accuracy"]),
            brier_score=float(report["brier_score"]),
            average_clv=float(report["average_clv"]),
            report_path=str(self.report_path),
        )


def run_football_maintenance_v14(
    settings: Settings,
    *,
    snapshot_retention_days: int = 45,
    diagnostics_retention_days: int = 14,
) -> FootballMaintenanceSummary:
    return FootballMaintenanceV14(
        settings,
        snapshot_retention_days=snapshot_retention_days,
        diagnostics_retention_days=diagnostics_retention_days,
    ).run()


if __name__ == "__main__":
    import os

    settings = Settings.from_env()
    summary = run_football_maintenance_v14(
        settings,
        snapshot_retention_days=int(
            os.getenv(
                "FOOTBALL_SNAPSHOT_RETENTION_DAYS",
                "45",
            )
        ),
        diagnostics_retention_days=int(
            os.getenv(
                "FOOTBALL_DIAGNOSTICS_RETENTION_DAYS",
                "14",
            )
        ),
    )

    print(
        "Football Maintenance v14: "
        f"deleted_market={summary.deleted_market_snapshots}, "
        f"deleted_diagnostics={summary.deleted_diagnostics_rows}, "
        f"settled={summary.settled_samples}, "
        f"open={summary.open_samples}, "
        f"accuracy={summary.accuracy:.3f}, "
        f"brier={summary.brier_score:.4f}, "
        f"avg_clv={summary.average_clv:.4f}"
    )
