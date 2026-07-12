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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class FootballMaintenanceSummary:
    deleted_market_snapshots: int
    settled_samples: int
    valid_probability_samples: int
    open_samples: int
    wins: int
    losses: int
    hit_rate: float | None
    threshold_accuracy: float | None
    brier_score: float | None
    log_loss: float | None
    average_clv: float | None
    closing_snapshots: int
    report_path: str


class FootballMaintenanceV14:
    """
    Safe maintenance and model-health reporting.

    Important:
    - settlement audit history is never deleted;
    - only old non-closing market snapshots are pruned;
    - probability metrics are calculated only from real stored
      probabilities, never from an artificial 0.50 fallback.
    """

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

        # Kept for backward compatibility with main.py and environment.
        # Audit learning data is intentionally never removed.
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

    def cleanup_old_rows(self) -> int:
        """
        Delete only old, non-closing market snapshots.

        Closing snapshots and all settlement/learning audit rows are
        preserved because they are valuable training history.
        """
        cutoff = (
            now_utc()
            - timedelta(days=self.snapshot_retention_days)
        ).isoformat(timespec="seconds")

        with self.connect() as conn:
            if not self._table_exists(
                conn,
                "football_market_snapshots_v14",
            ):
                return 0

            before = conn.total_changes

            conn.execute(
                """
                DELETE FROM football_market_snapshots_v14
                WHERE captured_at < ?
                  AND is_closing_window = 0
                """,
                (cutoff,),
            )

            deleted = conn.total_changes - before
            conn.commit()

        return deleted

    @staticmethod
    def _probability_column(
        columns: set[str],
    ) -> str:
        for candidate in (
            "final_probability",
            "model_probability",
            "model_consensus_probability",
            "consensus_probability",
            "predicted_probability",
        ):
            if candidate in columns:
                return candidate

        return ""

    @staticmethod
    def _calibration_bins(
        samples: list[dict[str, float]],
    ) -> list[dict[str, float | int]]:
        bins: list[dict[str, float | int]] = []

        for lower in (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ):
            upper = lower + 0.1

            selected = [
                item
                for item in samples
                if lower <= item["probability"] < upper
                or (
                    upper >= 1.0
                    and item["probability"] == 1.0
                )
            ]

            if not selected:
                continue

            bins.append(
                {
                    "lower": lower,
                    "upper": min(1.0, upper),
                    "samples": len(selected),
                    "mean_probability": sum(
                        item["probability"]
                        for item in selected
                    )
                    / len(selected),
                    "actual_win_rate": sum(
                        item["actual"]
                        for item in selected
                    )
                    / len(selected),
                }
            )

        return bins

    def _snapshot_stats(
        self,
        conn: sqlite3.Connection,
    ) -> dict[str, int]:
        if not self._table_exists(
            conn,
            "football_market_snapshots_v14",
        ):
            return {
                "market_snapshots": 0,
                "closing_snapshots": 0,
                "events_with_closing_snapshot": 0,
            }

        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(
                    CASE WHEN is_closing_window = 1
                    THEN 1 ELSE 0 END
                ) AS closing_total,
                COUNT(
                    DISTINCT CASE
                        WHEN is_closing_window = 1
                        THEN sport_key || '|' || commence_time || '|' || event
                    END
                ) AS closing_events
            FROM football_market_snapshots_v14
            """
        ).fetchone()

        return {
            "market_snapshots": int(row["total"] or 0),
            "closing_snapshots": int(
                row["closing_total"] or 0
            ),
            "events_with_closing_snapshot": int(
                row["closing_events"] or 0
            ),
        }

    def build_health_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "generated_at": iso_now(),
            "version": "v14.1",
            "settled_samples": 0,
            "valid_probability_samples": 0,
            "open_samples": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": None,
            "threshold_accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "average_clv": None,
            "meta_model_exists": Path(
                "models/football_meta_model.pkl"
            ).exists(),
            "metrics_reliable": False,
            "warnings": [],
            "calibration_bins": [],
            "market_snapshots": 0,
            "closing_snapshots": 0,
            "events_with_closing_snapshot": 0,
        }

        with self.connect() as conn:
            report.update(self._snapshot_stats(conn))

            if not self._table_exists(
                conn,
                "football_feature_history",
            ):
                report["warnings"].append(
                    "football_feature_history table does not exist"
                )
                return report

            columns = self._columns(
                conn,
                "football_feature_history",
            )
            probability_column = self._probability_column(
                columns
            )
            rows = conn.execute(
                "SELECT * FROM football_feature_history"
            ).fetchall()

        settled_total = 0
        open_count = 0
        wins = 0
        losses = 0

        probability_samples: list[dict[str, float]] = []
        clv_values: list[float] = []

        for row in rows:
            result = str(
                row["result"] or "OPEN"
            ).upper()

            if result == "OPEN":
                open_count += 1
                continue

            if result not in {"WON", "LOST"}:
                continue

            settled_total += 1
            actual = 1.0 if result == "WON" else 0.0

            if actual == 1.0:
                wins += 1
            else:
                losses += 1

            if probability_column:
                probability = safe_float(
                    row[probability_column],
                    None,
                )

                if probability is not None:
                    probability_samples.append(
                        {
                            "actual": actual,
                            "probability": clamp(
                                probability,
                                0.001,
                                0.999,
                            ),
                        }
                    )

            if "clv" in columns:
                clv = safe_float(row["clv"], None)

                if clv is not None:
                    clv_values.append(clv)

        report.update(
            {
                "settled_samples": settled_total,
                "valid_probability_samples": len(
                    probability_samples
                ),
                "open_samples": open_count,
                "wins": wins,
                "losses": losses,
            }
        )

        if probability_samples:
            hit_rate = sum(
                item["actual"]
                for item in probability_samples
            ) / len(probability_samples)

            # Kept only as a diagnostic. For betting selections, many valid
            # picks have probability below 0.50, so this is not the primary
            # quality metric.
            threshold_accuracy = sum(
                int(
                    (
                        item["probability"] >= 0.5
                    )
                    == bool(item["actual"])
                )
                for item in probability_samples
            ) / len(probability_samples)

            brier_score = sum(
                (
                    item["probability"]
                    - item["actual"]
                ) ** 2
                for item in probability_samples
            ) / len(probability_samples)

            epsilon = 1e-12
            log_loss = -sum(
                item["actual"]
                * math.log(
                    max(
                        epsilon,
                        item["probability"],
                    )
                )
                + (1.0 - item["actual"])
                * math.log(
                    max(
                        epsilon,
                        1.0 - item["probability"],
                    )
                )
                for item in probability_samples
            ) / len(probability_samples)

            report.update(
                {
                    "hit_rate": hit_rate,
                    "threshold_accuracy": threshold_accuracy,
                    "brier_score": brier_score,
                    "log_loss": log_loss,
                    "calibration_bins": (
                        self._calibration_bins(
                            probability_samples
                        )
                    ),
                }
            )
        else:
            report["warnings"].append(
                "No valid stored prediction probabilities "
                "were found for settled samples"
            )

        if clv_values:
            report["average_clv"] = (
                sum(clv_values) / len(clv_values)
            )
        else:
            report["warnings"].append(
                "No valid CLV values are available yet"
            )

        # With fewer than 20 settled probability samples, metrics are
        # descriptive only and must not be treated as reliable.
        report["metrics_reliable"] = (
            len(probability_samples) >= 20
            and wins >= 5
            and losses >= 5
        )

        if not report["metrics_reliable"]:
            report["warnings"].append(
                "Model-health metrics are not statistically reliable yet"
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
        deleted_market = self.cleanup_old_rows()
        report = self.build_health_report()
        self.write_report(report)

        return FootballMaintenanceSummary(
            deleted_market_snapshots=deleted_market,
            settled_samples=int(
                report["settled_samples"]
            ),
            valid_probability_samples=int(
                report["valid_probability_samples"]
            ),
            open_samples=int(report["open_samples"]),
            wins=int(report["wins"]),
            losses=int(report["losses"]),
            hit_rate=report["hit_rate"],
            threshold_accuracy=report["threshold_accuracy"],
            brier_score=report["brier_score"],
            log_loss=report["log_loss"],
            average_clv=report["average_clv"],
            closing_snapshots=int(
                report["closing_snapshots"]
            ),
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

    def format_metric(
        value: float | None,
        digits: int,
    ) -> str:
        if value is None:
            return "n/a"

        return f"{value:.{digits}f}"

    print(
        "Football Maintenance v14.1: "
        f"deleted_market={summary.deleted_market_snapshots}, "
        f"settled={summary.settled_samples}, "
        f"valid_probabilities="
        f"{summary.valid_probability_samples}, "
        f"open={summary.open_samples}, "
        f"closing_snapshots={summary.closing_snapshots}, "
        f"hit_rate={format_metric(summary.hit_rate, 3)}, "
        f"threshold_accuracy="
        f"{format_metric(summary.threshold_accuracy, 3)}, "
        f"brier={format_metric(summary.brier_score, 4)}, "
        f"log_loss={format_metric(summary.log_loss, 4)}, "
        f"avg_clv={format_metric(summary.average_clv, 4)}"
    )
