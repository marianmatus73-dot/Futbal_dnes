from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.config import Settings


@dataclass
class FootballPipelineMetrics:
    matches: int = 0
    resolved_matches: int = 0
    keys_created: int = 0
    joins_completed: int = 0
    closing_written: int = 0
    clv_ready: int = 0
    explainability_rows: int = 0


def load_football_pipeline_metrics(settings: Settings) -> FootballPipelineMetrics:
    """Measure V15 pipeline stages from one consistent dataset population."""
    db_file = Path(settings.db_file or "bets.db")
    if not db_file.exists():
        return FootballPipelineMetrics()

    with sqlite3.connect(db_file) as conn:
        table_names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        metrics = FootballPipelineMetrics()

        if "football_dataset_v15" in table_names:
            columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(football_dataset_v15)"
                ).fetchall()
            }
            metrics.matches = int(conn.execute(
                "SELECT COUNT(*) FROM football_dataset_v15"
            ).fetchone()[0] or 0)

            required_identity = {
                "source_hash", "sport_key", "event", "commence_time"
            }
            if required_identity.issubset(columns):
                metrics.resolved_matches = int(conn.execute(
                    """
                    SELECT COUNT(*) FROM football_dataset_v15
                    WHERE TRIM(COALESCE(sport_key, '')) <> ''
                      AND TRIM(COALESCE(event, '')) <> ''
                      AND TRIM(COALESCE(commence_time, '')) <> ''
                    """
                ).fetchone()[0] or 0)
                metrics.keys_created = int(conn.execute(
                    """
                    SELECT COUNT(*) FROM football_dataset_v15
                    WHERE TRIM(COALESCE(source_hash, '')) <> ''
                    """
                ).fetchone()[0] or 0)

            if "has_closing_line" in columns:
                metrics.joins_completed = int(conn.execute(
                    "SELECT COUNT(*) FROM football_dataset_v15 "
                    "WHERE has_closing_line=1"
                ).fetchone()[0] or 0)
                metrics.closing_written = metrics.joins_completed

            clv_columns = [
                name for name in ("clv_probability", "clv_odds_ratio")
                if name in columns
            ]
            if clv_columns:
                predicate = " OR ".join(
                    f"{name} IS NOT NULL" for name in clv_columns
                )
                metrics.clv_ready = int(conn.execute(
                    f"SELECT COUNT(*) FROM football_dataset_v15 WHERE {predicate}"
                ).fetchone()[0] or 0)

        if "football_explainability_v15" in table_names:
            metrics.explainability_rows = int(conn.execute(
                "SELECT COUNT(*) FROM football_explainability_v15"
            ).fetchone()[0] or 0)

    return metrics

