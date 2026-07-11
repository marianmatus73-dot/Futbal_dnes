from __future__ import annotations

import os
import sqlite3

from core.config import Settings


def cleanup_duplicates(settings: Settings) -> int:
    conn = sqlite3.connect(settings.db_file)

    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM sport_bets"
        ).fetchone()[0]

        conn.execute(
            """
            DELETE FROM sport_bets
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM sport_bets
                GROUP BY
                    sport,
                    league,
                    event,
                    market,
                    selection,
                    start_time
            )
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_sport_bets_unique_tip
            ON sport_bets (
                sport,
                league,
                event,
                market,
                selection,
                start_time
            )
            """
        )

        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) FROM sport_bets"
        ).fetchone()[0]

        removed = before - after
        print(f"Removed duplicate bets: {removed}")
        return removed

    finally:
        conn.close()


if __name__ == "__main__":
    settings = Settings.from_env()
    cleanup_duplicates(settings)
