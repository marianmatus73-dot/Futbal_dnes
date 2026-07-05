from __future__ import annotations

import sqlite3

from core.config import Settings
from core.model_stats import save_model_stats, load_model_stats


def update_model_stats(settings: Settings) -> dict:
    conn = sqlite3.connect(settings.db_file)

    try:
        cur = conn.cursor()

        total = cur.execute(
            """
            SELECT COUNT(*)
            FROM sport_bets
            WHERE result IN ('WON','LOST','V','P')
            """
        ).fetchone()[0]

        wins = cur.execute(
            """
            SELECT COUNT(*)
            FROM sport_bets
            WHERE result IN ('WON','V')
            """
        ).fetchone()[0]

        losses = cur.execute(
            """
            SELECT COUNT(*)
            FROM sport_bets
            WHERE result IN ('LOST','P')
            """
        ).fetchone()[0]

        profit = 0.0

        try:
            profit = float(
                cur.execute(
                    """
                    SELECT COALESCE(SUM(profit),0)
                    FROM sport_bets
                    """
                ).fetchone()[0]
            )
        except Exception:
            pass

        stake_sum = cur.execute(
    """
    SELECT COALESCE(SUM(stake),0)
    FROM sport_bets
    WHERE result IN ('WON','LOST')
    """
).fetchone()[0]

yield_pct = (
    (profit / stake_sum) * 100
    if stake_sum > 0
    else 0.0
)

        save_model_stats(
            total_bets=total,
            wins=wins,
            losses=losses,
            profit=profit,
            yield_pct=yield_pct,
        )

        return load_model_stats()

    finally:
        conn.close()


def performance_report(settings: Settings) -> str:
    stats = update_model_stats(settings)

    conn = sqlite3.connect(settings.db_file)

    try:
        rows = conn.execute(
            """
            SELECT COALESCE(result, ''), COUNT(*)
            FROM sport_bets
            GROUP BY COALESCE(result, '')
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    finally:
        conn.close()

    open_bets = 0
    settled_bets = 0

    for result, count in rows:
        if result:
            settled_bets += count
        else:
            open_bets += count

    text = (
        "\n=== MODEL PERFORMANCE ===\n"
        f"Total bets: {stats['total_bets']}\n"
        f"Wins: {stats['wins']}\n"
        f"Losses: {stats['losses']}\n"
        f"Winrate: {stats['winrate']:.2f}%\n"
        f"Yield: {stats['yield']:.2f}%\n"
        f"Profit: {stats['profit']:.2f}\n"
    )

    text += f"\nOpen bets: {open_bets}\n"
    text += f"Settled bets: {settled_bets}\n"

    text += "\nResult distribution:\n"

    for result, count in rows:
        label = result if result else "OPEN"
        text += f"- {label}: {count}\n"

    return text
