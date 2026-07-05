from __future__ import annotations

import sqlite3

from core.config import Settings
from core.model_stats import save_model_stats, load_model_stats


SETTLED_RESULTS = ("WON", "LOST", "V", "P")
WIN_RESULTS = ("WON", "V")
LOSS_RESULTS = ("LOST", "P")


def _group_stats(cur: sqlite3.Cursor, column: str) -> dict:
    allowed = {"sport", "bookmaker", "league"}
    if column not in allowed:
        return {}

    rows = cur.execute(
        f"""
        SELECT
            COALESCE({column}, 'UNKNOWN') AS name,
            COUNT(*) AS total,
            SUM(CASE WHEN result IN ('WON','V') THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN result IN ('LOST','P') THEN 1 ELSE 0 END) AS losses,
            COALESCE(SUM(profit),0) AS profit,
            COALESCE(SUM(stake),0) AS stake_sum
        FROM sport_bets
        WHERE result IN ('WON','LOST','V','P')
        GROUP BY COALESCE({column}, 'UNKNOWN')
        ORDER BY profit DESC
        """
    ).fetchall()

    data = {}

    for name, total, wins, losses, profit, stake_sum in rows:
        total = int(total or 0)
        wins = int(wins or 0)
        losses = int(losses or 0)
        profit = float(profit or 0)
        stake_sum = float(stake_sum or 0)

        data[str(name or "UNKNOWN")] = {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": round((wins / total * 100) if total else 0.0, 2),
            "profit": round(profit, 2),
            "stake_sum": round(stake_sum, 2),
            "yield": round((profit / stake_sum * 100) if stake_sum > 0 else 0.0, 2),
        }

    return data


def update_model_stats(settings: Settings) -> dict:
    conn = sqlite3.connect(settings.db_file)

    try:
        cur = conn.cursor()

        total = int(
            cur.execute(
                """
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                """
            ).fetchone()[0]
        )

        wins = int(
            cur.execute(
                """
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('WON','V')
                """
            ).fetchone()[0]
        )

        losses = int(
            cur.execute(
                """
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('LOST','P')
                """
            ).fetchone()[0]
        )

        profit = float(
            cur.execute(
                """
                SELECT COALESCE(SUM(profit),0)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                """
            ).fetchone()[0]
        )

        stake_sum = float(
            cur.execute(
                """
                SELECT COALESCE(SUM(stake),0)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                """
            ).fetchone()[0]
        )

        open_bets = int(
            cur.execute(
                """
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IS NULL OR result='' OR result='OPEN'
                """
            ).fetchone()[0]
        )

        settled_bets = total

        yield_pct = (profit / stake_sum) * 100 if stake_sum > 0 else 0.0

        save_model_stats(
            total_bets=total,
            wins=wins,
            losses=losses,
            profit=profit,
            yield_pct=yield_pct,
            stake_sum=stake_sum,
            open_bets=open_bets,
            settled_bets=settled_bets,
            by_sport=_group_stats(cur, "sport"),
            by_bookmaker=_group_stats(cur, "bookmaker"),
            by_league=_group_stats(cur, "league"),
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
            SELECT
                CASE
                    WHEN result IS NULL OR result='' OR result='OPEN'
                    THEN 'OPEN'
                    ELSE result
                END AS result_group,
                COUNT(*)
            FROM sport_bets
            GROUP BY result_group
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    finally:
        conn.close()

    text = (
        "\n=== MODEL PERFORMANCE ===\n"
        f"Total bets: {stats['total_bets']}\n"
        f"Wins: {stats['wins']}\n"
        f"Losses: {stats['losses']}\n"
        f"Winrate: {stats['winrate']:.2f}%\n"
        f"Yield: {stats['yield']:.2f}%\n"
        f"Profit: {stats['profit']:.2f}\n"
        f"Stake sum: {stats['stake_sum']:.2f}\n"
        f"\nOpen bets: {stats['open_bets']}\n"
        f"Settled bets: {stats['settled_bets']}\n"
    )

    text += "\nResult distribution:\n"

    for result, count in rows:
        text += f"- {result}: {count}\n"

    if stats.get("by_sport"):
        text += "\nBy sport:\n"
        for sport, data in stats["by_sport"].items():
            text += (
                f"- {sport}: "
                f"{data['wins']}-{data['losses']} | "
                f"profit {data['profit']:.2f} | "
                f"yield {data['yield']:.2f}%\n"
            )

    return text
