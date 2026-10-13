from __future__ import annotations

import sqlite3

from core.config import Settings
from core.model_stats import save_model_stats, load_model_stats


SETTLED_RESULTS = ("WON", "LOST", "V", "P")
WIN_RESULTS = ("WON", "V")
LOSS_RESULTS = ("LOST", "P")


def _live_clause(cur: sqlite3.Cursor) -> str:
    columns = {str(row[1]) for row in cur.execute("PRAGMA table_info(sport_bets)")}
    return (
        " AND COALESCE(NULLIF(TRIM(tracking_mode), ''), 'LIVE')='LIVE'"
        if "tracking_mode" in columns else ""
    )


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
        {_live_clause(cur)}
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
        live_clause = _live_clause(cur)

        total = int(
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                {live_clause}
                """
            ).fetchone()[0]
        )

        wins = int(
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('WON','V')
                {live_clause}
                """
            ).fetchone()[0]
        )

        losses = int(
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM sport_bets
                WHERE result IN ('LOST','P')
                {live_clause}
                """
            ).fetchone()[0]
        )

        profit = float(
            cur.execute(
                f"""
                SELECT COALESCE(SUM(profit),0)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                {live_clause}
                """
            ).fetchone()[0]
        )

        stake_sum = float(
            cur.execute(
                f"""
                SELECT COALESCE(SUM(stake),0)
                FROM sport_bets
                WHERE result IN ('WON','LOST','V','P')
                {live_clause}
                """
            ).fetchone()[0]
        )

        open_bets = int(
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM sport_bets
                WHERE (result IS NULL OR result='' OR result='OPEN')
                {live_clause}
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


def _append_top_group(
    text: str,
    title: str,
    data: dict,
    limit: int = 10,
) -> str:
    if not data:
        return text

    text += f"\n{title}:\n"

    items = sorted(
        data.items(),
        key=lambda x: x[1].get("profit", 0),
        reverse=True,
    )[:limit]

    for name, row in items:
        text += (
            f"- {name}: "
            f"{row['wins']}-{row['losses']} | "
            f"profit {row['profit']:.2f} | "
            f"yield {row['yield']:.2f}% | "
            f"bets {row['total']}\n"
        )

    return text


def performance_report(settings: Settings) -> str:
    stats = update_model_stats(settings)

    conn = sqlite3.connect(settings.db_file)

    try:
        cur = conn.cursor()
        live_clause = _live_clause(cur)
        rows = conn.execute(
            f"""
            SELECT
                CASE
                    WHEN result IS NULL OR result='' OR result='OPEN'
                    THEN 'OPEN'
                    ELSE result
                END AS result_group,
                COUNT(*)
            FROM sport_bets
            WHERE 1=1 {live_clause}
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

    text = _append_top_group(text, "By sport", stats.get("by_sport", {}))
    text = _append_top_group(text, "Top bookmakers", stats.get("by_bookmaker", {}))
    text = _append_top_group(text, "Top leagues", stats.get("by_league", {}))

    return text
