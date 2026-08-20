"""Canonical mapping for persistent multisport history synchronization.

The production runner restores these CSV files before a scan and exports the
same tables after a successful scan. Keeping the mapping as valid Python makes
it safe for maintenance scripts to import and for full-repository compilation.
"""

from __future__ import annotations


HISTORY_SYNC_MAP: dict[str, str] = {
    "sport_bets": "exports/history_sport_bets.csv",
    "sport_bookmaker_stats": "exports/history_bookmaker_stats.csv",
    "sport_elo_ratings": "exports/history_elo_ratings.csv",
}


def restore_pairs() -> tuple[tuple[str, str], ...]:
    """Return (csv_path, table) pairs used before a production scan."""
    return tuple((csv_path, table) for table, csv_path in HISTORY_SYNC_MAP.items())


def export_pairs() -> tuple[tuple[str, str], ...]:
    """Return (table, csv_path) pairs used after a production scan."""
    return tuple(HISTORY_SYNC_MAP.items())

