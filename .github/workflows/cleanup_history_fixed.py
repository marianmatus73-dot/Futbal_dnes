from __future__ import annotations

import csv
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Settings


UNIQUE_COLUMNS = (
    "sport",
    "league",
    "event",
    "market",
    "selection",
    "start_time",
)

SETTLED_RESULTS = {"WON", "LOST", "VOID", "V", "P"}
OPEN_RESULTS = {"", "OPEN", None}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_result(value: Any) -> str:
    result = str(value or "").strip().upper()

    if result in {"", "OPEN"}:
        return "OPEN"

    if result == "V":
        return "WON"

    if result == "P":
        return "LOST"

    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "") or "").strip() for column in UNIQUE_COLUMNS)


def _row_rank(row: dict[str, Any]) -> tuple[int, int, str, int]:
    """
    Vyberá najlepší riadok z duplicít:
    1. settled riadok pred OPEN,
    2. riadok s vyplneným profitom,
    3. najnovší settled_at,
    4. najvyššie id.
    """
    result = _normalize_result(row.get("result"))
    settled = 1 if result in SETTLED_RESULTS else 0
    has_profit = 1 if row.get("profit") not in (None, "") else 0
    settled_at = str(row.get("settled_at", "") or "")
    row_id = _safe_int(row.get("id"), 0)

    return settled, has_profit, settled_at, row_id


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return target


def cleanup_database(db_file: Path) -> tuple[int, int]:
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file}")

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sport_bets)").fetchall()
        }

        missing = [column for column in UNIQUE_COLUMNS if column not in columns]
        if missing:
            raise RuntimeError(
                "sport_bets is missing required columns: "
                + ", ".join(missing)
            )

        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM sport_bets ORDER BY id"
            ).fetchall()
        ]

        before = len(rows)

        best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}

        for row in rows:
            key = _row_key(row)
            current = best_by_key.get(key)

            if current is None or _row_rank(row) > _row_rank(current):
                best_by_key[key] = row

        keep_ids = {
            _safe_int(row.get("id"))
            for row in best_by_key.values()
            if _safe_int(row.get("id")) > 0
        }

        if keep_ids:
            placeholders = ",".join("?" for _ in keep_ids)
            conn.execute(
                f"DELETE FROM sport_bets WHERE id NOT IN ({placeholders})",
                tuple(sorted(keep_ids)),
            )

        conn.execute(
            """
            UPDATE sport_bets
            SET result='OPEN'
            WHERE result IS NULL OR TRIM(result)='' OR UPPER(result)='OPEN'
            """
        )

        conn.execute(
            """
            UPDATE sport_bets
            SET result='WON'
            WHERE UPPER(result)='V'
            """
        )

        conn.execute(
            """
            UPDATE sport_bets
            SET result='LOST'
            WHERE UPPER(result)='P'
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

    return before, int(after)


def cleanup_csv(csv_file: Path) -> tuple[int, int]:
    if not csv_file.exists():
        return 0, 0

    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    before = len(rows)

    best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in rows:
        row["result"] = _normalize_result(row.get("result"))
        key = _row_key(row)
        current = best_by_key.get(key)

        if current is None or _row_rank(row) > _row_rank(current):
            best_by_key[key] = row

    cleaned = list(best_by_key.values())
    cleaned.sort(key=lambda row: _safe_int(row.get("id"), 0))

    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    return before, len(cleaned)


def remove_meta_model(model_file: Path) -> bool:
    if not model_file.exists():
        return False

    model_file.unlink()
    return True


def main() -> None:
    settings = Settings.from_env()

    db_file = Path(settings.db_file or "bets.db")
    csv_file = Path("exports/history_sport_bets.csv")
    model_file = Path("models/meta_model.pkl")

    backup_dir = Path("backups") / f"history_cleanup_{_timestamp()}"

    db_backup = backup_file(db_file, backup_dir)
    csv_backup = backup_file(csv_file, backup_dir)
    model_backup = backup_file(model_file, backup_dir)

    print("Backups:")
    print(f"- database: {db_backup or 'not found'}")
    print(f"- history CSV: {csv_backup or 'not found'}")
    print(f"- meta model: {model_backup or 'not found'}")

    if db_file.exists():
        db_before, db_after = cleanup_database(db_file)
    else:
        db_before, db_after = 0, 0
        print()
        print(
            "Database file was not found. "
            "Skipping database cleanup and cleaning the persistent CSV history only."
        )

    csv_before, csv_after = cleanup_csv(csv_file)
    model_removed = remove_meta_model(model_file)

    print()
    print("Cleanup finished:")

    if db_file.exists():
        print(f"- database rows: {db_before} -> {db_after}")
        print(f"- database duplicates removed: {db_before - db_after}")
    else:
        print("- database cleanup: skipped (bets.db does not exist in this workflow)")

    print(f"- CSV rows: {csv_before} -> {csv_after}")
    print(f"- CSV duplicates removed: {csv_before - csv_after}")
    print(f"- meta_model.pkl removed: {model_removed}")
    print(f"- backups saved in: {backup_dir}")
    print()
    print("Next run:")
    print("python main.py --sport all")


if __name__ == "__main__":
    main()
