from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(database):
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def table_columns(conn, table):
    return [
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    ]
