from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    competition TEXT,
    event_key TEXT NOT NULL,
    event_time TEXT,
    home_name TEXT,
    away_name TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    home_score REAL,
    away_score REAL,
    result TEXT,
    UNIQUE(sport, event_key)
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    event_key TEXT NOT NULL,
    selection TEXT NOT NULL,
    model_probability REAL,
    market_probability REAL,
    odds REAL,
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    event_key TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds REAL NOT NULL,
    bookmaker TEXT,
    snapshot_time TEXT DEFAULT CURRENT_TIMESTAMP,
    snapshot_type TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE IF NOT EXISTS ml_feature_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    event_key TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    entity TEXT NOT NULL,
    rating REAL NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sport, entity)
);

CREATE TABLE IF NOT EXISTS ml_form (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    entity TEXT NOT NULL,
    form_score REAL NOT NULL,
    sample_size INTEGER NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sport, entity)
);

CREATE TABLE IF NOT EXISTS ml_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    event_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    training_ready INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sport, event_key)
);

CREATE TABLE IF NOT EXISTS ml_cycle_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    stages_json TEXT NOT NULL,
    errors_json TEXT NOT NULL
);
"""


def connect(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
