from __future__ import annotations

import json
import math
from typing import Any

from .database import connect
from .models import SportProfile


def settle_events(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        open_events = conn.execute(
            "SELECT COUNT(*) FROM ml_events WHERE sport=? AND status='OPEN'",
            (profile.name,),
        ).fetchone()[0]
        settled = conn.execute(
            "SELECT COUNT(*) FROM ml_events WHERE sport=? AND status='SETTLED'",
            (profile.name,),
        ).fetchone()[0]
    return {"open_events": open_events, "settled_events": settled, "status": "READY"}


def learn_results(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
                COUNT(*) AS total
            FROM ml_events
            WHERE sport=? AND status='SETTLED'
            """,
            (profile.name,),
        ).fetchone()
    wins = int(row["wins"] or 0)
    losses = int(row["losses"] or 0)
    total = int(row["total"] or 0)
    return {
        "samples": total,
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / total, 4) if total else None,
        "status": "READY",
    }


def rebuild_ratings(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        entities = conn.execute(
            """
            SELECT home_name AS entity FROM ml_events WHERE sport=?
            UNION
            SELECT away_name AS entity FROM ml_events WHERE sport=?
            """,
            (profile.name, profile.name),
        ).fetchall()
        rebuilt = 0
        for row in entities:
            entity = row["entity"]
            if not entity:
                continue
            games = conn.execute(
                """
                SELECT result, home_name, away_name
                FROM ml_events
                WHERE sport=? AND status='SETTLED'
                  AND (home_name=? OR away_name=?)
                """,
                (profile.name, entity, entity),
            ).fetchall()
            rating = 1500.0
            for game in games:
                won = (
                    (game["result"] == "HOME_WIN" and game["home_name"] == entity)
                    or (game["result"] == "AWAY_WIN" and game["away_name"] == entity)
                )
                rating += 12.0 if won else -8.0
            conn.execute(
                """
                INSERT INTO ml_ratings (sport, entity, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(sport, entity)
                DO UPDATE SET rating=excluded.rating, updated_at=CURRENT_TIMESTAMP
                """,
                (profile.name, entity, rating),
            )
            rebuilt += 1
        conn.commit()
    return {"rating_system": profile.rating_system, "rebuilt": rebuilt, "status": "READY"}


def rebuild_form(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        entities = conn.execute(
            "SELECT entity FROM ml_ratings WHERE sport=?",
            (profile.name,),
        ).fetchall()
        rebuilt = 0
        for row in entities:
            entity = row["entity"]
            games = conn.execute(
                """
                SELECT result, home_name, away_name
                FROM ml_events
                WHERE sport=? AND status='SETTLED'
                  AND (home_name=? OR away_name=?)
                ORDER BY id DESC LIMIT ?
                """,
                (profile.name, entity, entity, profile.form_window),
            ).fetchall()
            points = 0.0
            for game in games:
                if game["result"] == "DRAW":
                    points += 0.5
                elif (
                    game["result"] == "HOME_WIN" and game["home_name"] == entity
                ) or (
                    game["result"] == "AWAY_WIN" and game["away_name"] == entity
                ):
                    points += 1.0
            score = points / len(games) if games else 0.5
            conn.execute(
                """
                INSERT INTO ml_form (sport, entity, form_score, sample_size)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sport, entity)
                DO UPDATE SET
                    form_score=excluded.form_score,
                    sample_size=excluded.sample_size,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (profile.name, entity, score, len(games)),
            )
            rebuilt += 1
        conn.commit()
    return {"window": profile.form_window, "rebuilt": rebuilt, "status": "READY"}


def collect_market(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM ml_market_snapshots WHERE sport=?",
            (profile.name,),
        ).fetchone()[0]
        closing = conn.execute(
            """
            SELECT COUNT(*) FROM ml_market_snapshots
            WHERE sport=? AND snapshot_type='CLOSE'
            """,
            (profile.name,),
        ).fetchone()[0]
    coverage = closing / total if total else 0.0
    return {
        "snapshots": total,
        "closing_snapshots": closing,
        "closing_coverage": round(coverage, 4),
        "status": "READY" if total else "BUILDING",
    }


def build_dataset(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        events = conn.execute(
            """
            SELECT event_key, competition, result
            FROM ml_events
            WHERE sport=? AND status='SETTLED'
            """,
            (profile.name,),
        ).fetchall()
        inserted = 0
        for event in events:
            prediction = conn.execute(
                """
                SELECT model_probability, market_probability, odds, confidence
                FROM ml_predictions
                WHERE sport=? AND event_key=?
                ORDER BY id DESC LIMIT 1
                """,
                (profile.name, event["event_key"]),
            ).fetchone()
            features = conn.execute(
                """
                SELECT feature_name, feature_value
                FROM ml_feature_history
                WHERE sport=? AND event_key=?
                """,
                (profile.name, event["event_key"]),
            ).fetchall()
            payload = {
                "event_key": event["event_key"],
                "competition": event["competition"],
                "result": event["result"],
                "prediction": dict(prediction) if prediction else None,
                "features": {row["feature_name"]: row["feature_value"] for row in features},
            }
            ready = int(prediction is not None)
            conn.execute(
                """
                INSERT INTO ml_datasets (sport, event_key, payload_json, training_ready)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sport, event_key)
                DO UPDATE SET
                    payload_json=excluded.payload_json,
                    training_ready=excluded.training_ready
                """,
                (profile.name, event["event_key"], json.dumps(payload), ready),
            )
            inserted += 1
        conn.commit()
        ready_total = conn.execute(
            "SELECT COUNT(*) FROM ml_datasets WHERE sport=? AND training_ready=1",
            (profile.name,),
        ).fetchone()[0]
    return {
        "dataset_rows": inserted,
        "training_ready": ready_total,
        "status": "READY",
    }


def evaluate(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        rows = conn.execute(
            """
            SELECT e.result, p.model_probability
            FROM ml_events e
            JOIN ml_predictions p
              ON p.sport=e.sport AND p.event_key=e.event_key
            WHERE e.sport=? AND e.status='SETTLED'
              AND p.model_probability IS NOT NULL
            """,
            (profile.name,),
        ).fetchall()
    if not rows:
        return {
            "samples": 0,
            "brier_score": None,
            "log_loss": None,
            "status": "BUILDING",
        }

    brier = 0.0
    log_loss = 0.0
    for row in rows:
        target = 1.0 if row["result"] in {"WIN", "HOME_WIN"} else 0.0
        prob = min(max(float(row["model_probability"]), 1e-6), 1 - 1e-6)
        brier += (prob - target) ** 2
        log_loss += -(target * math.log(prob) + (1 - target) * math.log(1 - prob))
    count = len(rows)
    return {
        "samples": count,
        "brier_score": round(brier / count, 6),
        "log_loss": round(log_loss / count, 6),
        "status": "READY",
    }


def ai_health(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        samples = conn.execute(
            "SELECT COUNT(*) FROM ml_datasets WHERE sport=? AND training_ready=1",
            (profile.name,),
        ).fetchone()[0]
        snapshots = conn.execute(
            "SELECT COUNT(*) FROM ml_market_snapshots WHERE sport=?",
            (profile.name,),
        ).fetchone()[0]
        closing = conn.execute(
            """
            SELECT COUNT(*) FROM ml_market_snapshots
            WHERE sport=? AND snapshot_type='CLOSE'
            """,
            (profile.name,),
        ).fetchone()[0]
    maturity = (
        "READY" if samples >= profile.min_training_samples
        else "BUILDING" if samples > 0
        else "EMPTY"
    )
    return {
        "training_samples": samples,
        "market_snapshots": snapshots,
        "closing_snapshots": closing,
        "maturity": maturity,
        "status": "READY",
    }


def maintenance(database: str, profile: SportProfile) -> dict[str, Any]:
    with connect(database) as conn:
        deleted = conn.execute(
            """
            DELETE FROM ml_market_snapshots
            WHERE sport=? AND odds<=1
            """,
            (profile.name,),
        ).rowcount
        conn.commit()
    return {"deleted_invalid_snapshots": deleted, "status": "READY"}
