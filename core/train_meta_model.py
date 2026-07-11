from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "meta_model.pkl"


def train(db_file: str = "bets.db") -> bool:
    conn = sqlite3.connect(db_file)

    rows = conn.execute(
        """
        SELECT
            prob_market,
            prob_final,
            edge,
            odds,
            stake,
            CASE
                WHEN result IN ('WON','V') THEN 1
                ELSE 0
            END
        FROM sport_bets
        WHERE result IN ('WON','LOST','V','P')
        """
    ).fetchall()

    conn.close()

    if len(rows) < 100:
        print(f"Not enough settled bets ({len(rows)})")
        return False

    X = []
    y = []

    for row in rows:
        X.append(list(row[:-1]))
        y.append(row[-1])

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5,
    )

    model.fit(X, y)

    with MODEL_FILE.open("wb") as f:
        pickle.dump(model, f)

    print(f"Model trained on {len(rows)} bets.")
    return True


if __name__ == "__main__":
    train()
