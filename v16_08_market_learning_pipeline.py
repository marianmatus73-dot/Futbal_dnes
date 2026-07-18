
"""
V16.08 MARKET LEARNING PIPELINE

ODDS
 ↓
DATABASE
 ↓
LEARNING DATASET
"""

from v16_08_market_database import save_market
from v16_08_learning_layer import build_learning_dataset


def run_pipeline():
    record = {
        "event_id": "learning_demo_001",
        "opening_odds": 2.10,
        "closing_odds": 1.90,
        "clv": -9.52
    }

    storage = save_market(record)
    learning = build_learning_dataset()

    return {
        "version": "V16.08",
        "storage": storage["status"],
        "learning": learning["dataset_status"],
        "records": learning["records"],
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.08 MARKET LEARNING PIPELINE ===")
    print(run_pipeline())
