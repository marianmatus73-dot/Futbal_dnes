
"""
V16.02 CLV Integration Layer

Connects CLV pipeline with main application safely.

Safe extension:
- does not replace main.py
- keeps other sports untouched
"""


from clv_pipeline_runner_v16_01_8 import run_full_pipeline


def run_clv_module(events):
    result = run_full_pipeline(events)

    return {
        "module": "CLV_INTEGRATION",
        "version": "v16.02",
        "status": result["pipeline_status"],
        "clv_records": result["clv_records"],
        "output": result["clv_output"],
    }


if __name__ == "__main__":
    test_events = [
        {
            "event_id": "integration_test_001",
            "home_team": "Team A",
            "away_team": "Team B",
            "opening_odds": 2.00,
            "closing_odds": 1.80,
            "bookmaker": "demo",
        }
    ]

    print("=== V16.02 CLV INTEGRATION TEST ===")
    print(run_clv_module(test_events))
