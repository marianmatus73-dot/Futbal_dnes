
"""
Test runner for CLV Bridge V16.01

Safe test only:
- does not modify main.py
- loads sample snapshot structure
- verifies CLV bridge execution

Replace snapshot_rows loading with real football_market_snapshots_v14
loader after first validation.
"""

from clv_bridge_v16_01 import run_clv_bridge


def load_test_snapshots():
    return [
        {
            "event_id": "test_match_001",
            "opening_odds": 2.10,
            "closing_odds": 1.90,
        },
        {
            "event_id": "test_match_002",
            "opening_odds": 1.80,
            "closing_odds": 2.00,
        },
    ]


if __name__ == "__main__":
    snapshots = load_test_snapshots()

    result = run_clv_bridge(snapshots)

    print("=== V16.01 CLV BRIDGE TEST ===")
    print("Snapshots found:", result["snapshots_found"])
    print("CLV records created:", result["clv_records_created"])
    print("Status:", result["status"])

    for row in result["records"]:
        print(row)
