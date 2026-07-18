
"""
CLV Bridge V16.01
-----------------
Safe extension module for V16.00.
Does not modify main pipeline directly.

Flow:
snapshots -> resolver -> closing -> CLV -> learning payload
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class CLVRecord:
    event_id: str
    opening_odds: float
    closing_odds: float
    clv: float


def calculate_clv(opening_odds: float, closing_odds: float) -> float:
    if not opening_odds or not closing_odds:
        return 0.0

    return round(((closing_odds - opening_odds) / opening_odds) * 100, 4)


def build_clv_record(event_id, opening_odds, closing_odds):
    return CLVRecord(
        event_id=str(event_id),
        opening_odds=float(opening_odds),
        closing_odds=float(closing_odds),
        clv=calculate_clv(opening_odds, closing_odds),
    )


def run_clv_bridge(snapshot_rows):
    """
    snapshot_rows expected:
    [
      {
        'event_id': ...,
        'opening_odds': ...,
        'closing_odds': ...
      }
    ]
    """

    records = []

    for row in snapshot_rows:
        if row.get("opening_odds") and row.get("closing_odds"):
            records.append(
                build_clv_record(
                    row.get("event_id"),
                    row.get("opening_odds"),
                    row.get("closing_odds"),
                )
            )

    return {
        "version": "v16.01",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshots_found": len(snapshot_rows),
        "clv_records_created": len(records),
        "status": "READY",
        "records": [r.__dict__ for r in records],
    }
