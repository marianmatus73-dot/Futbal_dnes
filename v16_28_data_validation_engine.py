
"""
V16.28 DATA VALIDATION ENGINE

Validates incoming data before processing.
"""


def validate_record(record):
    required = ["event_id", "odds"]

    missing = [x for x in required if x not in record]

    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "status": "READY" if len(missing) == 0 else "INVALID"
    }


def validate_batch(records):
    results = [validate_record(r) for r in records]

    return {
        "records_checked": len(records),
        "valid_records": sum(1 for r in results if r["valid"]),
        "status": "READY"
    }
