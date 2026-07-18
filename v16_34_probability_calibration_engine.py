
"""
V16.34 PROBABILITY CALIBRATION ENGINE

Calibrates prediction confidence using historical feedback.
"""


def calibrate_probability(confidence, historical_accuracy):
    adjustment = (historical_accuracy - 0.5) * 0.2

    calibrated = confidence + adjustment
    calibrated = max(0.0, min(calibrated, 1.0))

    return {
        "original_confidence": confidence,
        "calibrated_probability": round(calibrated, 2),
        "status": "READY"
    }
