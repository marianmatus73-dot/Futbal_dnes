
"""
V16.18 MARKET ANOMALY DETECTION

Detects unusual market movements.
"""


def detect_anomaly(opening, current, average_move=0.05):
    movement = current - opening

    if abs(movement) > average_move:
        anomaly = True
    else:
        anomaly = False

    return {
        "opening": opening,
        "current": current,
        "movement": movement,
        "anomaly": anomaly,
        "status": "READY"
    }
