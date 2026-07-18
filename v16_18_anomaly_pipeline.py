
"""
V16.18 MARKET ANOMALY PIPELINE
"""

from v16_18_anomaly_detection import detect_anomaly


def run_pipeline():
    result = detect_anomaly(
        opening=2.20,
        current=1.90
    )

    return {
        "version": "V16.18",
        "anomaly_detection": result,
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.18 ANOMALY PIPELINE ===")
    print(run_pipeline())
