from core.football_real_closing_snapshot_finder_v15_52 import (
    run_real_closing_snapshot_finder_v15_52,
)


def run_full_pipeline_v15_52():
    report, _ = run_real_closing_snapshot_finder_v15_52()
    return {
        "real_closing_snapshot": report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_52())
