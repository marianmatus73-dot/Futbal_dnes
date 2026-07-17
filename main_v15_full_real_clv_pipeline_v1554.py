from core.football_full_real_clv_pipeline_v15_54 import (
    run_full_real_clv_pipeline_v15_54,
)


def run_full_pipeline_v15_54():

    return run_full_real_clv_pipeline_v15_54(
        {"status": "READY"},
        {"status": "READY"},
        {"status": "READY"},
        {"status": "READY"},
        {"status": "READY"},
    )


if __name__ == "__main__":
    print(run_full_pipeline_v15_54())
