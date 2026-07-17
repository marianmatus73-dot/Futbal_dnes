from core.football_clv_movement_validator_v15_51 import (
    run_clv_movement_validator_v15_51,
)


def run_full_pipeline_v15_51():
    return {
        "validator": run_clv_movement_validator_v15_51(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_51())
