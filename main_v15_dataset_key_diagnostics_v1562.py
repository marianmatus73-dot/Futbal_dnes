from core.football_dataset_key_diagnostics_v15_62 import (
    run_dataset_key_diagnostics_v15_62,
)


def run_full_pipeline_v15_62():
    return {
        "dataset_key_diagnostics": run_dataset_key_diagnostics_v15_62(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_62())
