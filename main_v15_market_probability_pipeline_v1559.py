from core.football_market_probability_joiner_v15_59 import (
    run_market_probability_joiner_v15_59,
)


def run_full_pipeline_v15_59():
    return {
        "market_probability_join": run_market_probability_joiner_v15_59(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_59())
