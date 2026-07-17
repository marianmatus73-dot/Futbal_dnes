from core.football_universal_market_probability_resolver_v15_61 import (
    run_universal_market_probability_resolver_v15_61,
)


def run_full_pipeline_v15_61():
    return {
        "universal_market_probability": run_universal_market_probability_resolver_v15_61(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_61())
