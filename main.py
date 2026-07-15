from core.football_match_diagnostics_v15_30 import (
    run_match_diagnostics_v15_30
)


def main() -> None:
    report = run_match_diagnostics_v15_30(top_candidates=3)
    print(report)


if __name__ == "__main__":
    main()
