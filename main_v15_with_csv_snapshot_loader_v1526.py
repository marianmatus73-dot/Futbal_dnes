from core.football_csv_snapshot_loader_v15_26 import (
    run_csv_snapshot_loader_v15_26
)

from core.football_snapshot_row_parser_adapter_v15_23 import (
    run_snapshot_row_parser_adapter_v15_23
)


def run_v15_26_pipeline():
    loader_report, snapshots = run_csv_snapshot_loader_v15_26()

    parser_report, rows = run_snapshot_row_parser_adapter_v15_23(
        snapshots=snapshots
    )

    return {
        "loader": loader_report,
        "parser": parser_report,
    }


if __name__ == "__main__":
    print(run_v15_26_pipeline())
