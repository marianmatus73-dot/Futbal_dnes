from core.football_snapshot_loader_v15_24 import (
    run_snapshot_loader_v15_24
)

from core.football_snapshot_row_parser_adapter_v15_23 import (
    run_snapshot_row_parser_adapter_v15_23
)


def run_v15_24_pipeline():
    loader_report, snapshots = run_snapshot_loader_v15_24()

    parser_report, rows = run_snapshot_row_parser_adapter_v15_23(
        snapshots=snapshots
    )

    return {
        "loader": loader_report,
        "parser": parser_report,
    }


if __name__ == "__main__":
    print(run_v15_24_pipeline())
