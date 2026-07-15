from core.football_csv_snapshot_loader_v15_26 import (
    run_csv_snapshot_loader_v15_26
)

from core.football_snapshot_row_parser_adapter_v15_23 import (
    run_snapshot_row_parser_adapter_v15_23
)

from core.football_closing_odds_match_writer_v15_27 import (
    run_closing_odds_match_writer_v15_27
)


def run_full_closing_pipeline_v15_27():

    loader_report, snapshots = run_csv_snapshot_loader_v15_26()

    parser_report, parsed_rows = run_snapshot_row_parser_adapter_v15_23(
        snapshots=snapshots
    )

    closing_report = run_closing_odds_match_writer_v15_27(
        join_ready_rows=[
            row for row in parsed_rows
            if row.get("join_ready")
        ]
    )

    return {
        "loader": loader_report,
        "parser": parser_report,
        "closing": closing_report,
    }


if __name__ == "__main__":
    print(run_full_closing_pipeline_v15_27())
