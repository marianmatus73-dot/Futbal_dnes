from core.football_csv_snapshot_loader_v15_26 import run_csv_snapshot_loader_v15_26
from core.football_snapshot_row_parser_adapter_v15_23 import run_snapshot_row_parser_adapter_v15_23
from core.football_event_split_adapter_v15_32 import run_event_split_adapter_v15_32
from core.football_smart_resolver_adapter_v15_35 import run_smart_resolver_adapter_v15_35


def run_full_pipeline_v15_35():

    loader_report, snapshots = run_csv_snapshot_loader_v15_26()

    parser_report, parsed_rows = run_snapshot_row_parser_adapter_v15_23(
        snapshots=snapshots
    )

    event_report, postmatch_rows = run_event_split_adapter_v15_32()

    ready_rows = [
        row for row in parsed_rows
        if row.get("join_ready")
    ]

    resolver_report, resolved = run_smart_resolver_adapter_v15_35(
        ready_rows,
        postmatch_rows
    )

    return {
        "loader": loader_report,
        "parser": parser_report,
        "event_split": event_report,
        "resolver": resolver_report,
        "resolved_matches": len(resolved),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_35())
