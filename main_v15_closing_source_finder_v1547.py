from core.football_correct_snapshot_source_loader_v15_39 import run_correct_snapshot_source_loader_v15_39
from core.football_event_split_adapter_v15_32 import run_event_split_adapter_v15_32
from core.football_source_hash_resolver_v15_37 import run_source_hash_resolver_v15_37
from core.football_closing_source_finder_v15_47 import run_closing_source_finder_v15_47


def main():

    _, snapshots = run_correct_snapshot_source_loader_v15_39()
    _, postmatch = run_event_split_adapter_v15_32()

    _, resolved = run_source_hash_resolver_v15_37(
        snapshots,
        postmatch,
    )

    print(
        run_closing_source_finder_v15_47(
            resolved
        )
    )


if __name__ == "__main__":
    main()
