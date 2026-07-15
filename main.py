from core.football_csv_snapshot_loader_v15_26 import run_csv_snapshot_loader_v15_26
from core.football_event_split_adapter_v15_32 import run_event_split_adapter_v15_32
from core.football_resolver_debug_v15_36 import debug_resolver


def main():
    _, snapshots = run_csv_snapshot_loader_v15_26()
    _, postmatch = run_event_split_adapter_v15_32()

    print(debug_resolver(snapshots, postmatch))


if __name__ == "__main__":
    main()
