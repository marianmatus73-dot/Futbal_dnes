from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pprint import pprint

from core.multisport_learning_v2.manager import MultisportLearningV2Manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    parser.add_argument("--table", default="sport_bets")
    parser.add_argument("--export-dir", default="exports")
    args = parser.parse_args()

    manager = MultisportLearningV2Manager(
        args.database,
        table=args.table,
    )
    print("=== MULTISPORT LEARNING BUNDLE V2 ===")
    pprint(manager.run_all(export_dir=args.export_dir))


if __name__ == "__main__":
    main()
