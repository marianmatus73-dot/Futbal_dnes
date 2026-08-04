from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pprint import pprint

from core.multisport_learning.manager import MultisportLearningManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sport",
        required=True,
        choices=["baseball", "basketball", "tennis", "hockey", "mma", "nfl"],
    )
    parser.add_argument("--database", default="multisport_learning.db")
    args = parser.parse_args()

    manager = MultisportLearningManager(args.database)
    print(f"=== {args.sport.upper()} COMPLETE LEARNING PIPELINE ===")
    pprint(manager.run_sport(args.sport).as_dict())


if __name__ == "__main__":
    main()
