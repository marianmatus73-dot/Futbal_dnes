from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command))
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    args = parser.parse_args()

    run_command(
        "scripts/migrate_database.py",
        "--database",
        args.database,
    )
    run_command(
        "scripts/validate_database.py",
        "--database",
        args.database,
    )
    run_command(
        "scripts/run_multisport_learning_v2.py",
        "--database",
        args.database,
    )

    print("Multisport Learning V2.1 installation: READY")
