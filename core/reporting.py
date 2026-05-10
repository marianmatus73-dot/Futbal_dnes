from __future__ import annotations

from core.types import SportResult


def print_report(results: list[SportResult]) -> None:
    print("\nMULTISPORT BETTING ENGINE REPORT")
    print("=" * 42)

    total = 0

    for result in results:
        print(f"\n[{result.sport.upper()}] {result.mode}")

        if result.message:
            print(result.message)

        if not result.bets:
            print("No bets found.")
            continue

        for i, b in enumerate(result.bets, 1):
            total += 1
            print(
                f"{i:02d}. {b.start_time} | {b.league} | {b.event} | "
                f"{b.selection} @ {b.odds:.2f} | "
                f"P {b.prob_final:.1%} | Edge {b.edge:.1%} | "
                f"Stake {b.stake:.2f} | {b.bookmaker}"
            )

    print(f"\nTotal bets: {total}")
