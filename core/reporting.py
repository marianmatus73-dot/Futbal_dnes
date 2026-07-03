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
            
            # Bezpečné vytiahnutie hodnôt - skontroluje, či je 'b' slovník alebo objekt
            if isinstance(b, dict):
                start_time = b.get("start_time", "N/A")
                league = b.get("league", "N/A")
                event = b.get("event", "N/A")
                selection = b.get("selection", "N/A")
                odds = b.get("odds", 0.0)
                prob_final = b.get("prob_final", 0.0)
                edge = b.get("edge", 0.0)
                stake = b.get("stake", 0.0)
                bookmaker = b.get("bookmaker", "N/A")
            else:
                start_time = getattr(b, "start_time", "N/A")
                league = getattr(b, "league", "N/A")
                event = getattr(b, "event", "N/A")
                selection = getattr(b, "selection", "N/A")
                odds = getattr(b, "odds", 0.0)
                prob_final = getattr(b, "prob_final", 0.0)
                edge = getattr(b, "edge", 0.0)
                stake = getattr(b, "stake", 0.0)
                bookmaker = getattr(b, "bookmaker", "N/A")

            # Výpis s pretypovaním na float, ak by náhodou prišiel string z DB/slovníka
            try:
                odds_val = float(odds)
                prob_val = float(prob_final)
                edge_val = float(edge)
                stake_val = float(stake)
            except (ValueError, TypeError):
                odds_val, prob_val, edge_val, stake_val = 0.0, 0.0, 0.0, 0.0

            print(
                f"{i:02d}. {start_time} | {league} | {event} | "
                f"{selection} @ {odds_val:.2f} | "
                f"P {prob_val:.1%} | Edge {edge_val:.1%} | "
                f"Stake {stake_val:.2f} | {bookmaker}"
            )

    print(f"\nTotal bets: {total}")
