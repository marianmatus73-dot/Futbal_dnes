"""
V16.16 LEAGUE PROFILE PIPELINE
"""

from v16_16_league_profile import save_profile, load_profiles


def run_pipeline():
    profiles = load_profiles()

    if not profiles:
        save_profile({
            "sport": "football",
            "league": "demo_league",
            "market": "1X2",
            "success_rate": 0.70
        })
        profiles = load_profiles()

    return {
        "version": "V16.16",
        "profiles": len(profiles),
        "status": "READY"
    }


if __name__ == "__main__":
    print("=== V16.16 LEAGUE PROFILE PIPELINE ===")
    print(run_pipeline())
