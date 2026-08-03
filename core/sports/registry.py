from core.multisport_learning.models import SportProfile


SPORT_PROFILES = {
    "baseball": SportProfile(
        name="baseball",
        rating_system="ELO_RUN_ADJUSTED",
        form_window=10,
        supports_draw=False,
        competition_label="league",
        min_training_samples=150,
    ),
    "basketball": SportProfile(
        name="basketball",
        rating_system="ELO_MARGIN_ADJUSTED",
        form_window=10,
        supports_draw=False,
        competition_label="league",
        min_training_samples=120,
    ),
    "tennis": SportProfile(
        name="tennis",
        rating_system="SURFACE_ELO",
        form_window=12,
        supports_draw=False,
        competition_label="tour",
        min_training_samples=150,
    ),
    "hockey": SportProfile(
        name="hockey",
        rating_system="ELO_GOAL_ADJUSTED",
        form_window=10,
        supports_draw=False,
        competition_label="league",
        min_training_samples=120,
    ),
    "mma": SportProfile(
        name="mma",
        rating_system="FIGHTER_ELO",
        form_window=5,
        supports_draw=True,
        competition_label="promotion",
        min_training_samples=100,
    ),
    "nfl": SportProfile(
        name="nfl",
        rating_system="ELO_POINT_ADJUSTED",
        form_window=8,
        supports_draw=True,
        competition_label="league",
        min_training_samples=100,
    ),
}
