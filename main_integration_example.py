"""
Example integration into the existing main.py.
"""

from core.multisport_learning.manager import MultisportLearningManager


def run_complete_sport_learning_bundle() -> dict:
    manager = MultisportLearningManager("bets.db")
    return manager.run_all(
        ["baseball", "basketball", "tennis", "hockey", "mma", "nfl"]
    )
