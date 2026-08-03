from __future__ import annotations

from pprint import pprint

from core.multisport_learning.manager import MultisportLearningManager


if __name__ == "__main__":
    manager = MultisportLearningManager("multisport_learning.db")
    print("=== MULTISPORT COMPLETE LEARNING PIPELINE ===")
    result = manager.export_report("exports/multisport_learning_report.json")
    pprint(result)
