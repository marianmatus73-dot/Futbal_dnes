"""
V16.72 NEXT GENERATION ARCHITECTURE BUILDER

Creates improved system configuration from evolution results.
"""


def build_next_generation(evolution_score, modules):
    upgrade_level = "GENERATION_UPGRADE" if evolution_score >= 0.9 else "OPTIMIZATION"

    return {
        "evolution_score": evolution_score,
        "modules_analyzed": len(modules),
        "upgrade_level": upgrade_level,
        "new_configuration_ready": True,
        "modules": modules,
        "status": "READY"
    }