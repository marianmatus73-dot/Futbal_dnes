from core.multisport_learning_v2.manager import MultisportLearningV2Manager


def run_multisport_v2_1(settings, db_path, log):
    manager = MultisportLearningV2Manager(db_path(settings))
    result = manager.run_all(export_dir="exports")

    log.info(
        "Multisport Learning V2.1: sports=%s ready=%s status=%s",
        result.get("sports_completed", 0),
        result.get("sports_ready", 0),
        result.get("status", "UNKNOWN"),
    )

    return result
