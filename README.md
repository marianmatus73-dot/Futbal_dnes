# Multisport Learning Bundle

Reusable learning framework for:

- baseball
- basketball
- tennis
- hockey
- mma
- nfl

The bundle provides one shared architecture for settlement, result learning,
feature history, ratings, form, market snapshots, datasets, evaluation,
AI health, learning progress, closing-odds/CLV readiness, maintenance and
competition profiles.

## Important

This is a production-oriented framework and integration scaffold. It does not
invent sport results or closing odds. Sport-specific APIs, scoring rules and
feature extraction should be connected through the adapters in
`core/sports/<sport>.py`.

## Quick test

```bash
python scripts/run_all_sports_pipeline.py
```

## Single sport

```bash
python scripts/run_sport_pipeline.py --sport baseball
```

## Output

- SQLite tables in `multisport_learning.db`
- JSON summaries in `exports/`
- one combined report in `exports/multisport_learning_report.json`

## Main integration

```python
from core.multisport_learning.manager import MultisportLearningManager

manager = MultisportLearningManager("multisport_learning.db")
result = manager.run_all()
```
