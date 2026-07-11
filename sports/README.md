# Sports modules v12

Replace files in your project `sports/` directory with these files.

Included: baseball, tennis, basketball, hockey, MMA, NFL, esports, and a standalone football v12 module.

Important: the new football module replaces the previous external-engine wrapper. Keep a backup of the old `sports/football.py` and the external football engine before switching.

All modules include:
- MetaFeatures + predict_probability
- safe fallback to the original probability
- Monte Carlo preview input
- adaptive sport/bookmaker/league weights
- audit tags showing META_MODEL or FALLBACK

Run syntax check:

```bash
python -m py_compile sports/baseball.py sports/tennis.py sports/basketball.py sports/hockey.py sports/mma.py sports/nfl.py sports/esports.py sports/football.py
```

Then run:

```bash
python main.py --sport all
```
