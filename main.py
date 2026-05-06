#!/usr/bin/env python3
"""
Vylepšený futbalový tipper - Stabilná verzia
Opravené: Načítanie BANK, chýbajúce importy a robustnosť dát.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import io  # Opravené: Pridaný chýbajúci import
import logging
import joblib
from scipy.stats import poisson
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lightgbm import LGBMClassifier

# ================= KONFIGURÁCIA =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')

# OPRAVA: Ošetrenie prázdneho reťazca v environmentálnych premenných
raw_bank = os.getenv('AKTUALNY_BANK', '1000')
BANK = float(raw_bank) if raw_bank and raw_bank.strip() else 1000.0

KELLY_FRAC = 0.25
HISTORY_FILE = "historia_tipov.csv"
MODEL_FILE = "ai_model.pkl"
CACHE_DIR = Path("cache")

CACHE_DIR.mkdir(exist_ok=True)

LIGY: Dict[str, Dict] = {
    'Premier League': {'csv': 'E0', 'api': 'soccer_epl', 'ha': 0.35},
    'La Liga': {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'ha': 0.38},
    'Bundesliga': {'csv': 'D1', 'api': 'soccer_germany_bundesliga', 'ha': 0.40},
    'Serie A': {'csv': 'I1', 'api': 'soccer_italy_serie_a', 'ha': 0.30},
    'Ligue 1': {'csv': 'F1', 'api': 'soccer_france_ligue_one', 'ha': 0.32},
    'Eredivisie': {'csv': 'N1', 'api': 'soccer_netherlands_eredivisie', 'ha': 0.42},
    'Championship': {'csv': 'E1', 'api': 'soccer_efl_champ', 'ha': 0.28},
    'Primeira Liga': {'csv': 'P1', 'api': 'soccer_portugal_primeira_liga', 'ha': 0.35},
    'Jupiler Pro League': {'csv': 'B1', 'api': 'soccer_belgium_pro_league', 'ha': 0.38},
    'Super Lig': {'csv': 'T1', 'api': 'soccer_turkey_super_league', 'ha': 0.40},
    'Scottish Premiership': {'csv': 'SC0', 'api': 'soccer_scotland_premier_league', 'ha': 0.36}
}

# ================= UTILITY FUNKCIE =================
async def fetch_csv(session: aiohttp.ClientSession, url: str, league: str) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{Path(url).stem}.csv"
    
    if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).seconds < 3600:
        return pd.read_csv(cache_file)
    
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                text = await resp.text()
                df = pd.read_csv(io.StringIO(text)) # Použitie importovaného io
                df.to_csv(cache_file, index=False)
                return df
    except Exception as e:
        logging.warning(f"Chyba pri sťahovaní {league}: {e}")
    return None

def validate_df(df: pd.DataFrame) -> bool:
    required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    return all(col in df.columns for col in required) and len(df.dropna(subset=['FTHG'])) > 50

def calculate_team_strength(df: pd.DataFrame) -> Tuple[Dict, float, float]:
    df_clean = df.dropna(subset=['FTHG', 'FTAG'])
    league_home_avg = max(df_clean['FTHG'].mean(), 0.1)
    league_away_avg = max(df_clean['FTAG'].mean(), 0.1)
    
    teams = {}
    for team in set(df_clean['HomeTeam']) | set(df_clean['AwayTeam']):
        home_games = df_clean[df_clean['HomeTeam'] == team]
        away_games = df_clean[df_clean['AwayTeam'] == team]
        
        teams[team] = {
            'attack_home': home_games['FTHG'].mean() / league_home_avg if len(home_games) > 0 else 1.0,
            'defense_home': home_games['FTAG'].mean() / league_away_avg if len(home_games) > 0 else 1.0,
            'attack_away': away_games['FTAG'].mean() / league_away_avg if len(away_games) > 0 else 1.0,
            'defense_away': away_games['FTHG'].mean() / league_home_avg if len(away_games) > 0 else 1.0
        }
    return teams, league_home_avg, league_away_avg

def expected_goals(home: str, away: str, teams: Dict, avg_h: float, avg_a: float, ha: float) -> Tuple[float, float]:
    h_stats, a_stats = teams.get(home), teams.get(away)
    if not h_stats or not a_stats: return avg_h, avg_a
    lh = h_stats['attack_home'] * a_stats['defense_away'] * avg_h + ha
    la = a_stats['attack_away'] * h_stats['defense_home'] * avg_a
    return max(lh, 0.1), max(la, 0.1)

def poisson_probs(lh: float, la: float) -> Dict[str, float]:
    matrix = np.outer(poisson.pmf(np.arange(10), lh), poisson.pmf(np.arange(10), la))
    matrix /= matrix.sum()
    return {
        '1': float(np.sum(np.tril(matrix, -1))),
        'X': float(np.sum(np.diag(matrix))),
        '2': float(np.sum(np.triu(matrix, 1))),
        'Over 2.5': float(np.sum(matrix[np.sum(np.indices((10, 10)), axis=0) >= 3]))
    }

# ================= ML MODEL =================
def train_model() -> Optional[Tuple[LGBMClassifier, List[str]]]:
    if not Path(HISTORY_FILE).exists(): return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if len(df) < 50: return None
        
        df['win'] = (df['Vysledok'] == 'V').astype(int)
        df['lh-la'] = df['lh'] - df['la']
        features = ['Edge', 'Kurz', 'lh', 'la', 'lh-la']
        
        model = LGBMClassifier(n_estimators=100, max_depth=3, verbosity=-1)
        model.fit(df[features], df['win'])
        return model, features
    except: return None

def model_filter(bet_data: Dict, model, features: List[str]) -> bool:
    if not model: return True
    row = pd.DataFrame([{**bet_data, 'lh-la': bet_data['lh'] - bet_data['la']}])
    return model.predict_proba(row[features])[0][1] >= 0.55

# ================= HLAVNÁ LOGIKA =================
async def process_league(session, name, cfg, model, features):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df = await fetch_csv(session, url, name)
    if df is None or not validate_df(df): return []

    teams, avg_h, avg_a = calculate_team_strength(df)
    
    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/',
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as resp:
        if resp.status != 200: return []
        odds_data = await resp.json()

    bets, now = [], datetime.utcnow()
    for match in odds_data:
        h, a = match['home_team'], match['away_team']
        m_time = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        if not (now <= m_time <= now + timedelta(hours=48)): continue

        lh, la = expected_goals(h, a, teams, avg_h, avg_a, cfg['ha'])
        probs = poisson_probs(lh, la)

        for bk in match.get('bookmakers', []):
            for mk in bk.get('markets', []):
                for out in mk.get('outcomes'):
                    label = '1' if out['name'] == h else ('2' if out['name'] == a else ('X' if out['name'] == 'Draw' else 'Over 2.5' if 'Over 2.5' in out['name'] else None))
                    if not label or label not in probs: continue

                    odds, edge = out['price'], (probs[label] * out['price']) - 1
                    if 0.03 <= edge <= 0.25:
                        bet_info = {'Edge': edge, 'Kurz': odds, 'lh': lh, 'la': la}
                        if model_filter(bet_info, model, features):
                            # Kellyho kritérium s limitom 15%
                            k = ((odds - 1) * probs[label] - (1 - probs[label])) / (odds - 1)
                            stake = max(0, min(k * KELLY_FRAC, 0.15)) * BANK
                            if stake > 0:
                                bets.append({
                                    'Datum': m_time.strftime('%Y-%m-%d %H:%M'),
                                    'Zápas': f"{h} vs {a}", 'Tip': label, 'Kurz': odds,
                                    'Edge': round(edge, 4), 'lh': round(lh, 2), 'la': round(la, 2),
                                    'Vklad': round(stake, 2), 'Vysledok': ''
                                })
    return bets

async def main():
    if not API_ODDS_KEY: return logging.error("Chýba API kľúč!")
    async with aiohttp.ClientSession() as session:
        model, features = train_model()
        sem = asyncio.Semaphore(5)
        tasks = [process_league(session, n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        all_bets = [b for sub in results for b in sub]

        if all_bets:
            new_df = pd.DataFrame(all_bets)
            if Path(HISTORY_FILE).exists():
                new_df = pd.concat([pd.read_csv(HISTORY_FILE), new_df]).drop_duplicates(subset=['Zápas', 'Tip', 'Datum'])
            new_df.to_csv(HISTORY_FILE, index=False)
            logging.info(f"Uložených {len(all_bets)} tipov.")

if __name__ == "__main__":
    asyncio.run(main())
