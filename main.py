#!/usr/bin/env python3
"""
Vylepšený futbalový tipper s Poisson modelom, Kelly criterionom a ML validáciou.
Opravené: importy, spracovanie dátumu a bezpečnosť výpočtov.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import io  # Pridaný chýbajúci import pre io.StringIO
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
# Bezpečné načítanie banku
BANK = float(os.getenv('AKTUALNY_BANK', '1000'))
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
    """Načítanie CSV s retry a cache."""
    cache_file = CACHE_DIR / f"{Path(url).stem}.csv"
    
    if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).seconds < 3600:
        logging.info(f"Cache hit pre {league}")
        return pd.read_csv(cache_file)
    
    for attempt in range(3):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    df = pd.read_csv(io.StringIO(text))
                    df.to_csv(cache_file, index=False)
                    logging.info(f"Načítané {len(df)} zápasov pre {league}")
                    return df
        except Exception as e:
            logging.warning(f"Chyba načítania {league} (pokus {attempt+1}): {e}")
            await asyncio.sleep(1)
    return None

def validate_df(df: pd.DataFrame, min_rows: int = 50) -> bool:
    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    if not all(col in df.columns for col in required_cols):
        return False
    if len(df.dropna(subset=['FTHG', 'FTAG'])) < min_rows:
        return False
    return True

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
    if not h_stats or not a_stats:
        return avg_h, avg_a
    lh = h_stats['attack_home'] * a_stats['defense_away'] * avg_h + ha
    la = a_stats['attack_away'] * h_stats['defense_home'] * avg_a
    return max(lh, 0.1), max(la, 0.1)

def poisson_probs(lh: float, la: float) -> Dict[str, float]:
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        '1': float(np.sum(np.tril(matrix, -1))),
        'X': float(np.sum(np.diag(matrix))),
        '2': float(np.sum(np.triu(matrix, 1))),
        'Over 2.5': float(sum(matrix[x, y] for x in range(10) for y in range(10) if x + y >= 3))
    }

# ================= ML MODEL =================
def train_model() -> Optional[Tuple[LGBMClassifier, List[str]]]:
    if not Path(HISTORY_FILE).exists():
        return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        # Prísny filter na validné dáta pre učenie
        df = df.dropna(subset=['Vysledok', 'Edge', 'Kurz', 'lh', 'la'])
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        
        if len(df) < 50:
            return None
        
        df['win'] = (df['Vysledok'] == 'V').astype(int)
        features = ['Edge', 'Kurz', 'lh', 'la', 'lh-la']
        df['lh-la'] = df['lh'] - df['la']
        
        model = LGBMClassifier(n_estimators=150, max_depth=4, verbosity=-1, random_state=42)
        model.fit(df[features], df['win'])
        joblib.dump((model, features), MODEL_FILE)
        logging.info(f"AI Model natrénovaný na {len(df)} zápasoch.")
        return model, features
    except Exception as e:
        logging.error(f"Chyba tréningu: {e}")
        return None

def model_filter(bet_data: Dict, model, features: List[str], threshold: float = 0.55) -> bool:
    if not model: return True
    row = pd.DataFrame([{
        'Edge': bet_data['edge'], 'Kurz': bet_data['odds'],
        'lh': bet_data['lh'], 'la': bet_data['la'], 'lh-la': bet_data['lh'] - bet_data['la']
    }])
    prob_win = model.predict_proba(row[features])[0][1]
    return prob_win >= threshold

def kelly_criterion(prob: float, odds: float) -> float:
    if prob <= 0 or odds <= 1: return 0.0
    k = ((odds - 1) * prob - (1 - prob)) / (odds - 1)
    return max(0, min(k * KELLY_FRAC, 0.15)) # Limit 15% banku

# ================= CORE LOGIKA =================
async def process_league(session: aiohttp.ClientSession, name: str, cfg: Dict, model, features: List[str]) -> List[Dict]:
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df = await fetch_csv(session, url, name)
    if df is None or not validate_df(df): return []

    teams, avg_h, avg_a = calculate_team_strength(df)
    
    # Odds API volanie
    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/',
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as resp:
        if resp.status != 200: return []
        odds_data = await resp.json()

    bets, now = [], datetime.utcnow()
    for match in odds_data:
        h, a = match['home_team'], match['away_team']
        try:
            m_time = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        except: continue

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
                        if not model_filter({'edge': edge, 'odds': odds, 'lh': lh, 'la': la}, model, features): continue
                        
                        stake = kelly_criterion(probs[label], odds) * BANK
                        if stake > 0:
                            bets.append({
                                'Datum': m_time.strftime('%d.%m.%Y %H:%M'),
                                'Zápas': f"{h} vs {a}", 'Tip': label, 'Kurz': odds,
                                'Edge': round(edge, 4), 'lh': round(lh, 2), 'la': round(la, 2),
                                'Vklad': round(stake, 2), 'Vysledok': ''
                            })
    return bets

async def main():
    if not API_ODDS_KEY:
        logging.error("Chýba API kľúč!")
        return

    async with aiohttp.ClientSession() as session:
        model, features = train_model()
        semaphore = asyncio.Semaphore(5)

        async def sem_process(n, c, m, f):
            async with semaphore: return await process_league(session, n, c, m, f)

        tasks = [sem_process(n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        all_bets = [b for sub in results for b in sub]

        if all_bets:
            new_df = pd.DataFrame(all_bets)
            if Path(HISTORY_FILE).exists():
                hist_df = pd.read_csv(HISTORY_FILE)
                final_df = pd.concat([hist_df, new_df]).drop_duplicates(subset=['Zápas', 'Tip', 'Datum'], keep='first')
            else:
                final_df = new_df
            final_df.to_csv(HISTORY_FILE, index=False)
            logging.info(f"Nájdených a uložených {len(all_bets)} tipov.")
        else:
            logging.info("Žiadne nové tipy.")

if __name__ == "__main__":
    asyncio.run(main())
