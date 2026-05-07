#!/usr/bin/env python3
"""
FINÁLNA VERZIA: Futbalový tipper s ML validáciou a e-mailovou notifikáciou.
Obsahuje: Opravu banku, Cache systém, Poisson model, Kelly criterion a Gmail report.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import io
import logging
import joblib
import smtplib
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lightgbm import LGBMClassifier

# ================= KONFIGURÁCIA =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Kľúče a heslá zo Secrets / .env
API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

# Bezpečné načítanie BANKU
raw_bank = os.getenv('AKTUALNY_BANK', '1000')
BANK = float(raw_bank) if raw_bank and raw_bank.strip() else 1000.0

KELLY_FRAC = 0.20  # Konzervatívny Kelly
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

# ================= FUNKCIE =================

def posli_email(tipy: List[Dict]):
    """Odošle nájdené tipy na e-mail."""
    if not tipy or not GMAIL_USER or not GMAIL_PASSWORD:
        logging.info("E-mail sa neodosiela (žiadne tipy alebo chýbajúce prihlasovacie údaje).")
        return

    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_RECEIVER
    msg['Subject'] = f"Value Tipy: {datetime.now().strftime('%d.%m.%Y')}"

    body = "Nájdené nové výhodné stávky:\n\n"
    for t in tipy:
        body += f"⚽ {t['Zápas']} | Tip: {t['Tip']} | Kurz: {t['Kurz']}\n"
        body += f"   Edge: {t['Edge']*100:.1f}% | Vklad: {t['Vklad']}€\n"
        body += f"   Dátum: {t['Datum']}\n"
        body += "-" * 30 + "\n"

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("E-mail úspešne odoslaný.")
    except Exception as e:
        logging.error(f"Chyba pri odosielaní e-mailu: {e}")

async def fetch_csv(session: aiohttp.ClientSession, url: str, league: str) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{Path(url).stem}.csv"
    if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).seconds < 3600:
        return pd.read_csv(cache_file)
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                df = pd.read_csv(io.StringIO(await resp.text()))
                df.to_csv(cache_file, index=False)
                return df
    except: return None
    return None

def poisson_probs(lh: float, la: float) -> Dict[str, float]:
    matrix = np.outer(poisson.pmf(np.arange(10), lh), poisson.pmf(np.arange(10), la))
    matrix /= matrix.sum()
    return {
        '1': float(np.sum(np.tril(matrix, -1))),
        'X': float(np.sum(np.diag(matrix))),
        '2': float(np.sum(np.triu(matrix, 1))),
        'Over 2.5': float(np.sum(matrix[np.sum(np.indices((10, 10)), axis=0) >= 3]))
    }

def train_model() -> Optional[Tuple[LGBMClassifier, List[str]]]:
    if not Path(HISTORY_FILE).exists(): return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if len(df) < 50: return None
        df['win'] = (df['Vysledok'] == 'V').astype(int)
        features = ['Edge', 'Kurz', 'lh', 'la']
        model = LGBMClassifier(n_estimators=100, max_depth=3, verbosity=-1)
        model.fit(df[features], df['win'])
        return model, features
    except: return None

# ================= HLAVNÁ LOGIKA =================

async def process_league(session, name, cfg, model, features):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df_hist = await fetch_csv(session, url, name)
    if df_hist is None or len(df_hist) < 40: return []

    # Výpočet sily tímov
    df_c = df_hist.dropna(subset=['FTHG', 'FTAG'])
    avg_h, avg_a = df_c['FTHG'].mean(), df_c['FTAG'].mean()
    teams = {}
    for team in set(df_c['HomeTeam']):
        h_g = df_c[df_c['HomeTeam'] == team]
        a_g = df_c[df_c['AwayTeam'] == team]
        teams[team] = {
            'ah': h_g['FTHG'].mean() / avg_h if len(h_g) > 0 else 1,
            'dh': h_g['FTAG'].mean() / avg_a if len(h_g) > 0 else 1,
            'aa': a_g['FTAG'].mean() / avg_a if len(a_g) > 0 else 1,
            'da': a_g['FTHG'].mean() / avg_h if len(a_g) > 0 else 1
        }

    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/',
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as resp:
        if resp.status != 200: return []
        odds_data = await resp.json()

    bets, now = [], datetime.utcnow()
    for match in odds_data:
        h, a = match['home_team'], match['away_team']
        if h not in teams or a not in teams: continue
        
        m_time = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        if not (now <= m_time <= now + timedelta(hours=48)): continue

        lh = teams[h]['ah'] * teams[a]['da'] * avg_h + cfg['ha']
        la = teams[a]['aa'] * teams[h]['dh'] * avg_a
        probs = poisson_probs(lh, la)

        for bk in match.get('bookmakers', []):
            for mk in bk.get('markets', []):
                for out in mk.get('outcomes'):
                    lbl = '1' if out['name'] == h else ('2' if out['name'] == a else ('X' if out['name'] == 'Draw' else 'Over 2.5' if 'Over 2.5' in out['name'] else None))
                    if not lbl or lbl not in probs: continue

                    odds = out['price']
                    edge = (probs[lbl] * odds) - 1
                    
                    if 0.03 <= edge <= 0.25:
                        # ML Filter
                        if model:
                            row = pd.DataFrame([{'Edge': edge, 'Kurz': odds, 'lh': lh, 'la': la}])
                            if model.predict_proba(row[features])[0][1] < 0.52: continue

                        # Kelly
                        k = ((odds - 1) * probs[lbl] - (1 - probs[lbl])) / (odds - 1)
                        stake = max(0, min(k * KELLY_FRAC, 0.15)) * BANK
                        
                        if stake > 1:
                            bets.append({
                                'Datum': m_time.strftime('%d.%m.%Y %H:%M'),
                                'Zápas': f"{h} vs {a}", 'Tip': lbl, 'Kurz': odds,
                                'Edge': round(edge, 4), 'lh': round(lh, 2), 'la': round(la, 2),
                                'Vklad': round(stake, 2), 'Vysledok': ''
                            })
    return bets

async def main():
    if not API_ODDS_KEY: return logging.error("Chýba API kľúč!")
    
    async with aiohttp.ClientSession() as session:
        model_data = train_model()
        model, features = model_data if model_data else (None, None)
        
        tasks = [process_league(session, n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        all_bets = [b for sub in results for b in sub]

        if all_bets:
            # Uloženie do CSV
            new_df = pd.DataFrame(all_bets)
            if Path(HISTORY_FILE).exists():
                hist_df = pd.read_csv(HISTORY_FILE)
                final_df = pd.concat([hist_df, new_df]).drop_duplicates(subset=['Zápas', 'Tip', 'Datum'])
            else:
                final_df = new_df
            final_df.to_csv(HISTORY_FILE, index=False)
            
            # E-mailový report
            posli_email(all_bets)
            logging.info(f"Spracovaných {len(all_bets)} tipov.")
        else:
            logging.info("Žiadne nové tipy.")

if __name__ == "__main__":
    asyncio.run(main())
