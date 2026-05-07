#!/usr/bin/env python3
"""
PRODUCTION FOOTBALL BETTING MODEL v5.0 (Unified)
-----------------------------------------------
Funkcie:
- Automatická migrácia z historia_tipov.csv do SQLite
- LightGBM ML model s kalibráciou pravdepodobnosti
- Poisson model s Time Decay (váha podľa veku zápasu)
- Kelly Criterion staking & Risk Management
- Email reporting & SQLite tracking
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import io
import logging
import sqlite3
import smtplib
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from scipy.stats import poisson
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

# ================= CONFIG =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

# Robustné načítanie banku
bank_env = os.getenv('AKTUALNY_BANK', '1000').strip()
BANK = float(bank_env) if bank_env else 1000.0

# Parametre stratégie
KELLY_FRAC = 0.08      # Štvrtinový Kelly (bezpečnejší)
MAX_STAKE_PCT = 0.03   # Max 3% banku na jeden tip
MIN_EDGE = 0.04        # Minimálna hodnota 4%
DB_FILE = 'bets.db'
HISTORY_CSV = 'historia_tipov.csv'
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

LIGY = {
    'Premier League': {'csv': 'E0', 'api': 'soccer_epl', 'ha': 0.35},
    'La Liga': {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'ha': 0.38},
    'Bundesliga': {'csv': 'D1', 'api': 'soccer_germany_bundesliga', 'ha': 0.40},
    'Serie A': {'csv': 'I1', 'api': 'soccer_italy_serie_a', 'ha': 0.30},
    'Ligue 1': {'csv': 'F1', 'api': 'soccer_france_ligue_one', 'ha': 0.32},
    'Eredivisie': {'csv': 'N1', 'api': 'soccer_netherlands_eredivisie', 'ha': 0.42},
    'Championship': {'csv': 'E1', 'api': 'soccer_efl_champ', 'ha': 0.28},
}

# ================= DATABASE & MIGRATION =================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datum TEXT, zapas TEXT, tip TEXT, kurz REAL,
            edge REAL, lh REAL, la REAL, vklad REAL,
            bookmaker TEXT, result TEXT
        )
    ''')
    
    # Skontrolujeme, či je DB prázdna. Ak áno, skúsime migrovať z CSV.
    cur.execute("SELECT COUNT(*) FROM bets")
    if cur.fetchone()[0] == 0 and os.path.exists(HISTORY_CSV):
        logging.info(f"Migrujem dáta z {HISTORY_CSV} do SQLite...")
        try:
            df_old = pd.read_csv(HISTORY_CSV)
            for _, r in df_old.iterrows():
                # Mapovanie stĺpcov podľa tvojho nahratého súboru
                cur.execute('''
                    INSERT INTO bets (datum, zapas, tip, kurz, edge, lh, la, vklad, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (str(r.get('Datum')), str(r.get('Zápas')), str(r.get('Tip')), 
                      float(r.get('Kurz', 0)), float(r.get('Edge', 0))/100 if r.get('Edge', 0) > 1 else float(r.get('Edge', 0)),
                      float(r.get('lh', 1.5)), float(r.get('la', 1.5)), 
                      float(r.get('Vklad', 0)), str(r.get('Vysledok', ''))))
            logging.info("Migrácia úspešne dokončená.")
        except Exception as e:
            logging.error(f"Chyba pri migrácii CSV: {e}")

    conn.commit()
    conn.close()

# ================= AI MODEL =================

def train_model():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM bets WHERE result IN ('V', 'P')", conn)
        if len(df) < 80:
            return None, None
        
        df['win'] = (df['result'] == 'V').astype(int)
        features = ['edge', 'kurz', 'lh', 'la']
        
        base_model = LGBMClassifier(n_estimators=100, max_depth=3, verbosity=-1)
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(df[features], df['win'])
        
        logging.info(f"AI Model natrénovaný na {len(df)} zápasoch.")
        return model, features
    except Exception as e:
        logging.warning(f"Model sa nepodarilo natrénovať: {e}")
        return None, None
    finally:
        conn.close()

# ================= ANALYTICS =================

def poisson_probs(lh, la):
    # Korekcia na remízy a limity gólov
    matrix = np.outer(poisson.pmf(np.arange(10), lh), poisson.pmf(np.arange(10), la))
    matrix /= matrix.sum()
    return {
        '1': float(np.sum(np.tril(matrix, -1))),
        'X': float(np.sum(np.diag(matrix))),
        '2': float(np.sum(np.triu(matrix, 1))),
        'Over 2.5': float(np.sum(matrix[np.sum(np.indices((10, 10)), axis=0) >= 3]))
    }

async def fetch_csv(session, url):
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

# ================= MAIN LOGIC =================

async def process_league(session, name, cfg, model, features):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df_hist = await fetch_csv(session, url)
    if df_hist is None or len(df_hist) < 40: return []

    # Time-decay weighting (novšie zápasy majú väčšiu váhu)
    df_hist = df_hist.dropna(subset=['FTHG', 'FTAG'])
    avg_h, avg_a = df_hist['FTHG'].mean(), df_hist['FTAG'].mean()
    
    teams = {}
    for team in set(df_hist['HomeTeam']):
        h_g = df_hist[df_hist['HomeTeam'] == team]
        a_g = df_hist[df_hist['AwayTeam'] == team]
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
            bk_title = bk.get('title', 'Unknown')
            for mk in bk.get('markets', []):
                for out in mk.get('outcomes'):
                    lbl = '1' if out['name'] == h else ('2' if out['name'] == a else ('X' if out['name'] == 'Draw' else 'Over 2.5' if 'Over 2.5' in out['name'] else None))
                    if not lbl: continue

                    odds, edge = out['price'], (probs[lbl] * out['price']) - 1
                    if MIN_EDGE <= edge <= 0.25:
                        # ML Filter: Ak máme model, musí potvrdiť pravdepodobnosť
                        if model:
                            row = pd.DataFrame([{'edge': edge, 'kurz': odds, 'lh': lh, 'la': la}])
                            if model.predict_proba(row[features])[0][1] < 0.52: continue

                        # Kelly Criterion Staking
                        kelly = ((odds - 1) * probs[lbl] - (1 - probs[lbl])) / (odds - 1)
                        stake_pct = min(max(0, kelly * KELLY_FRAC), MAX_STAKE_PCT)
                        stake = round(stake_pct * BANK, 2)
                        
                        if stake >= 1.0:
                            bets.append({
                                'Datum': m_time.strftime('%d.%m.%Y %H:%M'),
                                'Zapas': f"{h} vs {a}", 'Tip': lbl, 'Kurz': odds,
                                'Edge': round(edge, 4), 'lh': round(lh, 2), 'la': round(la, 2),
                                'Vklad': stake, 'Bookmaker': bk_title
                            })
    return bets

def send_report(bets):
    if not bets or not GMAIL_USER or not GMAIL_PASSWORD: return
    msg = MIMEMultipart()
    msg['Subject'] = f"Value Bets Report - {datetime.now().strftime('%d.%m %H:%M')}"
    body = "\n".join([f"⚽ {b['Zapas']} | {b['Tip']} @ {b['Kurz']} | Edge: {b['Edge']*100:.1f}% | Vklad: {b['Vklad']}€ ({b['Bookmaker']})" for b in bets])
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Email odoslaný.")
    except Exception as e: logging.error(f"Email error: {e}")

async def main():
    init_db()
    model, features = train_model()
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_league(session, n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        all_bets = [b for sub in results for b in sub]

        if all_bets:
            # Uloženie do DB
            conn = sqlite3.connect(DB_FILE)
            for b in all_bets:
                conn.execute("INSERT INTO bets (datum, zapas, tip, kurz, edge, lh, la, vklad, bookmaker) VALUES (?,?,?,?,?,?,?,?,?)",
                             (b['Datum'], b['Zapas'], b['Tip'], b['Kurz'], b['Edge'], b['lh'], b['la'], b['Vklad'], b['Bookmaker']))
            conn.commit()
            conn.close()
            
            send_report(all_bets)
            logging.info(f"Nájdených {len(all_bets)} nových tipov.")
        else:
            logging.info("Nenašli sa žiadne nové výhodné tipy.")

if __name__ == "__main__":
    asyncio.run(main())
