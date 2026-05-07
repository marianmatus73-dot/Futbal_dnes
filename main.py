#!/usr/bin/env python3
"""
PRODUCTION FOOTBALL BETTING MODEL v6.0 (Ultimate)
-----------------------------------------------
Funkcie:
- AUTOMATICKÉ VYHODNOCOVANIE VÝSLEDKOV (Auto-Settlement)
- Inteligentné párovanie názvov tímov (Fuzzy Matching)
- Migrácia z historia_tipov.csv do SQLite (bets.db)
- LightGBM AI model trénovaný na tvojej histórii
- Poisson model s Time Decay a Kelly stakingom
- Email reporting
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
import difflib
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from scipy.stats import poisson
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

# ================= KONFIGURÁCIA =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

# Načítanie banku (ošetrenie prázdnych hodnôt)
bank_env = os.getenv('AKTUALNY_BANK', '1000').strip()
BANK = float(bank_env) if bank_env else 1000.0

# Parametre
KELLY_FRAC = 0.08      
MAX_STAKE_PCT = 0.03   
MIN_EDGE = 0.04        
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

# ================= POMOCNÉ FUNKCIE =================

def match_teams(name, list_of_teams):
    """Nájde najbližší názov tímu (rieši rozdiely v názvosloví)."""
    matches = difflib.get_close_matches(name, list_of_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

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

# ================= DATABÁZA A VYHODNOCOVANIE =================

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
    
    # Migrácia z CSV pri prvom spustení
    cur.execute("SELECT COUNT(*) FROM bets")
    if cur.fetchone()[0] == 0 and os.path.exists(HISTORY_CSV):
        logging.info("Migrujem dáta z CSV do SQLite...")
        df_old = pd.read_csv(HISTORY_CSV)
        for _, r in df_old.iterrows():
            cur.execute('''
                INSERT INTO bets (datum, zapas, tip, kurz, edge, lh, la, vklad, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(r.get('Datum')), str(r.get('Zápas')), str(r.get('Tip')), 
                  float(r.get('Kurz', 0)), float(r.get('Edge', 0)),
                  float(r.get('lh', 1.5)), float(r.get('la', 1.5)), 
                  float(r.get('Vklad', 0)), str(r.get('Vysledok', ''))))
    conn.commit()
    conn.close()

async def settle_results(session):
    """Automaticky priradí V/P k odohraným zápasom v databáze."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT id, zapas, tip FROM bets WHERE result = '' OR result IS NULL")
    unsettled = cur.fetchall()
    
    if not unsettled:
        conn.close()
        return

    logging.info(f"Kontrolujem výsledky pre {len(unsettled)} zápasov...")

    for league_name, cfg in LIGY.items():
        url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
        df_res = await fetch_csv(session, url)
        if df_res is None: continue
        
        all_csv_teams = set(df_res['HomeTeam'].unique()) | set(df_res['AwayTeam'].unique())

        for row_id, zapas, tip in unsettled:
            try:
                if " vs " not in zapas: continue
                h_name, a_name = zapas.split(" vs ")
                csv_h = match_teams(h_name, all_csv_teams)
                csv_a = match_teams(a_name, all_csv_teams)
                
                if not csv_h or not csv_a: continue
                
                match_row = df_res[(df_res['HomeTeam'] == csv_h) & (df_res['AwayTeam'] == csv_a)]
                if not match_row.empty:
                    fthg, ftag, res_ftr = match_row.iloc[0]['FTHG'], match_row.iloc[0]['FTAG'], match_row.iloc[0]['FTR']
                    
                    is_win = False
                    if tip == '1' and res_ftr == 'H': is_win = True
                    elif tip == 'X' and res_ftr == 'D': is_win = True
                    elif tip == '2' and res_ftr == 'A': is_win = True
                    elif tip == 'Over 2.5' and (fthg + ftag) > 2.5: is_win = True
                    
                    cur.execute("UPDATE bets SET result = ? WHERE id = ?", ('V' if is_win else 'P', row_id))
                    logging.info(f"✅ Vyhodnotené: {zapas} -> {'V' if is_win else 'P'}")
            except: continue
    conn.commit()
    conn.close()

# ================= AI A POISSON =================

def train_model():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM bets WHERE result IN ('V', 'P')", conn)
        if len(df) < 100: return None, None
        df['win'] = (df['result'] == 'V').astype(int)
        features = ['edge', 'kurz', 'lh', 'la']
        model = CalibratedClassifierCV(LGBMClassifier(n_estimators=100, verbosity=-1), method='isotonic', cv=3)
        model.fit(df[features], df['win'])
        return model, features
    except: return None, None
    finally: conn.close()

def poisson_probs(lh, la):
    matrix = np.outer(poisson.pmf(np.arange(10), lh), poisson.pmf(np.arange(10), la))
    matrix /= matrix.sum()
    return {
        '1': float(np.sum(np.tril(matrix, -1))),
        'X': float(np.sum(np.diag(matrix))),
        '2': float(np.sum(np.triu(matrix, 1))),
        'Over 2.5': float(np.sum(matrix[np.sum(np.indices((10, 10)), axis=0) >= 3]))
    }

# ================= CORE LOGIC =================

async def process_league(session, name, cfg, model, features):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df_hist = await fetch_csv(session, url)
    if df_hist is None or len(df_hist) < 40: return []

    df_hist = df_hist.dropna(subset=['FTHG', 'FTAG'])
    avg_h, avg_a = df_hist['FTHG'].mean(), df_hist['FTAG'].mean()
    teams = {t: {'ah': df_hist[df_hist['HomeTeam']==t]['FTHG'].mean()/avg_h, 
                 'dh': df_hist[df_hist['HomeTeam']==t]['FTAG'].mean()/avg_a,
                 'aa': df_hist[df_hist['AwayTeam']==t]['FTAG'].mean()/avg_a,
                 'da': df_hist[df_hist['AwayTeam']==t]['FTHG'].mean()/avg_h} 
             for t in set(df_hist['HomeTeam'])}

    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/',
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as resp:
        if resp.status != 200: return []
        odds_data = await resp.json()

    bets, now = [], datetime.utcnow()
    for m in odds_data:
        h, a = m['home_team'], m['away_team']
        if h not in teams or a not in teams: continue
        m_t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        if not (now <= m_t <= now + timedelta(hours=48)): continue

        lh, la = teams[h]['ah']*teams[a]['da']*avg_h + cfg['ha'], teams[a]['aa']*teams[h]['dh']*avg_a
        probs = poisson_probs(lh, la)

        for bk in m.get('bookmakers', []):
            for mk in bk.get('markets', []):
                for out in mk.get('outcomes'):
                    lbl = '1' if out['name'] == h else ('2' if out['name'] == a else ('X' if out['name'] == 'Draw' else 'Over 2.5' if 'Over 2.5' in out['name'] else None))
                    if not lbl: continue
                    odds, edge = out['price'], (probs[lbl]*out['price'])-1
                    if MIN_EDGE <= edge <= 0.25:
                        if model:
                            if model.predict_proba(pd.DataFrame([{'edge':edge,'kurz':odds,'lh':lh,'la':la}]))[0][1] < 0.52: continue
                        
                        kelly = ((odds-1)*probs[lbl]-(1-probs[lbl]))/(odds-1)
                        stake = round(min(max(0, kelly*KELLY_FRAC), MAX_STAKE_PCT)*BANK, 2)
                        if stake >= 1.0:
                            bets.append({'Datum': m_t.strftime('%d.%m.%Y %H:%M'), 'Zapas': f"{h} vs {a}", 'Tip': lbl, 'Kurz': odds, 'Edge': round(edge,4), 'lh': round(lh,2), 'la': round(la,2), 'Vklad': stake, 'Bookmaker': bk.get('title')})
    return bets

def send_report(bets):
    if not bets or not GMAIL_USER: return
    msg = MIMEMultipart()
    msg['Subject'] = f"Value Bets - {datetime.now().strftime('%d.%m %H:%M')}"
    msg.attach(MIMEText("\n".join([f"⚽ {b['Zapas']} | {b['Tip']} @ {b['Kurz']} | Edge: {b['Edge']*100:.1f}% | {b['Vklad']}€" for b in bets]), 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
    except: pass

async def main():
    init_db()
    async with aiohttp.ClientSession() as session:
        await settle_results(session) # 1. Vyhodnotenie odohraných
        model, features = train_model() # 2. Tréning AI na nových dátach
        results = await asyncio.gather(*[process_league(session, n, c, model, features) for n, c in LIGY.items()])
        all_bets = [b for sub in results for b in sub]
        if all_bets:
            conn = sqlite3.connect(DB_FILE)
            for b in all_bets:
                conn.execute("INSERT INTO bets (datum, zapas, tip, kurz, edge, lh, la, vklad, bookmaker, result) VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (b['Datum'], b['Zapas'], b['Tip'], b['Kurz'], b['Edge'], b['lh'], b['la'], b['Vklad'], b['Bookmaker'], ''))
            conn.commit()
            conn.close()
            send_report(all_bets)
            logging.info(f"Nájdených {len(all_bets)} nových tipov.")

if __name__ == "__main__":
    asyncio.run(main())
