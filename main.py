import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import csv
import logging
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"

LIGY_CONFIG = {
    # FUTBAL
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal', 'ha': 0.25},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal', 'ha': 0.28},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal', 'ha': 0.22},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal', 'ha': 0.32},
    # HOKEJ
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.15},
    '🏒 Česko Extraliga':  {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'sport': 'hokej', 'ha': 0.30},
    '🏒 Nemecko DEL':      {'csv': 'GER', 'api': 'icehockey_germany_del', 'sport': 'hokej', 'ha': 0.28},
    '🏒 Švédsko SHL':      {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'sport': 'hokej', 'ha': 0.25},
    '🏒 Fínsko Liiga':     {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'sport': 'hokej', 'ha': 0.22},
    # OSTATNÉ
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal', 'ha': 3.1},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis', 'ha': 0}
}

# --- 2. POMOCNÉ FUNKCIE ---

def uloz_do_historie(bets):
    file_exists = os.path.isfile(HISTORY_FILE)
    try:
        with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = ['Datum_Zapisu', 'Čas', 'Sport', 'Liga', 'Zápas', 'Tip', 'Kurz', 'Edge', 'Vklad', 'Skóre', 'Vysledok']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            for b in bets:
                row = b.copy()
                row['Datum_Zapisu'] = datetime.now().strftime('%d.%m.%Y')
                row['Vysledok'] = ""
                writer.writerow(row)
    except Exception as e: logging.error(f"Chyba ukladania CSV: {e}")

async def fetch_csv(session, liga, cfg):
    try:
        now = datetime.now()
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"
        elif cfg['sport'] == 'hokej':
            repo = "martineon/nhl-historical-data/master/data/nhl_results_2025.csv" if cfg['csv'] == 'NHL' else f"pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
            url = f"https://raw.githubusercontent.com/{repo}"
        elif cfg['sport'] == 'tenis':
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{now.year}.csv"
        elif cfg['sport'] == 'basketbal':
            url = "https://raw.githubusercontent.com/fivethirtyeight/nba-model/master/nba_elo.csv"
        
        async with session.get(url, timeout=10) as r:
            if r.status == 200: return liga, await r.read()
        return liga, None
    except: return liga, None

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'tenis':
            wins = df['winner_name'].value_counts()
            losses = df['loser_name'].value_counts()
            players = list(set(df['winner_name'].dropna()) | set(df['loser_name'].dropna()))
            stats = pd.DataFrame(index=players)
            stats['WinRate'] = [(wins.get(p, 0) + 1) / (wins.get(p, 0) + losses.get(p, 0) + 2) for p in players]
            return stats, 0, 0
        
        if 'home_team' in df.columns:
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        elif 'team1' in df.columns:
            df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG'})
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        h_stats = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'})
        a_stats = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'})
        
        stats = pd.DataFrame(index=list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
        stats['AH'] = h_stats['FTHG'] / avg_h
        stats['DH'] = h_stats['FTAG'] / avg_a
        stats['AA'] = a_stats['FTAG'] / avg_a
        stats['DA'] = a_stats['FTHG'] / avg_h
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def analyzuj():
    print(f"🚀 ŠTART ANALÝZY: {datetime.now().strftime('%H:%M')}")
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets = []
        now_utc = datetime.utcnow()
        limit_utc = now_utc + timedelta(hours=24)

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats, avg_h, avg_a = spracuj_stats(content, cfg)
            if stats is None: continue
            
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                   params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                m_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if not (now_utc <= m_time <= limit_utc): continue

                c1_m = process.extractOne(m['home_team'], stats.index)
                c2_m = process.extractOne(m['away_team'], stats.index)
                if not c1_m or not c2_m or c1_m[1] < 70: continue
                c1, c2 = c1_m[0], c2_m[0]
                
                probs, exp_score, lim = {}, "", 0
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    probs = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh = stats.at[c1,'AH'] * stats.at[c2,'DA'] * avg_h + cfg.get('ha', 0)
                    la = stats.at[c2,'AA'] * stats.at[c1,'DH'] * avg_a
                    exp_score = f"{round(lh, 1)}:{round(la, 1)}"
                    
                    p1 = sum(poisson.pmf(i, lh) * poisson.pmf(j, la) for i in range(15) for j in range(i))
                    px = sum(poisson.pmf(i, lh) * poisson.pmf(i, la) for i in range(15))
                    p2 = sum(poisson.pmf(i, lh) * poisson.pmf(j, la) for i in range(15) for j in range(i+1, 15))
                    
                    lim = 5.5 if cfg['sport'] == 'hokej' else 2.5
                    p_u = sum(poisson.pmf(i, lh) * poisson.pmf(j, la) for i in range(15) for j in range(15) if i+j < lim)
                    probs = {'1': p1, 'X': px, '2': p2, f'Over {lim}': 1-p_u, f'Under {lim}': p_u}

                match_best = {}
                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for out in mk['outcomes']:
                            if mk['key'] == 'totals':
                                if out.get('point') != lim: continue
                                lbl = f"{out['name']} {lim}"
                            else:
                                lbl = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X')
                            
                            prob, price = probs.get(lbl, 0), out['price']
                            edge = (prob * price) - 1
                            if 0.05 <= edge <= 0.45:
                                if lbl not in match_best or price > match_best[lbl]['Kurz']:
                                    v_val = (((price-1)*prob-(1-prob))/(price-1)) * 0.05
                                    vklad = round(min(max(0, v_val), 0.02)*AKTUALNY_BANK, 2)
                                    match_best[lbl] = {'Čas': m_time.strftime('%H:%M'), 'Liga': liga, 'Zápas': f"{m['home_team']} vs {m['away_team']}", 'Tip': lbl, 'Kurz': price, 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{vklad}€", 'Sport': cfg['sport'], 'Skóre': exp_score}
                all_bets.extend(match_best.values())

        if all_bets:
            all_bets = sorted(all_bets, key=lambda x: float(x['Edge'].replace('%','')), reverse=True)
            uloz_do_historie(all_bets)
            posli_email(all_bets)

def posli_email(bets):
    msg = MIMEMultipart()
    msg['From'], msg['To'] = GMAIL_USER, GMAIL_RECEIVER
    msg['Subject'] = f"📊 AI REPORT - {len(bets)} tipov"
    html = f"<h2>🔥 TOP AI TIPY ({datetime.now().strftime('%d.%m.')})</h2>"
    html += "<table border='1' style='border-collapse:collapse; width:100%; text-align:center; font-family:sans-serif;'>"
    html += "<tr style='background:#f2f2f2;'><th>Čas</th><th>S</th><th>Zápas</th><th>Exp.</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
    for b in bets[:35]:
        ik = "⚽" if b['Sport']=='futbal' else ("🎾" if b['Sport']=='tenis' else ("🏒" if b['Sport']=='hokej' else "🏀"))
        html += f"<tr><td>{b['Čas']}</td><td>{ik}</td><td>{b['Zápas']}</td><td>{b['Skóre']}</td><td><b>{b['Tip']}</b></td><td>{b['Kurz']}</td><td style='color:green;'>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
    html += "</table>"
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
        print("✅ Email úspešne odoslaný.")
    except Exception as e: print(f"❌ Chyba emailu: {e}")

if __name__ == "__main__":
    asyncio.run(analyzuj())
