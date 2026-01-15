import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
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

LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal', 'ha': 0.25},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal', 'ha': 0.28},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal', 'ha': 0.22},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal', 'ha': 0.32},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.15},
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal', 'ha': 3.1},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis', 'ha': 0}
}

# --- 2. POMOCNÉ FUNKCIE ---

async def fetch_csv(session, liga, cfg):
    try:
        now = datetime.now()
        urls = []
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            urls = [f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"]
        elif cfg['sport'] == 'tenis':
            urls = [f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{now.year}.csv",
                    f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{now.year-1}.csv"]
        elif cfg['sport'] == 'hokej':
            urls = ["https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_2025.csv"]
        elif cfg['sport'] == 'basketbal':
            urls = ["https://raw.githubusercontent.com/fivethirtyeight/nba-model/master/nba_elo.csv"]

        for url in urls:
            async with session.get(url, timeout=10) as r:
                if r.status == 200: return liga, await r.read()
        return liga, None
    except: return liga, None

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'tenis':
            wins, losses = df['winner_name'].value_counts(), df['loser_name'].value_counts()
            players = list(set(df['winner_name'].dropna()) | set(df['loser_name'].dropna()))
            stats = pd.DataFrame(index=players)
            stats['WinRate'] = [(wins.get(p, 0) + 1) / (wins.get(p, 0) + losses.get(p, 0) + 2) for p in players]
            return stats, 0, 0
        
        if cfg['sport'] == 'basketbal':
            df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG'})
        elif 'home_team' in df.columns:
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        h_stats = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'})
        a_stats = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'})
        
        all_teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        stats = pd.DataFrame(index=all_teams)
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
            
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey': API_ODDS_KEY, 'regions': 'eu'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                m_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if not (now_utc <= m_time <= limit_utc): continue

                c1_match = process.extractOne(m['home_team'], stats.index)
                c2_match = process.extractOne(m['away_team'], stats.index)
                if not c1_match or not c2_match or c1_match[1] < 70: continue
                c1, c2 = c1_match[0], c2_match[0]
                
                exp_score = ""
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh = stats.at[c1,'AH'] * stats.at[c2,'DA'] * avg_h
                    la = stats.at[c2,'AA'] * stats.at[c1,'DH'] * avg_a
                    if cfg['sport'] == 'futbal': lh += cfg['ha']
                    exp_score = f"{round(lh, 1)}:{round(la, 1)}"
                    
                    p1 = sum(poisson.pmf(i, lh) * poisson.pmf(j, la) for i in range(15) for j in range(i))
                    px = sum(poisson.pmf(i, lh) * poisson.pmf(i, la) for i in range(15))
                    p2 = sum(poisson.pmf(i, lh) * poisson.pmf(j, la) for i in range(15) for j in range(i+1, 15))
                    p = {'1': p1, 'X': px, '2': p2}

                match_best = {}
                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        if mk['key'] != 'h2h': continue
                        for out in mk['outcomes']:
                            lbl = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X')
                            prob, price = p.get(lbl, 0), out['price']
                            edge = (prob * price) - 1
                            if 0.03 <= edge <= 0.45:
                                if lbl not in match_best or price > match_best[lbl]['Kurz']:
                                    vklad_val = (((price-1)*prob-(1-prob))/(price-1)) * 0.05
                                    vklad = min(max(0, vklad_val), 0.02)
                                    match_best[lbl] = {
                                        'Čas': m_time.strftime('%H:%M'), 'Liga': liga, 
                                        'Zápas': f"{m['home_team']} vs {m['away_team']}", 'Tip': out['name'], 
                                        'Kurz': price, 'Edge': f"{round(edge*100,1)}%", 
                                        'Vklad': f"{round(vklad*AKTUALNY_BANK,2)}€", 'Sport': cfg['sport'],
                                        'Skóre': exp_score
                                    }
                all_bets.extend(match_best.values())

        # ODOSIELANIE
        msg = MIMEMultipart()
        msg['From'], msg['To'] = GMAIL_USER, GMAIL_RECEIVER
        if all_bets:
            all_bets = sorted(all_bets, key=lambda x: float(x['Edge'].replace('%','')), reverse=True)
            html = f"<h2>🔥 TOP AI TIPY ({datetime.now().strftime('%d.%m.')})</h2><table border='1' style='border-collapse:collapse; width:100%; text-align:center; font-family: sans-serif;'>"
            html += "<tr style='background:#f2f2f2;'><th>Čas</th><th>Šport</th><th>Liga</th><th>Zápas</th><th>Exp. Skóre</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
            for b in all_bets[:30]:
                ikona = "⚽" if b['Sport'] == 'futbal' else ("🎾" if b['Sport'] == 'tenis' else ("🏒" if b['Sport'] == 'hokej' else "🏀"))
                html += f"<tr><td>{b['Čas']}</td><td>{ikona}</td><td>{b['Liga']}</td><td>{b['Zápas']}</td><td>{b['Skóre']}</td><td><b>{b['Tip']}</b></td><td>{b['Kurz']}</td><td style='color:green;'>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
            html += "</table>"
            msg['Subject'] = f"📊 AI REPORT - {len(all_bets)} tipov"
        else:
            html = "<p>Žiadne tipy s Edge > 3% na najbližších 24h.</p>"
            msg['Subject'] = "📊 AI REPORT - Žiadne tipy"

        msg.attach(MIMEText(html, 'html'))
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as s:
                s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
            print(f"✅ Email odoslaný ({len(all_bets)} tipov).")
        except Exception as e: print(f"❌ Chyba emailu: {e}")

if __name__ == "__main__":
    asyncio.run(analyzuj())
