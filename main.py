import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import logging
from scipy.stats import poisson, norm
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
            players = set(df['winner_name'].dropna()) | set(df['loser_name'].dropna())
            stats = pd.DataFrame(index=list(players))
            stats['WinRate'] = [(wins.get(p, 0) + 1) / (wins.get(p, 0) + losses.get(p, 0) + 2) for p in players]
            return stats, 0, 0
        
        # Ostatné športy - zjednodušený výpočet
        if 'team1' in df.columns: df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG'})
        elif 'home_team' in df.columns: df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        h = df.groupby('HomeTeam').apply(lambda x: pd.Series({'AH': x['FTHG'].mean()/avg_h, 'DH': x['FTAG'].mean()/avg_a}), include_groups=False)
        a = df.groupby('AwayTeam').apply(lambda x: pd.Series({'AA': x['FTAG'].mean()/avg_a, 'DA': x['FTHG'].mean()/avg_h}), include_groups=False)
        return h.join(a, how='outer', lsuffix='_h', rsuffix='_a').fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def analyzuj():
    print(f"🚀 ŠTART ANALÝZY: {datetime.now().strftime('%H:%M')}")
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets = []
        now_utc = datetime.utcnow()
        limit_utc = now_utc + timedelta(hours=24) # Pozeráme 24 hodín dopredu

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats_data = spracuj_stats(content, cfg)
            if stats_data[0] is None: continue
            stats, avg_h, avg_a = stats_data
            
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey': API_ODDS_KEY, 'regions': 'eu'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                m_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if not (now_utc <= m_time <= limit_utc): continue # Iba najbližších 24h

                c1, c2 = process.extractOne(m['home_team'], stats.index)[0], process.extractOne(m['away_team'], stats.index)[0]
                
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh, la = stats.at[c1,'AH']*stats.at[c2,'DA_h']*avg_h, stats.at[c2,'AA_a']*stats.at[c1,'DH_h']*avg_a
                    if cfg['sport'] == 'futbal': lh += cfg['ha']
                    p = {'1': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for x in range(12) for y in range(x)),
                         'X': sum(poisson.pmf(x,lh)*poisson.pmf(x,la) for x in range(12)),
                         '2': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for y in range(12) for x in range(y))}

                match_best = {}
                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        if mk['key'] != 'h2h': continue
                        for out in mk['outcomes']:
                            lbl = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X')
                            prob, price = p.get(lbl, 0), out['price']
                            edge = (prob * price) - 1
                            if 0.03 <= edge <= 0.40:
                                if lbl not in match_best or price > match_best[lbl]['Kurz']:
                                    vklad = min(max(0, (((price-1)*prob-(1-prob))/(price-1)) * 0.05), 0.02)
                                    match_best[lbl] = {'Čas': m_time.strftime('%H:%M'), 'Liga': liga, 'Zápas': f"{m['home_team']} vs {m['away_team']}", 'Tip': out['name'], 'Kurz': price, 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{round(vklad*AKTUALNY_BANK,2)}€", 'Sport': cfg['sport']}
                all_bets.extend(match_best.values())

        # ODOSIELANIE
        msg = MIMEMultipart()
        msg['From'], msg['To'] = GMAIL_USER, GMAIL_RECEIVER
        
        if all_bets:
            all_bets = sorted(all_bets, key=lambda x: float(x['Edge'].replace('%','')), reverse=True)
            html = "<h2>🔥 TOP AI TIPY (Najbližších 24h)</h2><table border='1' style='border-collapse:collapse; width:100%; text-align:center;'>"
            html += "<tr style='background:#f2f2f2;'><th>Čas</th><th>Šport</th><th>Liga</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
            for b in all_bets[:25]:
                ikona = "⚽" if b['Sport'] == 'futbal' else ("🎾" if b['Sport'] == 'tenis' else ("🏒" if b['Sport'] == 'hokej' else "🏀"))
                html += f"<tr><td>{b['Čas']}</td><td>{ikona}</td><td>{b['Liga']}</td><td>{b['Zápas']}</td><td><b>{b['Tip']}</b></td><td>{b['Kurz']}</td><td style='color:green;'>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
            html += "</table>"
            msg['Subject'] = f"📊 AI REPORT - {len(all_bets)} tipov"
        else:
            html = "<p>Analýza prebehla úspešne, ale nenašli sa žiadne tipy s Edge nad 3%.</p>"
            msg['Subject'] = "📊 AI REPORT - Žiadne tipy"
            print("ℹ️ Žiadne tipy s hodnotou.")

        msg.attach(MIMEText(html, 'html'))
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as s:
                s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
            print("✅ Email odoslaný.")
        except Exception as e: print(f"❌ Chyba emailu: {e}")

if __name__ == "__main__":
    asyncio.run(analyzuj())
