import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import logging
from scipy.stats import poisson, norm
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"

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

def fuzzy_match_team(name, choices):
    if choices is None or len(choices) == 0: return None
    match, score = process.extractOne(name, choices)
    return match if score >= 80 else None

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
            urls = ["https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_2025.csv",
                    "https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_2024.csv"]
        elif cfg['sport'] == 'basketbal':
            urls = ["https://raw.githubusercontent.com/fivethirtyeight/nba-model/master/nba_elo.csv"]

        for url in urls:
            async with session.get(url, timeout=10) as r:
                if r.status == 200:
                    print(f"✅ Načítané dáta pre {liga}")
                    return liga, await r.read()
        return liga, None
    except:
        return liga, None

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'tenis':
            wins = df['winner_name'].value_counts()
            losses = df['loser_name'].value_counts()
            players = set(df['winner_name'].dropna()) | set(df['loser_name'].dropna())
            stats = pd.DataFrame(index=list(players))
            stats['WinRate'] = [(wins.get(p, 0) + 1) / (wins.get(p, 0) + losses.get(p, 0) + 2) for p in players]
            return stats, 0, 0
        
        # Ostatné športy
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') if 'Date' in df.columns else datetime.now()
        if cfg['sport'] == 'basketbal':
            df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG'})
        elif cfg['sport'] == 'hokej':
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        h = df.groupby('HomeTeam').apply(lambda x: pd.Series({'AH': x['FTHG'].mean()/avg_h, 'DH': x['FTAG'].mean()/avg_a}), include_groups=False)
        a = df.groupby('AwayTeam').apply(lambda x: pd.Series({'AA': x['FTAG'].mean()/avg_a, 'DA': x['FTHG'].mean()/avg_h}), include_groups=False)
        return h.join(a, how='outer', lsuffix='_h', rsuffix='_a').fillna(1.0), avg_h, avg_a
    except:
        return None, 0, 0

async def analyzuj():
    print(f"🚀 ŠTART ANALÝZY: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets = []
        now_utc = datetime.utcnow()

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats_data = spracuj_stats(content, cfg)
            if stats_data[0] is None: continue
            stats, avg_h, avg_a = stats_data
            
            params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h'}
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params=params) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                match_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if match_time.date() != now_utc.date(): continue

                c1, c2 = fuzzy_match_team(m['home_team'], stats.index), fuzzy_match_team(m['away_team'], stats.index)
                if not (c1 and c2): continue
                
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*avg_h,
