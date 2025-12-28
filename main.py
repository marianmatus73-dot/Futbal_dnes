import requests
import pandas as pd
import smtplib
import os
import io
import time
from scipy.stats import poisson
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. NASTAVENIA ---
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

LIGY_CONFIG = {
    '🇬🇧 Premier League':   {'csv': 'E0',  'api': 'soccer_epl'},
    '🇪🇸 La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga'},
    '🇩🇪 Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga'},
    '🇮🇹 Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a'},
    '🇫🇷 Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one'},
    '🇺🇸 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl'}
}

MIN_VALUE_EDGE = 0.05
KELLY_FRACTION = 0.2
MAX_BANK_PCT = 0.02

# --- 2. POMOCNÉ FUNKCIE PRE DÁTUMY ---

def ziskaj_nhl_rok():
    """Vráti rok, ktorým končí aktuálna sezóna (pre CSV URL)."""
    now = datetime.now()
    # Ak je mesiac 10, 11, 12 (okt-dec), sezóna končí v nasledujúcom roku
    if now.month >= 10:
        return str(now.year + 1)
    # Ak je mesiac 1-9 (jan-sep), sezóna končí v tomto roku
    return str(now.year)

def ziskaj_futbal_sezonu():
    """Vráti kód sezóny pre football-data.co.uk (napr. '2526')."""
    now = datetime.now()
    # Futbalová sezóna sa láme v auguste (mesiac 8)
    if now.month >= 8:
        start_year = now.strftime('%y')
        end_year = (now.year + 1) % 100
        return f"{start_year}{end_year:02d}"
    else:
        start_year = (now.year - 1) % 100
        end_year = now.strftime('%y')
        return f"{start_year:02d}{end_year}"

# Globálne premenné pre sezóny (vypočítajú sa pri štarte)
AKTUALNA_SEZONA_FUTBAL = ziskaj_futbal_sezonu()
AKTUALNA_SEZONA_NHL = ziskaj_nhl_rok()

# --- 3. JADRO MODELU ---

def vypocitaj_kelly(pravdepodobnost, kurz):
    if kurz <= 1 or pravdepodobnost <= 0: return 0
    b = kurz - 1
    p = pravdepodobnost
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0: return 0
    return round(min(f_star * KELLY_FRACTION, MAX_BANK_PCT) * 100, 2)

def vypocitaj_silu_timov(df):
    if df.empty or len(df) < 10: return None, 0, 0

    avg_home = df['FTHG'].mean()
    avg_away = df['FTAG'].mean()

    home_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
    away_stats = df.groupby('AwayTeam')[['FTHG', 'FTAG']].mean()

    teams = pd.DataFrame(index=home_stats.index)
    teams['Att_Home'] = home_stats['FTHG'] / avg_home
    teams['Def_Home'] = home_stats['FTAG'] / avg_away

    teams_away = pd.DataFrame(index=away_stats.index)
    teams_away['Att_Away'] = away_stats['FTAG'] / avg_away
    teams_away['Def_Away'] = away_stats['FTHG'] / avg_home

    final_stats = teams.join(teams_away, how='outer').fillna(1.0)
    return final_stats, avg_home, avg_away

def predikuj_poisson(home, away, stats, avg_h, avg_a, sport='futbal'):
    try:
        h_att = stats.at[home, 'Att_Home']
        h_def = stats.at[home, 'Def_Home']
        a_att = stats.at[away, 'Att_Away']
        a_def = stats.at[away, 'Def_Away']
    except KeyError: return 0.0

    lamb_h = h_att * a_def * avg_h
    lamb_a = a_att * h_def * avg_a

    prob_h_win = 0
    max_g = 12 if sport == 'nhl' else 8

    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lamb_h) * poisson.pmf(y, lamb_a)
            if x > y: prob_h_win += p

    return prob_h_win

# --- 4. DÁTA A API ---

def stiahni_csv_data(liga_kod):
    if liga_kod == 'NHL':
        # Použije dynamický rok (napr. 2026)
        url = f"https://raw.githubusercontent.com/lbenz730/NHL_Draft_Analysis/master/data/nhl_scores_{AKTUALNA_SEZONA_NHL}.csv"
    else:
        # Použije dynamickú sezónu (napr. 2526)
        url = f"https://www.football-data.co.uk/mmz4281/{AKTUALNA_SEZONA_FUTBAL}/{liga_kod}.csv"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))

        if liga_kod == 'NHL':
            cols_map = {'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'}
            if not set(cols_map.keys()).issubset(df.columns): return pd.DataFrame()
            df = df.rename(columns=cols_map)

        return df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    except Exception as e:
        print(f"⚠️ Chyba dát ({liga_kod}): {e}")
        return pd.DataFrame()

def ziskaj_kurzy(sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else []
    except: return []

def fuzzy_match_team(api_name, csv_teams):
    if not csv_teams: return None
    match, score = process.extractOne(api_name, csv_teams)
    return match if score >= 75 else None

# --- 5. HLAVNÝ PROCES ---

def spustit_analyzu():
    report_data = []
    print(f"📅 Sezóna Futbal: {AKTUALNA_SEZONA_FUTBAL}, Sezóna NHL: {AKTUALNA_SEZONA_NHL}")

    for nazov_ligy, cfg in LIGY_CONFIG.items():
        print(f"🌍 Spracovávam: {nazov_ligy}")
        df = stiahni_csv_data(cfg['csv'])
        if df.empty: continue

        stats, avg_h, avg_a = vypocitaj_silu_timov(df)
        if stats is None: continue

        csv_tims = stats.index.tolist()
        matches = ziskaj_kurzy(cfg['api'])

        for m in matches:
            api_h, api_a = m['home_team'], m['away_team']
            csv_h = fuzzy_match_team(api_h, csv_tims)
            csv_a = fuzzy_match_team(api_a, csv_tims)

            if not csv_h or not csv_a: continue

            sport_type = 'nhl' if 'NHL' in nazov_ligy else 'futbal'
            ph = predikuj_poisson(csv_h, csv_a, stats, avg_h, avg_a, sport_type)

            try:
                best_p, best_b = 0, ""
                for bookie in m['bookmakers']:
                    market = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
                    if not market: continue
                    outc = next((o for o in market['outcomes'] if o['name'] == api_h), None)
                    if outc and outc['price'] > best_p:
                        best_p, best_b = outc['price'], bookie['title']

                if best_p > 1.0:
                    edge = (best_p * ph) - 1
                    if edge > MIN_VALUE_EDGE:
                        vklad = vypocitaj_kelly(ph, best_p)
                        if vklad > 0:
                            report_data.append({
                                'Liga': nazov_ligy,
                                'Zápas': f"{api_h} vs {api_a}",
                                'Tip': f"🏠 {api_h}",
                                'Model': f"{ph:.1%}",
                                'Kurz': best_p,
                                'Edge': f"{edge:.1%}",
                                'Vklad': f"<b>{vklad}%</b>",
                                'Sort': edge
                            })
                            print(f"   ✅ VALUE: {api_h} (Edge: {edge:.1%})")
            except: continue

    if report_data:
        odosli_email(report_data)
    else:
        print("😞 Žiadne value tipy.")

def odosli_email(data):
    df = pd.DataFrame(data).sort_values(by='Sort', ascending=False).drop(columns=['Sort'])

    style = """<style>table {border-collapse: collapse; width: 100%; font-family: sans-serif;}
               th {background: #2E7D32; color: white; padding: 10px;}
               td {border-bottom: 1px solid #ddd; padding: 10px; text-align: center;}
               tr:nth-child(even) {background: #f2f2f2;}</style>"""

    html = f"<html><head>{style}</head><body><h2>🎯 AI Value Report</h2>{df.to_html(index=False, escape=False, border=0)}</body></html>"

    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 AI Tipy: {len(data)} príležitostí"
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_RECEIVER
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
        print("📧 Email odoslaný!")
    except Exception as e:
        print(f"❌ Chyba emailu: {e}")

if __name__ == "__main__":
    spustit_analyzu()