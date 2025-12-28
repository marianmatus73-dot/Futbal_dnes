import requests
import pandas as pd
import smtplib
import os
import io
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

MIN_VALUE_EDGE = 0.05  # Minimálna výhoda 5%
KELLY_FRACTION = 0.2
MAX_BANK_PCT = 0.02

# --- 2. POMOCNÉ FUNKCIE ---
def ziskaj_nhl_rok():
    now = datetime.now()
    return str(now.year + 1) if now.month >= 10 else str(now.year)

def ziskaj_futbal_sezonu():
    now = datetime.now()
    if now.month >= 8:
        return f"{now.strftime('%y')}{(now.year + 1) % 100:02d}"
    return f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"

AKTUALNA_SEZONA_FUTBAL = ziskaj_futbal_sezonu()
AKTUALNA_SEZONA_NHL = ziskaj_nhl_rok()

# --- 3. JADRO MODELU ---
def vypocitaj_kelly(pravdepodobnost, kurz):
    if kurz <= 1 or pravdepodobnost <= 0: return 0
    f_star = ((kurz - 1) * pravdepodobnost - (1 - pravdepodobnost)) / (kurz - 1)
    return round(max(0, min(f_star * KELLY_FRACTION, MAX_BANK_PCT)) * 100, 2)

def vypocitaj_silu_timov(df):
    if df.empty or len(df) < 10: return None, 0, 0
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
    home_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
    away_stats = df.groupby('AwayTeam')[['FTAG', 'FTHG']].mean()
    teams = pd.DataFrame(index=home_stats.index)
    teams['Att_Home'] = home_stats['FTHG'] / avg_h
    teams['Def_Home'] = home_stats['FTAG'] / avg_a
    teams['Att_Away'] = away_stats['FTAG'] / avg_a
    teams['Def_Away'] = away_stats['FTHG'] / avg_h
    return teams.fillna(1.0), avg_h, avg_a

def predikuj_poisson(home, away, stats, avg_h, avg_a, sport='futbal'):
    try:
        h_att, h_def = stats.at[home, 'Att_Home'], stats.at[home, 'Def_Home']
        a_att, a_def = stats.at[away, 'Att_Away'], stats.at[away, 'Def_Away']
    except: return 0.0, 0.0
    lamb_h, lamb_a = h_att * a_def * avg_h, a_att * h_def * avg_a
    p_win, p_over = 0, 0
    limit = 5.5 if sport == 'nhl' else 2.5
    for x in range(12):
        for y in range(12):
            p = poisson.pmf(x, lamb_h) * poisson.pmf(y, lamb_a)
            if x > y: p_win += p
            if (x + y) > limit: p_over += p
    return p_win, p_over

# --- 4. HLAVNÝ PROCES ---
def spustit_analyzu():
    report_data = []
    for nazov_ligy, cfg in LIGY_CONFIG.items():
        print(f"🌍 Analyzujem: {nazov_ligy}")
        df = stiahni_csv_data(cfg['csv'])
        stats, avg_h, avg_a = vypocitaj_silu_timov(df)
        if stats is None: continue
        
        matches = ziskaj_kurzy(cfg['api'])
        for m in matches:
            csv_h = process.extractOne(m['home_team'], stats.index.tolist())[0]
            csv_a = process.extractOne(m['away_team'], stats.index.tolist())[0]
            sport_type = 'nhl' if 'NHL' in nazov_ligy else 'futbal'
            ph, po = predikuj_poisson(csv_h, csv_a, stats, avg_h, avg_a, sport_type)
            
            for bookie in m['bookmakers']:
                # Kontrola výhry domáceho tímu
                h2h = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
                if h2h:
                    out = next((o for o in h2h['outcomes'] if o['name'] == m['home_team']), None)
                    if out and (out['price'] * ph - 1) > MIN_VALUE_EDGE:
                        report_data.append({'Liga': nazov_ligy, 'Zápas': f"{m['home_team']}-{m['away_team']}", 'Tip': f"🏠 {m['home_team']}", 'Model': f"{ph:.1%}", 'Kurz': out['price'], 'Edge': f"{(out['price']*ph-1):.1%}", 'Vklad': f"<b>{vypocitaj_kelly(ph, out['price'])}%</b>", 'Sort': out['price']*ph-1})

                # Kontrola Over gólov (2.5 futbal / 5.5 NHL)
                totals = next((mk for mk in bookie['markets'] if mk['key'] == 'totals'), None)
                if totals:
                    line = 5.5 if sport_type == 'nhl' else 2.5
                    out = next((o for o in totals['outcomes'] if o['name'] == 'Over' and o['point'] == line), None)
                    if out and (out['price'] * po - 1) > MIN_VALUE_EDGE:
                        report_data.append({'Liga': nazov_ligy, 'Zápas': f"{m['home_team']}-{m['away_team']}", 'Tip': f"🔥 Over {line}", 'Model': f"{po:.1%}", 'Kurz': out['price'], 'Edge': f"{(out['price']*po-1):.1%}", 'Vklad': f"<b>{vypocitaj_kelly(po, out['price'])}%</b>", 'Sort': out['price']*po-1})

    if report_data: odosli_email(report_data)
    else: print("😞 Žiadne value tipy.")

def stiahni_csv_data(liga_kod):
    url = f"https://www.football-data.co.uk/mmz4281/{AKTUALNA_SEZONA_FUTBAL}/{liga_kod}.csv" if liga_kod != 'NHL' else \
          f"https://raw.githubusercontent.com/lbenz730/NHL_Draft_Analysis/master/data/nhl_scores_{AKTUALNA_SEZONA_NHL}.csv"
    try:
        r = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))
        if liga_kod == 'NHL': df = df.rename(columns={'home_team':'HomeTeam','away_team':'AwayTeam','home_goals':'FTHG','away_goals':'FTAG'})
        return df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    except: return pd.DataFrame()

def ziskaj_kurzy(sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals', 'oddsFormat': 'decimal'}
    try:
        r = requests.get(url, params=params)
        return r.json() if r.status_code == 200 else []
    except: return []

def odosli_email(data):
    df = pd.DataFrame(data).sort_values(by='Sort', ascending=False).drop(columns=['Sort'])
    style = "<style>table {border-collapse: collapse; width: 100%; font-family: sans-serif;} th {background: #2E7D32; color: white; padding: 10px;} td {border-bottom: 1px solid #ddd; padding: 10px; text-align: center;}</style>"
    html = f"<html><head>{style}</head><body><h2>🎯 AI Value Report</h2>{df.to_html(index=False, escape=False)}</body></html>"
    msg = MIMEMultipart(); msg['Subject'] = f"🚀 AI Tipy: {len(data)}"; msg['From'] = GMAIL_USER; msg['To'] = GMAIL_RECEIVER
    msg.attach(MIMEText(html, 'html'))
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
    print("📧 Email odoslaný!")

if __name__ == "__main__":
    spustit_analyzu()
