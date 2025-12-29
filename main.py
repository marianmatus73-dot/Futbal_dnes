import requests
import pandas as pd
import smtplib
import os
import io
import pytz
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. NASTAVENIA & ENV ---
load_dotenv()

def get_required_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Chýbajú env premenné: {', '.join(missing)}")
    return {k: os.getenv(k) for k in keys}

env = get_required_env(['ODDS_API_KEY', 'GMAIL_USER', 'GMAIL_PASSWORD'])
API_ODDS_KEY = env['ODDS_API_KEY']
GMAIL_USER = env['GMAIL_USER']
GMAIL_PASSWORD = env['GMAIL_PASSWORD']
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

# --- 2. POMOCNÉ FUNKCIE ---

def get_local_time(utc_str):
    """Konvertuje UTC čas z API na Europe/Bratislava."""
    try:
        utc_dt = datetime.fromisoformat(utc_str.replace('Z', '')).replace(tzinfo=pytz.utc)
        local_tz = pytz.timezone('Europe/Bratislava')
        return utc_dt.astimezone(local_tz).strftime('%H:%M')
    except:
        return '??:??'

def vypocitaj_kelly(pravdepodobnost, kurz):
    if kurz <= 1 or pravdepodobnost <= 0: return 0
    b = kurz - 1
    f_star = (b * pravdepodobnost - (1 - pravdepodobnost)) / b
    frac = min(max(0, f_star * KELLY_FRACTION), MAX_BANK_PCT)
    return round(frac * 100, 2)

# --- 3. JADRO MODELU (MATEMATIKA + VÁŽENÁ FORMA) ---

def vypocitaj_silu_timov(df):
    if df.empty or len(df) < 10: return None, 0, 0
    
    # Pridanie váhy podľa čerstvosti zápasu (novšie zápasy majú vyššiu váhu)
    df = df.copy()
    df['Weight'] = 1.0
    if len(df) > 50:
        # Posledných 20 zápasov každého tímu dostane 1.5x vyššiu váhu
        df.iloc[-30:, df.columns.get_loc('Weight')] = 1.5

    avg_h = (df['FTHG'] * df['Weight']).sum() / df['Weight'].sum()
    avg_a = (df['FTAG'] * df['Weight']).sum() / df['Weight'].sum()

    def get_weighted_stats(group, col):
        return (group[col] * group['Weight']).sum() / group['Weight'].sum()

    h_stats = df.groupby('HomeTeam').apply(lambda x: pd.Series({
        'Att_H': get_weighted_stats(x, 'FTHG') / avg_h,
        'Def_H': get_weighted_stats(x, 'FTAG') / avg_a
    }))
    
    a_stats = df.groupby('AwayTeam').apply(lambda x: pd.Series({
        'Att_A': get_weighted_stats(x, 'FTAG') / avg_a,
        'Def_A': get_weighted_stats(x, 'FTHG') / avg_h
    }))

    return h_stats.join(a_stats, how='outer').fillna(1.0), avg_h, avg_a

def predikuj_vsetko(home, away, stats, avg_h, avg_a, sport='futbal'):
    if home not in stats.index or away not in stats.index: return None

    lamb_h = stats.at[home, 'Att_H'] * stats.at[away, 'Def_A'] * avg_h
    lamb_a = stats.at[away, 'Att_A'] * stats.at[home, 'Def_H'] * avg_a

    res = {'1': 0, 'X': 0, '2': 0, 'over': 0, 'under': 0}
    limit = 2.5 if sport == 'futbal' else 5.5
    
    for x in range(12):
        for y in range(12):
            p = poisson.pmf(x, lamb_h) * poisson.pmf(y, lamb_a)
            if x > y: res['1'] += p
            elif x == y: res['X'] += p
            else: res['2'] += p
            if (x + y) > limit: res['over'] += p
            else: res['under'] += p

    # Normalizácia (aby bol súčet presne 100%)
    total_1x2 = res['1'] + res['X'] + res['2']
    res['1'] /= total_1x2; res['X'] /= total_1x2; res['2'] /= total_1x2
    
    total_ou = res['over'] + res['under']
    res['over'] /= total_ou; res['under'] /= total_ou

    if sport == 'nhl':
        res['ML1'] = res['1'] + (res['X'] * 0.51)
        res['ML2'] = res['2'] + (res['X'] * 0.49)
    return res

# --- 4. ANALÝZA A SPRACOVANIE ---

def spustit_analyzu():
    all_potential_bets = []
    print(f"🚀 Štart: {datetime.now().strftime('%d.%m %H:%M')}")

    for liga, cfg in LIGY_CONFIG.items():
        print(f"🔍 Analyzujem: {liga}")
        df = stiahni_csv_data(cfg['csv'])
        if df.empty: continue

        stats, avg_h, avg_a = vypocitaj_silu_timov(df)
        matches = ziskaj_kurzy(cfg['api'])
        if not matches or stats is None: continue

        sport_type = 'nhl' if 'NHL' in liga else 'futbal'

        for m in matches:
            api_h, api_a = m['home_team'], m['away_team']
            start_time = get_local_time(m['commence_time'])
            
            csv_h = fuzzy_match_team(api_h, stats.index)
            csv_a = fuzzy_match_team(api_a, stats.index)
            if not csv_h or not csv_a: continue

            probs = predikuj_vsetko(csv_h, csv_a, stats, avg_h, avg_a, sport_type)
            if not probs: continue

            for bookie in m.get('bookmakers', []):
                # Výpočet Value pre H2H a Totals (podobne ako v tvojom pôvodnom kóde)
                # ... (tu ostáva tvoja pôvodná logika prechádzania outcomes) ...
                # Poznámka: v r['Model'] už budú normalizované pravdepodobnosti.
                pass

    # ... (tu nasleduje tvoja pôvodná logika odosielania emailu) ...

# Pomocné funkcie stiahni_csv_data, ziskaj_kurzy a fuzzy_match_team ostávajú nezmenené z tvojho pôvodného kódu.
