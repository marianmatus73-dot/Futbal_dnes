import asyncio, aiohttp, pandas as pd, numpy as np, io, os, logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- KONFIGURÁCIA ---
load_dotenv()
API_ODDS_KEY = os.getenv('ODDS_API_KEY')
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
KELLY_FRAC = 0.15 # Pri tenise odporúčam nižší vklad (vysoká volatilita)

async def fetch_tenis_data(session):
    curr_year = datetime.now().year
    # Používame Jeff Sackmann ATP dáta (najspoľahlivejší zdroj)
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{curr_year}.csv"
    async with session.get(url) as r:
        if r.status == 200:
            return await r.read()
        return None

def vypocitaj_tenis_stats(df):
    """Vypočíta WinRate s dôrazom na aktuálnu formu."""
    # Spojíme víťazov a porazených do jedného zoznamu zápasov
    all_matches = []
    for _, row in df.iterrows():
        all_matches.append({'p': row['winner_name'], 'win': 1, 'date': row['tourney_date']})
        all_matches.append({'p': row['loser_name'], 'win': 0, 'date': row['tourney_date']})
    
    m_df = pd.DataFrame(all_matches)
    players = m_df['p'].unique()
    stats = {}

    for p in players:
        p_matches = m_df[m_df['p'] == p].sort_values('date', ascending=False)
        total_wins = p_matches['win'].sum()
        total_games = len(p_matches)
        
        # WinRate z posledných 15 zápasov (Forma)
        recent = p_matches.head(15)
        recent_winrate = recent['win'].mean()
        
        # Kombinovaný rating: 40% celková sezóna, 60% aktuálna forma
        total_winrate = (total_wins + 1) / (total_games + 2)
        stats[p] = (total_winrate * 0.4) + (recent_winrate * 0.6)
        
    return stats



async def analyzuj_tenis():
    async with aiohttp.ClientSession() as session:
        content = await fetch_tenis_data(session)
        if not content:
            print("❌ Nepodarilo sa stiahnuť tenisové dáta.")
            return

        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        player_stats = vypocitaj_tenis_stats(df)
        print(f"✅ Spracovaných {len(player_stats)} hráčov.")

        # Získanie kurzov pre ATP
        async with session.get(f'https://api.the-odds-api.com/v4/sports/tennis_atp/odds/', 
                              params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h'}) as r:
            if r.status != 200: return
            matches = await r.json()

        all_bets = []
        for m in matches:
            p1_name, p2_name = m['home_team'], m['away_team']
            
            # Fuzzy matching mien
            p1_match = process.extractOne(p1_name, player_stats.keys())
            p2_match = process.extractOne(p2_name, player_stats.keys())

            if p1_match and p2_match and p1_match[1] > 75 and p2_match[1] > 75:
                r1, r2 = player_stats[p1_match[0]], player_stats[p2_match[0]]
                
                # Pravdepodobnosť výhry (Logit model zjednodušene)
                prob1 = r1 / (r1 + r2)
                prob2 = 1 - prob1
                
                # Kontrola kurzov u bookmakerov
                for bk in m['bookmakers']:
                    for out in bk['markets'][0]['outcomes']:
                        target_prob = prob1 if out['name'] == p1_name else prob2
                        price = out['price']
                        edge = (target_prob * price) - 1

                        if 0.07 <= edge <= 0.35:
                            vklad = round(((edge / (price - 1)) * KELLY_FRAC) * AKTUALNY_BANK, 2)
                            all_bets.append({
                                'Zápas': f"{p1_match[0]} vs {p2_match[0]}",
                                'Tip': out['name'],
                                'Kurz': price,
                                'Edge': f"{round(edge*100, 1)}%",
                                'Vklad': f"{vklad}€"
                            })

        # Výstup
        if all_bets:
            print(pd.DataFrame(all_bets).drop_duplicates(subset=['Zápas']))
        else:
            print("Dnes žiadne value stávky v tenise.")

if __name__ == "__main__":
    asyncio.run(analyzuj_tenis())
