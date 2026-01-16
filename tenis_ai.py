import asyncio, aiohttp, pandas as pd, numpy as np, io, os, logging
from datetime import datetime
from dotenv import load_dotenv
from fuzzywuzzy import process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
KELLY_FRAC = 0.15

async def fetch_tenis_data(session):
    curr_year = datetime.now().year
    for year in [curr_year, curr_year - 1]:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try:
            async with session.get(url, timeout=15) as r:
                if r.status == 200:
                    logging.info(f"✅ Tenis: Načítané dáta pre rok {year}")
                    return await r.read()
        except: continue
    return None

def vypocitaj_stats(df):
    all_m = []
    for _, r in df.iterrows():
        all_m.append({'p': r['winner_name'], 'win': 1, 'd': r['tourney_date']})
        all_m.append({'p': r['loser_name'], 'win': 0, 'd': r['tourney_date']})
    m_df = pd.DataFrame(all_m)
    res = {}
    for p in m_df['p'].unique():
        p_m = m_df[m_df['p'] == p].sort_values('d', ascending=False)
        res[p] = (p_m['win'].mean() * 0.4) + (p_m.head(15)['win'].mean() * 0.6)
    return res

async def analyzuj_tenis():
    async with aiohttp.ClientSession() as session:
        content = await fetch_tenis_data(session)
        if not content: return
        p_stats = vypocitaj_stats(pd.read_csv(io.StringIO(content.decode('utf-8'))))
        async with session.get(f'https://api.the-odds-api.com/v4/sports/tennis_atp/odds/', params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h'}) as r:
            if r.status != 200: return
            for m in await r.json():
                p1_m = process.extractOne(m['home_team'], p_stats.keys())
                p2_m = process.extractOne(m['away_team'], p_stats.keys())
                if p1_m and p2_m and p1_m[1] > 75 and p2_m[1] > 75:
                    prob1 = p_stats[p1_m[0]] / (p_stats[p1_m[0]] + p_stats[p2_m[0]])
                    for bk in m['bookmakers']:
                        for out in bk['markets'][0]['outcomes']:
                            target = prob1 if out['name'] == m['home_team'] else (1 - prob1)
                            edge = (target * out['price']) - 1
                            if 0.07 < edge < 0.35:
                                v = round(((edge/(out['price']-1))*KELLY_FRAC)*AKTUALNY_BANK, 2)
                                print(f"🎾 Tenis: {m['home_team']} vs {m['away_team']} | Tip: {out['name']} @{out['price']} | Edge: {round(edge*100,1)}% | Vklad: {v}€")

if __name__ == "__main__":
    asyncio.run(analyzuj_tenis())
