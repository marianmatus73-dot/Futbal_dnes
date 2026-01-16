import asyncio, aiohttp, pandas as pd, io, os
from datetime import datetime
from fuzzywuzzy import process

async def ziskaj_tenis_tipy():
    api_key = os.getenv('ODDS_API_KEY')
    bank = float(os.getenv('AKTUALNY_BANK', 1000))
    nase_tipy = []
    curr = datetime.now().year
    
    async with aiohttp.ClientSession() as session:
        content = None
        for y in [curr, curr-1]:
            async with session.get(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv") as r:
                if r.status == 200: content = await r.read(); break
        if not content: return []

        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        all_m = []
        for _, r in df.iterrows():
            all_m.append({'p': r['winner_name'], 'win': 1, 'd': r['tourney_date']})
            all_m.append({'p': r['loser_name'], 'win': 0, 'd': r['tourney_date']})
        m_df = pd.DataFrame(all_m)
        p_stats = {p: (m_df[m_df['p']==p]['win'].mean()*0.4 + m_df[m_df['p']==p].head(15)['win'].mean()*0.6) for p in m_df['p'].unique()}

        async with session.get(f'https://api.the-odds-api.com/v4/sports/tennis_atp/odds/', params={'apiKey':api_key,'regions':'eu','markets':'h2h'}) as r:
            if r.status == 200:
                for m in await r.json():
                    p1 = process.extractOne(m['home_team'], p_stats.keys())
                    p2 = process.extractOne(m['away_team'], p_stats.keys())
                    if p1 and p2 and p1[1] > 75 and p2[1] > 75:
                        prob1 = p_stats[p1[0]] / (p_stats[p1[0]] + p_stats[p2[0]])
                        for bk in m['bookmakers']:
                            for out in bk['markets'][0]['outcomes']:
                                target = prob1 if out['name'] == m['home_team'] else (1-prob1)
                                edge = (target * out['price']) - 1
                                if 0.07 < edge < 0.35:
                                    v = round(((edge/(out['price']-1))*0.15)*bank, 2)
                                    nase_tipy.append({'Šport': '🎾 Tenis', 'Liga': 'ATP Tour', 'Zápas': f"{p1[0]}-{p2[0]}", 'Tip': out['name'], 'Kurz': out['price'], 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{v}€"})
    return nase_tipy
