# --- UPRAVENÁ ČASŤ V CYKLE ANALYZUJ ---

            for m in matches:
                m_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if not (now_utc <= m_time <= limit_utc): continue

                c1_m, c2_m = process.extractOne(m['home_team'], stats.index), process.extractOne(m['away_team'], stats.index)
                if not c1_m or not c2_m or c1_m[1] < 70: continue
                c1, c2 = c1_m[0], c2_m[0]

                probs, lim = {}, (5.5 if cfg['sport']=='hokej' else 2.5)
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1,'WinRate'], stats.at[c2,'WinRate']
                    probs = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh, la = (stats.at[c1,'AH']*stats.at[c2,'DA']*ah + cfg['ha']), (stats.at[c2,'AA']*stats.at[c1,'DH']*aa)
                    pu = sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(12) if i+j < lim)
                    probs = {'1':sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(i)), 'X':sum(poisson.pmf(i,lh)*poisson.pmf(i,la) for i in range(12)), '2':sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(i+1,12)), f'Over {lim}':1-pu, f'Under {lim}':pu}

                # Pomocný slovník pre filtrovanie najlepšieho kurzu na úrovni zápasu
                match_best_odds = {}

                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for out in mk['outcomes']:
                            lbl = f"{out['name']} {lim}" if mk['key']=='totals' and out.get('point')==lim else ('1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X'))
                            
                            if lbl in probs:
                                prob, price = probs[lbl], out['price']
                                edge = (prob * price) - 1
                                
                                if 0.05 <= edge <= 0.45:
                                    # AK UŽ TENTO TIP MÁME, ALE S NIŽŠÍM KURZOM, NAHRADÍME HO
                                    if lbl not in match_best_odds or price > match_best_odds[lbl]['Kurz']:
                                        v = round(min(max(0, (((price-1)*prob-(1-prob))/(price-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                        match_best_odds[lbl] = {
                                            'Zápas': f"{c1} vs {c2}",
                                            'Tip': lbl,
                                            'Kurz': price,
                                            'Edge': f"{round(edge*100,1)}%",
                                            'Vklad': f"{v}€",
                                            'Sport': cfg['sport'],
                                            'Skóre': f"{round(lh,1)}:{round(la,1)}" if cfg['sport']!='tenis' else ""
                                        }
                
                # Pridáme len tie najlepšie vyfiltrované kurzy pre daný zápas
                all_bets.extend(match_best_odds.values())

        if all_bets:
            # ODSTRÁNENIE DUPLICÍT AJ MEDZI RÔZNYMI LIGAMI (ak by sa náhodou vyskytli)
            final_df = pd.DataFrame(all_bets).sort_values('Kurz', ascending=False).drop_duplicates(subset=['Zápas', 'Tip'])
            final_bets = final_df.to_dict('records')
            
            alerts = uloz_a_clv(final_bets)
            posli_email(final_bets, alerts)
