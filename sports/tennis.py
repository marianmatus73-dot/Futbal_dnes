    async def scan(self, settings: Settings) -> SportResult:
        init_sport_db(settings)

        # Čistý zoznam kľúčov kompatibilný s The Odds API v4
        configured_keys = os.getenv(
            "TENNIS_SPORT_KEYS",
            ",".join(
                [
                    "tennis_atp_singles",
                    "tennis_wta_singles",
                    "tennis_atp_doubles",
                    "tennis_wta_doubles",
                    "tennis_atp_challenger_singles",
                    "tennis_wta_challenger_singles"
                ]
            ),
        ).split(",")

        clean_sport_keys = [
            sport_key.strip()
            for sport_key in configured_keys
            if sport_key.strip()
        ]

        if os.getenv("SPORT_KEY_AUTO_DISCOVERY", "1") == "1":
            active_keys = await discover_active_sport_keys(
                settings.odds_api_key,
                ["Tennis"],
            )

            clean_sport_keys = filter_active_keys(
                clean_sport_keys,
                active_keys,
            )

        # Zoznam nájdených turnajov pre report
        active_tournaments = ", ".join(clean_sport_keys)
        log.info("Tennis: Skenujem tieto kľúče: %s", active_tournaments)

        settled = await settle_sport_bets(
            settings=settings,
            sport=self.name,
            sport_keys=clean_sport_keys,
        )

        updated_clv = update_closing_lines(settings, self.name)
        refresh_bookmaker_stats(settings, self.name)

        min_books = int(os.getenv("MIN_TENNIS_BOOKMAKERS", "2"))
        top_n = int(os.getenv("TOP_N_REPORT", "8"))
        grade_min_samples = int(
            os.getenv("TENNIS_BOOKMAKER_GRADE_MIN_SAMPLES", "20")
        )

        bets: list[Bet] = []
        snapshots_saved = 0
        blocked = 0
        scanned_events = 0

        for sport_key in clean_sport_keys:
            data = await fetch_odds(
                settings.odds_api_key,
                sport_key,
                markets="h2h",
            )

            if not data:
                continue

            for event in data:
                league = sport_key
                home = str(event.get("home_team", ""))
                away = str(event.get("away_team", ""))
                start = str(event.get("commence_time", ""))
                event_name = f"{home} vs {away}"
                bookmakers = event.get("bookmakers", [])

                scanned_events += 1

                snapshots_saved += self._save_snapshot(
                    settings,
                    sport_key,
                    event_name,
                    home,
                    away,
                    bookmakers,
                )

                consensus = consensus_h2h(
                    bookmakers,
                    min_books=min_books,
                )

                if not consensus:
                    blocked += 1
                    continue

                for bookmaker, selection, odds in best_outlier_prices(bookmakers):
                    prob_market = consensus.get(selection)
                    if not prob_market:
                        continue

                    grade = bookmaker_grade(settings, self.name, bookmaker, min_samples=grade_min_samples)
                    elo_adj = elo_adjustment(settings, self.name, home, away, selection)
                    surface_adj = tennis_surface_adjustment(sport_key)

                    ensemble = build_ensemble_probability(
                        EnsembleInput(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_adjustment=(grade - 1.0) * 0.02,
                            sport_adjustment=surface_adj,
                        ),
                        odds=odds,
                    )

                    fallback_probability = ensemble.probability
                    fallback_edge = ensemble.edge

                    try:
                        features = MetaFeatures(
                            market_probability=prob_market,
                            elo_adjustment=elo_adj,
                            form_adjustment=0.0,
                            clv_adjustment=0.0,
                            bookmaker_grade=grade,
                            sport_weight=sport_weight(self.name),
                            league_weight=league_weight(league),
                            confidence=0.5, # Zjednodušené pre stabilitu
                            monte_carlo_probability=0.5,
                        )
                        prob_final = max(0.01, min(0.99, predict_probability(features)))
                        edge = prob_final * odds - 1.0
                    except Exception:
                        prob_final = fallback_probability
                        edge = fallback_edge

                    stake = round(kelly_stake(prob_final, odds, settings) * grade, 2)

                    if settings.min_edge <= edge <= settings.max_edge and stake > 0 and odds <= settings.max_odds:
                        bet = Bet(
                            sport=self.name,
                            league=league,
                            event=event_name,
                            market="h2h",
                            selection=selection,
                            odds=odds,
                            prob_model=prob_market,
                            prob_market=prob_market,
                            prob_final=prob_final,
                            edge=edge,
                            stake=stake,
                            bookmaker=bookmaker,
                            start_time=start,
                            score=float(edge * 100),
                        )
                        bets.append(bet)
                        self._save_bet(settings, bet)
                    else:
                        blocked += 1

        bets = dedupe_best_bets(bets)
        analytics = sport_analytics_report(settings, self.name)

        return SportResult(
            sport=self.name,
            mode="scan",
            bets=bets[:top_n],
            message=(
                f"Tennis v16: Aktivné turnaje: {active_tournaments}. "
                f"Settled: {settled}. Events scanned: {scanned_events}. "
                f"Snapshots saved: {snapshots_saved}. Blocked: {blocked}. "
                f"Stored candidates: {len(bets)}.
{analytics}"
            ),
        )
