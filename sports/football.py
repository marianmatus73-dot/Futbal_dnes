from __future__ import annotations
import csv
import os
import asyncio
import subprocess
import sys
from pathlib import Path

from core.config import Settings
from core.types import SportResult, SportTip
from sports.base import SportModule
from sports.football_signals import enrich_football_tip

class FootballModule(SportModule):
    name = "football"

    # ... (metódy _engine_path, _export_dir, _latest_csv_candidates zostávajú)

    def _normalize_to_tip(self, row: dict) -> SportTip | None:
        """Parsuje CSV riadok priamo do Pydantic modelu."""
        try:
            # Tu sa odohráva kúzlo: Mapujeme rôzne názvy stĺpcov na jednotné polia
            data = {
                "sport": "football",
                "league": row.get("league") or row.get("competition") or "Unknown",
                "match": row.get("match") or f"{row.get('home_team')} vs {row.get('away_team')}" or "Unknown",
                "pick": row.get("pick") or row.get("selection") or "",
                "odds": float(row.get("odds") or row.get("price") or 0.0),
                "model_probability": float(row.get("model_probability") or row.get("prob") or 0.0),
                "bookmaker": row.get("bookmaker") or row.get("site") or "N/A",
                "reason": row.get("reason") or row.get("edge_reason") or ""
            }
            
            # Validácia a obohatenie
            tip = SportTip(**data)
            # Voliteľne zavolaj pôvodnú enrich logiku (ak ju potrebuješ zachovať)
            return enrich_football_tip(tip) 
        except (ValueError, TypeError, Exception):
            return None

    def _load_exported_tips(self) -> list[SportTip]:
        tips: list[SportTip] = []
        for csv_file in self._latest_csv_candidates():
            with csv_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tip = self._normalize_to_tip(row)
                    if tip and tip.odds > 1.0: # Základná validácia
                        tips.append(tip)
        
        # Deduplikácia podľa (league, match, pick, odds)
        return list({(t.league, t.match, t.pick, t.odds): t for t in tips}.values())

    async def _run_engine(self, args: list[str], settings: Settings) -> SportResult:
        # ... (zostáva pôvodná logika volania subprocess)
        
        tips = self._load_exported_tips() # Toto teraz vracia list[SportTip]
        
        return SportResult(
            sport=self.name,
            mode=" ".join(args),
            bets=tips, # <--- Tu posielame list objektov SportTip
            message="Football engine completed."
        )

    # ... (metódy scan, backtest, analytics zostávajú rovnaké)
