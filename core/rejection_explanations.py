from __future__ import annotations

from typing import Any


EXPLANATIONS: dict[str, tuple[str, str, str]] = {
    "drawdown pause": ("BANKROLL", "Ochranná pauza po poklese bankrollu.", "Tip sa znovu posúdi po obnovení povoleného drawdownu."),
    "odds outside sport limits": ("KURZ", "Kurz je mimo bezpečného rozsahu daného športu.", "Sleduj zápas bez vkladu alebo počkaj na zmenu kurzu."),
    "league CLV below minimum": ("LIGA", "Liga dlhodobo neporáža closing kurz.", "Liga zostáva zablokovaná, kým sa CLV nepotvrdí na novej vzorke."),
    "opening price already shortened": ("POHYB KURZU", "Trh už zobral väčšinu pôvodnej hodnoty.", "Nevstupovať za horší kurz; čakať na nový value bod."),
    "conservative edge below sport minimum": ("VALUE", "Výhoda po započítaní neistoty je príliš malá.", "Model potrebuje lepší kurz alebo silnejšiu pravdepodobnosť."),
    "conservative edge above sport maximum": ("ANOMÁLIA", "Vypočítaná výhoda je nezvyčajne vysoká.", "Skontrolovať vstupné dáta a identitu udalosti."),
    "confidence below sport minimum": ("ISTOTA", "Zhoda modelových signálov je príliš slabá.", "Počkať na nové dáta, zostavy alebo finálne potvrdenie."),
    "duplicate event in current run": ("KORELÁCIA", "Na ten istý zápas už bol vybraný silnejší trh.", "Použiť iba jeden najsilnejší výber zo zápasu."),
    "opposite selection already allocated today": ("KONFLIKT", "Na zápas už existuje opačný výber.", "Nepridávať vzájomne sa rušiace stávky."),
    "correlated event exposure reached": ("KORELÁCIA", "Na tento zápas už je využitá povolená expozícia.", "Počkať na settlement alebo zrušiť slabší výber."),
    "odds band tip limit reached": ("DIVERZIFIKÁCIA", "Limit tipov v rovnakom kurzovom pásme je naplnený.", "Uprednostniť najlepší tip v pásme."),
    "daily sport tip limit reached": ("LIMIT", "Denný počet tipov pre šport je naplnený.", "Ďalší kandidát zostáva iba v sledovaní."),
    "daily exposure limit reached": ("BANKROLL", "Denná celková expozícia dosiahla limit.", "Nepridávať ďalší vklad v tento deň."),
    "sport exposure limit reached": ("BANKROLL", "Expozícia daného športu dosiahla limit.", "Rozložiť riziko medzi športy alebo počkať na settlement."),
    "league exposure limit reached": ("BANKROLL", "Expozícia jednej ligy dosiahla limit.", "Nepridávať ďalší korelovaný ligový tip."),
    "daily loss limit reached": ("BANKROLL", "Dnešná strata prekročila bezpečnostný limit.", "Nové tipy sú pozastavené do ďalšieho dňa."),
    "not selected for the published top list": ("PORADIE", "Kandidát nepatrí medzi najsilnejšie dnešné výbery.", "Zostáva uložený na vyhodnotenie a učenie."),
}


def explain_rejection(reason: str | None) -> dict[str, Any]:
    raw = str(reason or "unknown reason").strip()
    category, explanation, next_step = EXPLANATIONS.get(
        raw,
        ("FILTER", "Kandidát nesplnil jednu z produkčných podmienok.", "Pozri namerané hodnoty a prah filtra."),
    )
    return {
        "code": raw.upper().replace(" ", "_"),
        "category": category,
        "explanation": explanation,
        "next_step": next_step,
    }


def enrich_rejection(candidate: dict[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    item["rejection_explanation"] = explain_rejection(item.get("rejection_reason"))
    return item
