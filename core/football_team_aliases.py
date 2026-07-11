from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_ALIAS_FILE = "config/football_team_aliases.json"


BUILTIN_ALIASES: dict[str, str] = {
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "ac milan": "milan",
    "sporting": "sporting cp",
    "sporting lisbon": "sporting cp",
    "ifk goteborg": "ifk goteborg",
    "ifk göteborg": "ifk goteborg",
    "bayern munich": "bayern munchen",
    "bayern münchen": "bayern munchen",
    "borussia monchengladbach": "borussia monchengladbach",
    "borussia mönchengladbach": "borussia monchengladbach",
    "athletic bilbao": "athletic club",
    "ath bilbao": "athletic club",
    "real sociedad san sebastian": "real sociedad",
    "atletico madrid": "atletico de madrid",
    "atlético madrid": "atletico de madrid",
    "red bull salzburg": "rb salzburg",
    "rb leipzig": "rasenballsport leipzig",
    "sparta prague": "sparta praha",
    "slavia prague": "slavia praha",
    "fc copenhagen": "kobenhavn",
    "fc københavn": "kobenhavn",
    "malmo ff": "malmo",
    "malmö ff": "malmo",
    "goteborg": "ifk goteborg",
    "viking": "viking fk",
    "sarpsborg 08": "sarpsborg fk",
    "sarpsborg 08 ff": "sarpsborg fk",
    "aberdeen fc": "aberdeen",
    "heart of midlothian": "hearts",
    "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    "tottenham hotspur": "tottenham",
    "newcastle utd": "newcastle united",
    "west ham utd": "west ham united",
    "brighton hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "nottingham forest": "nottm forest",
    "sheffield utd": "sheffield united",
    "sheffield weds": "sheffield wednesday",
}


STOP_WORDS = {
    "fc",
    "cf",
    "sc",
    "afc",
    "ac",
    "club",
    "football",
    "futbol",
    "calcio",
    "the",
}


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def canonical_team_name(value: Any) -> str:
    text = strip_accents(str(value or "").casefold())
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [
        token
        for token in text.split()
        if token not in STOP_WORDS
    ]
    return " ".join(tokens).strip()


class FootballTeamAliasEngine:
    def __init__(
        self,
        alias_file: str = DEFAULT_ALIAS_FILE,
    ) -> None:
        self.alias_file = Path(alias_file)
        self.aliases = {
            canonical_team_name(key): canonical_team_name(value)
            for key, value in BUILTIN_ALIASES.items()
        }
        self._load_external_aliases()

    def _load_external_aliases(self) -> None:
        if not self.alias_file.exists():
            return

        try:
            payload = json.loads(
                self.alias_file.read_text(encoding="utf-8")
            )
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        for alias, canonical in payload.items():
            alias_key = canonical_team_name(alias)
            canonical_value = canonical_team_name(canonical)

            if alias_key and canonical_value:
                self.aliases[alias_key] = canonical_value

    def canonical(self, value: Any) -> str:
        normalized = canonical_team_name(value)
        return self.aliases.get(normalized, normalized)

    def similarity(self, left: Any, right: Any) -> float:
        left_canonical = self.canonical(left)
        right_canonical = self.canonical(right)

        if not left_canonical or not right_canonical:
            return 0.0

        if left_canonical == right_canonical:
            return 1.0

        left_tokens = set(left_canonical.split())
        right_tokens = set(right_canonical.split())

        token_union = left_tokens | right_tokens
        token_score = (
            len(left_tokens & right_tokens) / len(token_union)
            if token_union
            else 0.0
        )

        sequence_score = SequenceMatcher(
            None,
            left_canonical,
            right_canonical,
        ).ratio()

        containment_score = 1.0 if (
            left_canonical in right_canonical
            or right_canonical in left_canonical
        ) else 0.0

        return max(
            sequence_score,
            token_score * 0.90 + sequence_score * 0.10,
            containment_score * 0.92,
        )

    def is_match(
        self,
        left: Any,
        right: Any,
        *,
        threshold: float = 0.84,
    ) -> bool:
        return self.similarity(left, right) >= threshold


_default_engine = FootballTeamAliasEngine()


def canonical_team_name_with_aliases(value: Any) -> str:
    return _default_engine.canonical(value)


def team_similarity(left: Any, right: Any) -> float:
    return _default_engine.similarity(left, right)


def teams_match(
    left: Any,
    right: Any,
    *,
    threshold: float = 0.84,
) -> bool:
    return _default_engine.is_match(
        left,
        right,
        threshold=threshold,
    )
