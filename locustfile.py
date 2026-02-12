"""
    Locust load generator cycling deterministic single-word names via GET /predict.
"""

# -*- coding: utf-8 -*-

from itertools import cycle
from pathlib import Path
from urllib.parse import quote_plus
import random
from locust import HttpUser, task, between

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_PAYLOADS = 100
SHUFFLE_SEED = 1337


def _is_single_word(text: str) -> bool:
    """
        Return True when the candidate has no internal whitespace.

        Args:
            text (str): Name candidate to inspect.

        Returns:
            bool: True if the candidate contains no whitespace characters.
    """
    return bool(text) and not any(ch.isspace() for ch in text)


def _load_candidate_strings() -> list[str]:
    """
        Collect and deduplicate single-word names from bilingual and per-language files.

        Returns:
            list[str]: Unique single-word names harvested from the data directory.
    """
    data_sources = []
    eng_fra_file = DATA_DIR / "eng-fra.txt"
    if eng_fra_file.exists():
        with eng_fra_file.open(encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                if "\t" in cleaned:
                    for part in cleaned.split("\t"):
                        trimmed = part.strip()
                        if trimmed and _is_single_word(trimmed):
                            data_sources.append(trimmed)
                else:
                    if _is_single_word(cleaned):
                        data_sources.append(cleaned)
    names_dir = DATA_DIR / "names"
    if names_dir.exists():
        for name_file in sorted(names_dir.glob("*.txt")):
            with name_file.open(encoding="utf-8") as handle:
                for line in handle:
                    cleaned = line.strip()
                    if cleaned and _is_single_word(cleaned):
                        data_sources.append(cleaned)
    if not data_sources:
        raise RuntimeError(f"No readable payload sources found under {DATA_DIR}.")
    deduped: list[str] = []
    seen = set()
    for text in data_sources:
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def _build_payloads(limit: int = MAX_PAYLOADS) -> list[str]:
    """
        Shuffle to a deterministic subset of unique names capped by limit.

        Args:
            limit (int): Maximum size of the payload list.

        Returns:
            list[str]: Deterministically shuffled subset of unique names.
    """
    uniques = _load_candidate_strings()
    if len(uniques) < limit:
        raise RuntimeError(
            f"Only {len(uniques)} unique entries discovered under {DATA_DIR}. Need at least {limit}."
        )
    shuffled = uniques[:]
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    return shuffled[:limit]


PAYLOADS = _build_payloads()


def _payload_cycle():
    """
        Create a per-user iterator that endlessly cycles through randomized names.

        Returns:
            itertools.cycle: Iterator cycling through randomized names.
    """
    selections = PAYLOADS[:]
    random.shuffle(selections)
    return cycle(selections)


class ModelUser(HttpUser):
    """
        Locust user issuing GET /predict requests with rotating single-word names.
    """

    wait_time = between(1, 2)

    def on_start(self):
        self.payloads = _payload_cycle()

    @task
    def predict(self):
        name = next(self.payloads)
        self.client.get(f"/predict?name={quote_plus(name)}")
