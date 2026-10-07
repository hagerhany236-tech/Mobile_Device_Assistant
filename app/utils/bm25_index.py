"""Reusable BM25 index (built once, reused for every request)."""
from functools import lru_cache
from typing import List

from rank_bm25 import BM25Okapi

from app.data.devices import DEVICE_DOCUMENTS


def tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, documents: List[str]):
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def rank(self, query: str) -> List[int]:
        """Document indices, most relevant first."""
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    return BM25Index(DEVICE_DOCUMENTS)
