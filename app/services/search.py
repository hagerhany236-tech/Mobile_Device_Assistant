"""Hybrid search: BM25 + dense embeddings fused with Reciprocal Rank Fusion."""
from typing import Dict, List, Tuple

from app.data.devices import DEVICE_DOCUMENTS
from app.utils.bm25_index import get_bm25_index
from app.utils.embeddings import dense_rank

RRF_K = 60


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), ranks starting at 1."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(query: str, k: int = 3) -> List[Tuple[str, float]]:
    sparse = get_bm25_index().rank(query)
    dense = dense_rank(query)
    fused = reciprocal_rank_fusion([sparse, dense])
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(DEVICE_DOCUMENTS[i], score) for i, score in top]
