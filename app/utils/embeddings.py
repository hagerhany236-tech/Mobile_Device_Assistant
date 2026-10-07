"""Dense retrieval helper: SentenceTransformer + cached document embeddings."""
from functools import lru_cache
from typing import List

import numpy as np

from app.data.devices import DEVICE_DOCUMENTS

MODEL_ID = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID)


@lru_cache(maxsize=1)
def get_doc_embeddings() -> np.ndarray:
    return get_model().encode(DEVICE_DOCUMENTS, normalize_embeddings=True)


def dense_rank(query: str) -> List[int]:
    """Document indices ordered by cosine similarity (dot product of normalized vectors)."""
    q = get_model().encode(query, normalize_embeddings=True)
    scores = get_doc_embeddings() @ q
    return [int(i) for i in np.argsort(-scores)]
