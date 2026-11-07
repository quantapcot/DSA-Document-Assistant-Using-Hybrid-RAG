"""C1: text and query embeddings — sentence-transformers, offline, no API key."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Model nhẹ, multilingual (hỗ trợ cả tiếng Việt lẫn tiếng Anh)
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Load model một lần, cache lại (lazy load)."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """
    Chuyển list[str] → list[vector float32].
    
    Parameters
    ----------
    texts      : Danh sách văn bản cần embed.
    model_name : Tên model sentence-transformers.

    Returns
    -------
    list[list[float]] — mỗi phần tử là vector 384 chiều.
    """
    if not texts:
        return []
    model = _get_model(model_name)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity = dot product
    )
    return embeddings.tolist()


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Embed một câu query duy nhất."""
    return embed_texts([query], model_name)[0]
