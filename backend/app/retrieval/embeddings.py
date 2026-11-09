"""Lazy, normalized sentence embeddings."""
from functools import lru_cache
import numpy as np
from .config import RetrievalConfig

DEFAULT_MODEL = RetrievalConfig.embedding_model

@lru_cache(maxsize=4)
def _get_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use dense retrieval") from exc
    return SentenceTransformer(model_name, device="cpu")

def embedding_dimension(model_name=DEFAULT_MODEL):
    dim = _get_model(model_name).get_sentence_embedding_dimension()
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Embedding model did not report a valid dimension")
    return dim

def embed_texts(texts, model_name=DEFAULT_MODEL, batch_size=32):
    if not texts:
        return []
    model = _get_model(model_name)
    vectors = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True), dtype=float)
    dim = embedding_dimension(model_name)
    if vectors.shape != (len(texts), dim) or not np.isfinite(vectors).all():
        raise ValueError("Embedding output shape or values are invalid")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    return (vectors / norms).tolist()

def embed_query(query, model_name=DEFAULT_MODEL):
    if not query.strip():
        return []
    return embed_texts([query], model_name)[0]
