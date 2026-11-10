"""Lazy, normalized sentence embeddings with configurable device."""
from functools import lru_cache
import logging
import numpy as np
from .config import RetrievalConfig

logger = logging.getLogger(__name__)
DEFAULT_MODEL = RetrievalConfig.embedding_model


def resolve_device(device: str) -> str:
    """Return a concrete PyTorch device string.

    Accepts ``"cpu"``, ``"cuda"`` or ``"auto"``.  When *auto* is chosen,
    CUDA is selected only if ``torch.cuda.is_available()`` returns True;
    otherwise it falls back to CPU with an informational log message.
    """
    if device in ("cpu", "cuda"):
        return device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA device for embeddings")
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA unavailable; falling back to CPU for embeddings")
        return "cpu"
    raise ValueError(f"Unknown device '{device}'; choose cpu, cuda, or auto")


@lru_cache(maxsize=4)
def _get_model(model_name: str, device: str = "cpu"):
    """Load and cache a SentenceTransformer, keyed by (model_name, device)."""
    resolved = resolve_device(device)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use dense retrieval") from exc
    logger.info("Loading embedding model %s on %s", model_name, resolved)
    return SentenceTransformer(model_name, device=resolved)


def embedding_dimension(model_name=DEFAULT_MODEL, *, device: str = "cpu"):
    dim = _get_model(model_name, device).get_sentence_embedding_dimension()
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Embedding model did not report a valid dimension")
    return dim


def embed_texts(texts, model_name=DEFAULT_MODEL, batch_size=32, *, device: str = "cpu"):
    if not texts:
        return []
    model = _get_model(model_name, device)
    vectors = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True), dtype=float)
    dim = embedding_dimension(model_name, device=device)
    if vectors.shape != (len(texts), dim) or not np.isfinite(vectors).all():
        raise ValueError("Embedding output shape or values are invalid")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    return (vectors / norms).tolist()


def embed_query(query, model_name=DEFAULT_MODEL, *, device: str = "cpu"):
    if not query.strip():
        return []
    return embed_texts([query], model_name, device=device)[0]
