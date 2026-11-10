"""Optional cross-encoder reranking with configurable device; no model selected without approval."""
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Return a concrete device string from ``"cpu"``, ``"cuda"`` or ``"auto"``."""
    if device in ("cpu", "cuda"):
        return device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA device for reranker")
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA unavailable; falling back to CPU for reranker")
        return "cpu"
    raise ValueError(f"Unknown reranker device '{device}'; choose cpu, cuda, or auto")


@lru_cache(maxsize=2)
def _get_model(name: str, device: str = "cpu"):
    """Load and cache a CrossEncoder, keyed by (name, device).

    Raises a clear error when CUDA runs out of memory or the model fails to
    load, rather than silently falling back.
    """
    resolved = _resolve_device(device)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use reranking") from exc
    try:
        logger.info("Loading reranker model %s on %s", name, resolved)
        return CrossEncoder(name, device=resolved)
    except RuntimeError as exc:
        # Surface CUDA OOM or device errors clearly.
        msg = str(exc)
        if "CUDA" in msg or "out of memory" in msg.lower():
            raise RuntimeError(
                f"Failed to load reranker '{name}' on {resolved}: {msg}. "
                f"Try setting reranker_device to 'cpu' or freeing GPU memory."
            ) from exc
        raise


def rerank(query, candidates, *, model_name=None, scorer=None, top_k=5,
           batch_size=32, device="cpu", max_length=0):
    """Rerank *candidates* for *query* using a CrossEncoder or custom scorer.

    Parameters
    ----------
    query : str
        The search query.
    candidates : list[dict]
        Retrieval results, each containing at least ``"chunk_id"`` and ``"text"``.
    model_name : str or None
        CrossEncoder model identifier.  Required when *scorer* is ``None``.
    scorer : callable or None
        Optional custom scorer ``(pairs) -> scores``.  Overrides *model_name*.
    top_k : int
        Maximum results to return.
    batch_size : int
        Prediction batch size for the CrossEncoder.
    device : str
        Device for the CrossEncoder (``"cpu"``, ``"cuda"``, ``"auto"``).
    max_length : int
        Maximum token length per input pair.  ``0`` uses the model default.
    """
    if not candidates or not query.strip() or top_k <= 0:
        return []
    if scorer is None:
        if not model_name:
            raise ValueError("Reranker model is not configured or approved")

        def scorer(pairs, *, _name=model_name, _bs=batch_size, _dev=device, _ml=max_length):
            model = _get_model(_name, _dev)
            predict_kwargs = {"batch_size": _bs, "show_progress_bar": False}
            if _ml > 0:
                predict_kwargs["max_length"] = _ml
            return model.predict(pairs, **predict_kwargs)

    pairs = [(query, c["text"]) for c in candidates]
    scores = list(scorer(pairs))
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned an unexpected number of scores")
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), x[0]["chunk_id"]))[:top_k]
    return [{**chunk, "score": float(score), "rank": rank, "retrieval_method": "hybrid_rerank"} for rank, (chunk, score) in enumerate(ranked, 1)]
