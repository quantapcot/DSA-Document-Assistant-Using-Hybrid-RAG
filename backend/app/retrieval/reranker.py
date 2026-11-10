"""Cross-encoder reranking with configurable device and max_length.

Sentence Transformers v5.7+ API
-------------------------------
* ``CrossEncoder.__init__`` accepts ``max_length`` to set the tokenizer
  truncation length at construction time.
* ``CrossEncoder.predict`` does **not** accept ``max_length``.
* ``max_length=0`` or ``None`` → use the model's built-in default.

The model is lazy-loaded and cached by ``(name, device, max_length)``
so it is never reloaded unnecessarily.
"""
from functools import lru_cache
import logging
import math

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


@lru_cache(maxsize=4)
def _get_model(name: str, device: str = "cpu", max_length: int = 0):
    """Load and cache a CrossEncoder, keyed by (name, resolved_device, max_length).

    ``max_length`` is set at construction time per the ST v5.7 API.
    A value of 0 or None uses the model's built-in default.

    Raises a clear error on CUDA OOM or model load failure — never
    silently falls back to CPU.
    """
    resolved = _resolve_device(device)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers to use reranking"
        ) from exc

    init_kwargs: dict = {"device": resolved}
    if max_length and max_length > 0:
        init_kwargs["max_length"] = max_length

    try:
        logger.info(
            "Loading reranker model %s on %s (max_length=%s)",
            name, resolved, max_length or "default",
        )
        model = CrossEncoder(name, **init_kwargs)
        try:
            actual_device = str(next(model.model.parameters()).device)
        except Exception:
            actual_device = resolved
        logger.info(
            "Reranker model %s initialized successfully on actual device: %s",
            name, actual_device,
        )
        return model
    except RuntimeError as exc:
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
        Passed to the CrossEncoder constructor (NOT to predict).
    """
    if not candidates or not query.strip() or top_k <= 0:
        return []
    if scorer is None:
        if not model_name:
            raise ValueError("Reranker model is not configured or approved")

        resolved_device = _resolve_device(device)

        def scorer(pairs, *, _name=model_name, _bs=batch_size,
                   _dev=resolved_device, _ml=max_length):
            model = _get_model(_name, _dev, _ml)
            try:
                return model.predict(
                    pairs, batch_size=_bs, show_progress_bar=False,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "CUDA" in msg or "out of memory" in msg.lower():
                    raise RuntimeError(
                        f"CUDA OOM during reranker predict (batch_size={_bs}). "
                        f"Try reducing reranker_batch_size or candidate_top_k."
                    ) from exc
                raise

    pairs = [(query, c["text"]) for c in candidates]
    scores = list(scorer(pairs))
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned an unexpected number of scores")
    # Validate scores — reject NaN/Inf.
    for i, s in enumerate(scores):
        if not math.isfinite(float(s)):
            raise ValueError(
                f"Reranker returned non-finite score ({s}) for candidate {i}"
            )
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: (-float(x[1]), x[0]["chunk_id"]),
    )[:top_k]
    return [
        {**chunk, "score": float(score), "rank": rank,
         "retrieval_method": "hybrid_rerank"}
        for rank, (chunk, score) in enumerate(ranked, 1)
    ]
