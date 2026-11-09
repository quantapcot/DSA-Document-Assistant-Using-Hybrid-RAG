"""Optional cross-encoder reranking; no model selected without approval."""
from functools import lru_cache

@lru_cache(maxsize=2)
def _get_model(name):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use reranking") from exc
    return CrossEncoder(name, device="cpu")

def rerank(query, candidates, *, model_name=None, scorer=None, top_k=5, batch_size=32):
    if not candidates or not query.strip() or top_k <= 0:
        return []
    if scorer is None:
        if not model_name:
            raise ValueError("Reranker model is not configured or approved")
        scorer = lambda pairs: _get_model(model_name).predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = list(scorer([(query, c["text"]) for c in candidates]))
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned an unexpected number of scores")
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), x[0]["chunk_id"]))[:top_k]
    return [{**chunk, "score": float(score), "rank": rank, "retrieval_method": "hybrid_rerank"} for rank, (chunk, score) in enumerate(ranked, 1)]
