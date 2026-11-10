"""Scoped dense retrieval."""
from .embeddings import embed_query
from .vector_store import scope_filter, check_model, DENSE_VECTOR_NAME

def search_dense(client, query, *, corpus_id, workspace_id, config, top_k=20, query_embedder=None):
    scope = scope_filter(corpus_id, workspace_id)
    if not query.strip() or top_k <= 0 or not client.collection_exists(config.collection):
        return []
    check_model(client, config)
    vector = (query_embedder or (lambda q: embed_query(q, config.embedding_model, device=config.embedding_device)))(query)
    try:
        points = client.query_points(config.collection, query=vector, using=DENSE_VECTOR_NAME, query_filter=scope, limit=top_k, with_payload=True).points
    except Exception as exc:
        raise RuntimeError(f"Qdrant dense search failed: {exc}") from exc
    return [{**p.payload, "score": float(p.score), "rank": rank, "retrieval_method": "dense"} for rank, p in enumerate(points, 1)]
