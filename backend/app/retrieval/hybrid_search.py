"""Unified scoped BM25, dense, RRF, and optional reranking interface."""
from .config import RetrievalConfig
from .keyword_search import BM25Retriever
from .vector_search import search_dense
from .vector_store import get_client, scope_filter
from .reranker import rerank

def rrf_fuse(lists, *, k=60, top_k=20):
    if k < 0:
        raise ValueError("rrf_k must be nonnegative")
    combined = {}
    for results in lists:
        seen = set()
        for rank, chunk in enumerate(results, 1):
            key = chunk["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            if key not in combined:
                combined[key] = [chunk, 0.0]
            combined[key][1] += 1.0 / (k + rank)
    ranked = sorted(combined.values(), key=lambda x: (-x[1], x[0]["chunk_id"]))[:max(0, top_k)]
    return [{**chunk, "score": score, "rank": rank, "retrieval_method": "hybrid_rrf"} for rank, (chunk, score) in enumerate(ranked, 1)]

class RetrievalEngine:
    def __init__(self, *, client=None, config=None, query_embedder=None, rerank_scorer=None):
        self.config = config or RetrievalConfig.from_env()
        self.client = client or get_client(config=self.config)
        self.query_embedder = query_embedder
        self.rerank_scorer = rerank_scorer

    def _bm25(self, query, corpus_id, workspace_id, top_k):
        scope = scope_filter(corpus_id, workspace_id)
        if not self.client.collection_exists(self.config.collection):
            return []
        records, offset = [], None
        while True:
            points, offset = self.client.scroll(self.config.collection, scroll_filter=scope, limit=256, offset=offset, with_payload=True, with_vectors=False)
            records.extend(p.payload for p in points)
            if offset is None:
                break
        index = BM25Retriever(records, k1=self.config.bm25_k1, b=self.config.bm25_b, accent_insensitive=self.config.accent_insensitive)
        return index.search(query, top_k)

    def retrieve(self, query: str, method: str, top_k: int = 5, corpus_id: str | None = None, workspace_id: str | None = None):
        scope_filter(corpus_id, workspace_id)
        if method not in {"bm25", "dense", "hybrid_rrf", "hybrid_rerank"}:
            raise ValueError("Unknown retrieval method")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0 or not query.strip():
            return []
        candidate_k = max(top_k, self.config.candidate_top_k)
        lexical = self._bm25(query, corpus_id, workspace_id, candidate_k) if method != "dense" else []
        dense = search_dense(self.client, query, corpus_id=corpus_id, workspace_id=workspace_id, config=self.config, top_k=candidate_k, query_embedder=self.query_embedder) if method != "bm25" else []
        if method == "bm25": return lexical[:top_k]
        if method == "dense": return dense[:top_k]
        fused = rrf_fuse([lexical, dense], k=self.config.rrf_k, top_k=candidate_k)
        if method == "hybrid_rrf": return fused[:top_k]
        return rerank(query, fused, model_name=self.config.reranker_model, scorer=self.rerank_scorer, top_k=top_k, batch_size=self.config.batch_size)
