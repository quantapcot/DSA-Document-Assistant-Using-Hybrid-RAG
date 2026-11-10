"""Unified scoped BM25, dense, RRF, and optional reranking interface.

BM25 cache
----------
Building a BM25 index from 5 941 Qdrant payloads on every query is wasteful.
This module caches ``BM25Retriever`` instances keyed by
``(collection, corpus_id, workspace_id, bm25_k1, bm25_b, accent_insensitive)``
so that repeated queries against the same scope reuse the index.

Invalidation strategy (designed for a single-server course project):

* **In-process invalidation** — ``invalidate_bm25_cache`` must be called
  after any scope mutation (add/update/delete/rebuild).  ``build_index``
  calls this automatically.

* **Cross-process staleness** — When indexing and retrieval run in
  different Python processes, in-process invalidation alone is
  insufficient.  A **TTL** (``bm25_cache_ttl``, default 300 s) ensures
  that a stale cache is refreshed within a bounded window.  The TTL is
  the trade-off between latency and freshness; set it to ``0`` to
  disable caching entirely and rebuild on every query.

* **Version counting** — Each cached entry stores the Qdrant scope count
  at build time.  Before reusing a cache entry, the engine compares the
  current count.  If the count differs, the cache is evicted even within
  the TTL.  This catches most additions and deletions without a full
  rescan — though not same-count replacements, which are caught by the
  TTL expiry.

Concurrency: a ``threading.Lock`` serialises cache reads/writes so that
concurrent ``retrieve()`` calls don't corrupt the cache dict.

Memory: ``bm25_cache_max_entries`` (default 16) caps the number of
cached scopes; the least-recently-used entry is evicted when the limit
is reached.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from .config import RetrievalConfig
from .keyword_search import BM25Retriever
from .vector_search import search_dense
from .vector_store import get_client, scope_filter
from .reranker import rerank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 scope cache
# ---------------------------------------------------------------------------

class _BM25CacheEntry:
    """One cached BM25 index for a specific scope and configuration."""

    __slots__ = ("retriever", "scope_count", "created_at")

    def __init__(self, retriever: BM25Retriever, scope_count: int):
        self.retriever = retriever
        self.scope_count = scope_count
        self.created_at: float = time.monotonic()


class _BM25Cache:
    """Thread-safe, scope-keyed, LRU-bounded cache for BM25Retriever.

    Each key is a tuple that uniquely identifies a scope *and* the BM25
    parameters that affect the index (k1, b, accent_insensitive).
    """

    def __init__(self):
        self._store: OrderedDict[tuple, _BM25CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(collection: str, corpus_id: str, workspace_id: str,
                  k1: float, b: float, accent_insensitive: bool) -> tuple:
        return (collection, corpus_id, workspace_id, k1, b, accent_insensitive)

    def get(self, key: tuple, *, ttl: float, current_count: int | None) -> BM25Retriever | None:
        """Return a cached retriever if the entry is still valid."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            # TTL expiry
            if ttl > 0 and (time.monotonic() - entry.created_at) > ttl:
                logger.debug("BM25 cache TTL expired for %s", key[:3])
                del self._store[key]
                return None
            # Version (scope count) check
            if current_count is not None and entry.scope_count != current_count:
                logger.info(
                    "BM25 cache invalidated for %s: scope count changed (%d -> %d)",
                    key[:3], entry.scope_count, current_count,
                )
                del self._store[key]
                return None
            # Mark as recently used
            self._store.move_to_end(key)
            return entry.retriever

    def put(self, key: tuple, retriever: BM25Retriever, scope_count: int,
            *, max_entries: int) -> None:
        """Insert or replace a cache entry, evicting LRU if over capacity."""
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = _BM25CacheEntry(retriever, scope_count)
            while len(self._store) > max(1, max_entries):
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("BM25 cache evicted LRU entry %s", evicted_key[:3])

    def invalidate(self, collection: str | None = None,
                   corpus_id: str | None = None,
                   workspace_id: str | None = None) -> int:
        """Remove matching entries.

        * No arguments → clear everything.
        * Only *collection* → clear all scopes in that collection.
        * *collection* + *corpus_id* + *workspace_id* → clear that exact
          scope (across all BM25 parameter variants).
        """
        with self._lock:
            if collection is None:
                count = len(self._store)
                self._store.clear()
                return count
            to_remove = []
            for key in self._store:
                k_col, k_corp, k_ws = key[:3]
                if k_col != collection:
                    continue
                if corpus_id is not None and k_corp != corpus_id:
                    continue
                if workspace_id is not None and k_ws != workspace_id:
                    continue
                to_remove.append(key)
            for key in to_remove:
                del self._store[key]
            return len(to_remove)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton.  Shared by all RetrievalEngine instances in the
# same process, which is the expected deployment pattern.
_bm25_cache = _BM25Cache()


def invalidate_bm25_cache(collection: str | None = None,
                          corpus_id: str | None = None,
                          workspace_id: str | None = None) -> int:
    """Public API for cache invalidation.

    Called automatically by ``indexing.build_index`` after a successful
    upsert/rebuild.  Callers that mutate Qdrant directly should call this
    explicitly.

    Returns the number of evicted entries.
    """
    count = _bm25_cache.invalidate(collection, corpus_id, workspace_id)
    if count:
        logger.info("BM25 cache invalidated: %d entries removed", count)
    return count


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(lists, *, k=60, top_k=20):
    if k < 0:
        raise ValueError("rrf_k must be nonnegative")
    combined: dict[str, list[Any]] = {}
    for results in lists:
        seen: set[str] = set()
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


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    def __init__(self, *, client=None, config=None, query_embedder=None, rerank_scorer=None):
        self.config = config or RetrievalConfig.from_env()
        self.client = client or get_client(config=self.config)
        self.query_embedder = query_embedder
        self.rerank_scorer = rerank_scorer

    def _bm25(self, query: str, corpus_id: str, workspace_id: str, top_k: int) -> list[dict]:
        scope = scope_filter(corpus_id, workspace_id)
        if not self.client.collection_exists(self.config.collection):
            return []

        cfg = self.config
        ttl = cfg.bm25_cache_ttl

        # When caching is disabled (ttl == 0), always rebuild.
        if ttl == 0:
            return self._build_bm25_from_qdrant(scope, corpus_id, workspace_id).search(query, top_k)

        cache_key = _BM25Cache._make_key(
            cfg.collection, corpus_id, workspace_id,
            cfg.bm25_k1, cfg.bm25_b, cfg.accent_insensitive,
        )

        # Lightweight version check: compare scope point count.
        current_count = self.client.count(
            cfg.collection, count_filter=scope, exact=True
        ).count

        cached = _bm25_cache.get(cache_key, ttl=ttl, current_count=current_count)
        if cached is not None:
            logger.debug("BM25 cache hit for %s/%s", corpus_id, workspace_id)
            return cached.search(query, top_k)

        # Cache miss — rebuild.
        logger.info("BM25 cache miss for %s/%s; building index", corpus_id, workspace_id)
        retriever = self._build_bm25_from_qdrant(scope, corpus_id, workspace_id)
        _bm25_cache.put(
            cache_key, retriever, current_count,
            max_entries=cfg.bm25_cache_max_entries,
        )
        return retriever.search(query, top_k)

    def _build_bm25_from_qdrant(self, scope, corpus_id: str, workspace_id: str) -> BM25Retriever:
        """Scroll Qdrant and build a fresh BM25 index for the given scope."""
        records: list[dict] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.config.collection, scroll_filter=scope, limit=256,
                offset=offset, with_payload=True, with_vectors=False,
            )
            records.extend(p.payload for p in points)
            if offset is None:
                break
        return BM25Retriever(
            records,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            accent_insensitive=self.config.accent_insensitive,
        )

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
        return rerank(
            query, fused,
            model_name=self.config.reranker_model,
            scorer=self.rerank_scorer,
            top_k=top_k,
            batch_size=self.config.reranker_batch_size,
            device=self.config.reranker_device,
            max_length=self.config.reranker_max_length,
        )
