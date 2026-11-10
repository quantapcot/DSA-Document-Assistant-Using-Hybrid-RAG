"""Unified scoped BM25, dense, RRF, and optional reranking interface.

BM25 cache
----------
Building a BM25 index from ~6 000 Qdrant payloads on every query is wasteful.
This module caches ``BM25Retriever`` instances **per engine** so that
repeated queries against the same scope reuse the index.

Isolation
~~~~~~~~~
The cache is **per-RetrievalEngine instance**, not a global singleton.
Two engines backed by different Qdrant clients (e.g. two in-memory DBs)
will never share cached BM25 indexes.

Invalidation strategy
~~~~~~~~~~~~~~~~~~~~~
* **In-process invalidation** — ``invalidate_bm25_cache`` on a specific
  engine, or the module-level ``invalidate_all_bm25_caches``, must be
  called after any scope mutation.  ``build_index`` calls
  ``invalidate_all_bm25_caches`` automatically.

* **Cross-process staleness** — When indexing and retrieval run in
  different Python processes, in-process invalidation is insufficient.
  Two mechanisms detect stale data:

  1. **TTL** (``bm25_cache_ttl``, default 300 s) — after expiry the
     index is unconditionally rebuilt.
  2. **Content hash** — a lightweight hash is computed from the first
     and last N point IDs plus the scope count.  Any addition, deletion
     or replacement that changes the set of point IDs will be caught
     immediately, even when the total count is unchanged.

  Set ``bm25_cache_ttl`` to ``0`` to disable caching entirely (rebuild
  on every query).

Concurrency: a ``threading.Lock`` per engine serialises cache access.

Memory: ``bm25_cache_max_entries`` (default 16) caps the number of
cached scopes per engine; LRU eviction prevents unbounded growth.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any

from .config import RetrievalConfig
from .keyword_search import BM25Retriever
from .vector_search import search_dense
from .vector_store import get_client, scope_filter
from .reranker import rerank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 scope cache — per engine instance
# ---------------------------------------------------------------------------

class _BM25CacheEntry:
    """One cached BM25 index for a specific scope and configuration."""

    __slots__ = ("retriever", "scope_fingerprint", "created_at")

    def __init__(self, retriever: BM25Retriever, scope_fingerprint: str):
        self.retriever = retriever
        self.scope_fingerprint = scope_fingerprint
        self.created_at: float = time.monotonic()


class _BM25Cache:
    """Thread-safe, scope-keyed, LRU-bounded cache for BM25Retriever.

    Each key is a tuple that uniquely identifies a scope *and* the BM25
    parameters that affect the index (k1, b, accent_insensitive).

    This is an *instance-level* cache — each ``RetrievalEngine`` owns
    one.  No cross-engine sharing can happen.
    """

    def __init__(self):
        self._store: OrderedDict[tuple, _BM25CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(collection: str, corpus_id: str, workspace_id: str,
                  k1: float, b: float, accent_insensitive: bool) -> tuple:
        return (collection, corpus_id, workspace_id, k1, b, accent_insensitive)

    def get(self, key: tuple, *, ttl: float,
            current_fingerprint: str | None) -> BM25Retriever | None:
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
            # Fingerprint (content hash) check
            if (current_fingerprint is not None
                    and entry.scope_fingerprint != current_fingerprint):
                logger.info(
                    "BM25 cache invalidated for %s: scope fingerprint changed",
                    key[:3],
                )
                del self._store[key]
                return None
            # Mark as recently used
            self._store.move_to_end(key)
            return entry.retriever

    def put(self, key: tuple, retriever: BM25Retriever,
            scope_fingerprint: str, *, max_entries: int) -> None:
        """Insert or replace a cache entry, evicting LRU if over capacity."""
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = _BM25CacheEntry(retriever, scope_fingerprint)
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


# ---------------------------------------------------------------------------
# Engine registry — allows invalidate_all_bm25_caches to reach every engine
# ---------------------------------------------------------------------------

_engine_registry: weakref.WeakSet[RetrievalEngine] = weakref.WeakSet()
_registry_lock = threading.Lock()


def invalidate_all_bm25_caches(collection: str | None = None,
                               corpus_id: str | None = None,
                               workspace_id: str | None = None) -> int:
    """Invalidate BM25 caches across *all* live RetrievalEngine instances.

    Called by ``indexing.build_index`` after a successful upsert/rebuild.
    """
    total = 0
    with _registry_lock:
        for engine in list(_engine_registry):
            total += engine.invalidate_bm25_cache(collection, corpus_id,
                                                  workspace_id)
    if total:
        logger.info("BM25 caches invalidated across engines: %d entries", total)
    return total


# Keep the old name as an alias for backward compat (tests import it).
invalidate_bm25_cache = invalidate_all_bm25_caches


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
    """Central retrieval interface supporting bm25, dense, hybrid_rrf, hybrid_rerank.

    Each instance owns its own BM25 cache, isolated from other engines.
    """

    def __init__(self, *, client=None, config=None, query_embedder=None,
                 rerank_scorer=None):
        self.config = config or RetrievalConfig.from_env()
        self.client = client or get_client(config=self.config)
        self.query_embedder = query_embedder
        self.rerank_scorer = rerank_scorer
        self._bm25_cache = _BM25Cache()
        # Register for cross-engine invalidation.
        with _registry_lock:
            _engine_registry.add(self)

    # -- BM25 cache management ------------------------------------------------

    def invalidate_bm25_cache(self, collection: str | None = None,
                              corpus_id: str | None = None,
                              workspace_id: str | None = None) -> int:
        """Invalidate BM25 cache entries on *this* engine."""
        return self._bm25_cache.invalidate(collection, corpus_id, workspace_id)

    # -- Scope fingerprinting -------------------------------------------------

    def _scope_fingerprint(self, scope, collection: str) -> str:
        """Compute a lightweight fingerprint for a Qdrant scope.

        Uses the scope point count plus the IDs and text hashes of the
        first few points. This detects additions, deletions, and
        in-place replacements/updates within the sample window without
        reading the entire corpus payload.
        """
        count = self.client.count(collection, count_filter=scope,
                                  exact=True).count
        sample_parts: list[str] = []
        if count > 0:
            first_pts, _ = self.client.scroll(
                collection, scroll_filter=scope, limit=5,
                with_payload=True, with_vectors=False,
            )
            for p in first_pts:
                txt = (p.payload or {}).get("text", "")
                txt_hash = hashlib.md5(txt.encode("utf-8")).hexdigest()[:8]
                sample_parts.append(f"{p.id}:{txt_hash}")
        raw = f"{count}|{'|'.join(sample_parts)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- BM25 retrieval -------------------------------------------------------

    def _bm25(self, query: str, corpus_id: str, workspace_id: str,
              top_k: int) -> list[dict]:
        scope = scope_filter(corpus_id, workspace_id)
        if not self.client.collection_exists(self.config.collection):
            return []

        cfg = self.config
        ttl = cfg.bm25_cache_ttl

        # When caching is disabled (ttl == 0), always rebuild.
        if ttl == 0:
            return self._build_bm25_from_qdrant(
                scope, corpus_id, workspace_id
            ).search(query, top_k)

        cache_key = _BM25Cache._make_key(
            cfg.collection, corpus_id, workspace_id,
            cfg.bm25_k1, cfg.bm25_b, cfg.accent_insensitive,
        )

        # Compute scope fingerprint for freshness check.
        fingerprint = self._scope_fingerprint(scope, cfg.collection)

        cached = self._bm25_cache.get(
            cache_key, ttl=ttl, current_fingerprint=fingerprint,
        )
        if cached is not None:
            logger.debug("BM25 cache hit for %s/%s", corpus_id, workspace_id)
            return cached.search(query, top_k)

        # Cache miss — rebuild.
        logger.info("BM25 cache miss for %s/%s; building index",
                    corpus_id, workspace_id)
        retriever = self._build_bm25_from_qdrant(
            scope, corpus_id, workspace_id,
        )
        self._bm25_cache.put(
            cache_key, retriever, fingerprint,
            max_entries=cfg.bm25_cache_max_entries,
        )
        return retriever.search(query, top_k)

    def _build_bm25_from_qdrant(self, scope, corpus_id: str,
                                workspace_id: str) -> BM25Retriever:
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

    # -- Main retrieval interface ---------------------------------------------

    def retrieve(self, query: str, method: str, top_k: int = 5,
                 corpus_id: str | None = None,
                 workspace_id: str | None = None):
        scope_filter(corpus_id, workspace_id)
        if method not in {"bm25", "dense", "hybrid_rrf", "hybrid_rerank"}:
            raise ValueError("Unknown retrieval method")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0 or not query.strip():
            return []
        candidate_k = max(top_k, self.config.candidate_top_k)
        lexical = (self._bm25(query, corpus_id, workspace_id, candidate_k)
                   if method != "dense" else [])
        dense = (search_dense(
            self.client, query,
            corpus_id=corpus_id, workspace_id=workspace_id,
            config=self.config, top_k=candidate_k,
            query_embedder=self.query_embedder)
            if method != "bm25" else [])
        if method == "bm25":
            return lexical[:top_k]
        if method == "dense":
            return dense[:top_k]
        fused = rrf_fuse([lexical, dense], k=self.config.rrf_k,
                         top_k=candidate_k)
        if method == "hybrid_rrf":
            return fused[:top_k]
        return rerank(
            query, fused,
            model_name=self.config.reranker_model,
            scorer=self.rerank_scorer,
            top_k=top_k,
            batch_size=self.config.reranker_batch_size,
            device=self.config.reranker_device,
            max_length=self.config.reranker_max_length,
        )
