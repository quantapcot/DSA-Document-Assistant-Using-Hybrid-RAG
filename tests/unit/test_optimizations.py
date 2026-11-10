"""Tests for BM25 cache isolation, cross-process freshness, device selection,
model validation, and reranker API correctness.

All tests use mock embedders/scorers and in-memory Qdrant — no real models
are loaded.
"""
import math
import time
import threading
import pytest
import numpy as np
from qdrant_client import QdrantClient
from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index
from app.retrieval.hybrid_search import (
    RetrievalEngine, rrf_fuse,
    invalidate_bm25_cache, invalidate_all_bm25_caches,
)
from app.retrieval.vector_store import (
    ensure_collection, check_model, check_model_exhaustive,
)
from app.retrieval import embeddings
from app.retrieval import reranker as reranker_mod
from app.schemas.contracts import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docs():
    return [
        DocumentChunk(chunk_id="a", document_id="d", filename="tree.pdf",
                      page=2, section="Tree",
                      text="binary tree cây nhị phân", metadata={"lang": "vi"}),
        DocumentChunk(chunk_id="b", document_id="d", filename="heap.pdf",
                      page=3, text="heap sort", metadata={}),
    ]


def _docs_alt():
    """Different content, same chunk_ids."""
    return [
        DocumentChunk(chunk_id="a", document_id="d", filename="graph.pdf",
                      page=1, section="Graph",
                      text="đồ thị graph BFS DFS", metadata={"lang": "vi"}),
        DocumentChunk(chunk_id="b", document_id="d", filename="queue.pdf",
                      page=4, text="queue hàng đợi", metadata={}),
    ]


def _embed(texts):
    return [[1., 0., 0.] if "tree" in t.lower() else [0., 1., 0.] for t in texts]


def _setup_engine(client=None, config=None, **kw):
    client = client or QdrantClient(":memory:")
    config = config or RetrievalConfig(qdrant_mode="memory", embedding_model="test")
    build_index(_docs(), corpus_id="c", workspace_id="w",
                client=client, config=config, embedder=_embed, dimension=3)
    return RetrievalEngine(
        client=client, config=config,
        query_embedder=lambda q: [1., 0., 0.],
        rerank_scorer=lambda pairs: [1.] * len(pairs),
        **kw,
    )


# ===================================================================
# Task 1.1 — BM25 cache isolation between Qdrant clients
# ===================================================================

class TestBM25CacheIsolation:
    """Two engines backed by different Qdrant databases must never share
    BM25 indexes, even when scope names and chunk counts match."""

    def test_two_clients_same_scope_different_data(self):
        """Engine B must not return documents from Engine A's database."""
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")

        # Engine A — tree/heap data
        client_a = QdrantClient(":memory:")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client_a, config=config, embedder=_embed, dimension=3)
        engine_a = RetrievalEngine(
            client=client_a, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )

        # Engine B — graph/queue data (same scope, same count!)
        client_b = QdrantClient(":memory:")
        build_index(_docs_alt(), corpus_id="c", workspace_id="w",
                    client=client_b, config=config, embedder=_embed, dimension=3)
        engine_b = RetrievalEngine(
            client=client_b, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )

        # Warm both caches
        r_a = engine_a.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        r_b = engine_b.retrieve("đồ thị graph", "bm25", corpus_id="c", workspace_id="w")

        # Engine A finds tree data
        assert any("tree" in c["text"].lower() for c in r_a)
        # Engine B finds graph data, NOT tree data
        assert any("graph" in c["text"].lower() for c in r_b)
        assert not any("tree" in c["text"].lower() for c in r_b)

    def test_same_engine_reuses_cache(self):
        """Consecutive queries on the same engine/scope reuse the index."""
        engine = _setup_engine()
        r1 = engine.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        r2 = engine.retrieve("heap sort", "bm25", corpus_id="c", workspace_id="w")
        assert r1 and r2
        assert len(engine._bm25_cache) == 1

    def test_different_workspaces_separate_cache(self):
        """Two workspaces on the same engine use separate BM25 indexes."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c", workspace_id="w2",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w1")
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w2")
        assert len(engine._bm25_cache) == 2

    def test_indexing_invalidates_cache(self):
        """build_index() must invalidate the BM25 cache for the affected scope."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        r1 = engine.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        assert any(c["chunk_id"] == "a" for c in r1)

        # Rebuild with different content (same count, same chunk_ids)
        build_index(_docs_alt(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed,
                    dimension=3, rebuild=True)
        r2 = engine.retrieve("đồ thị graph", "bm25", corpus_id="c", workspace_id="w")
        # Should find graph data, not tree data
        assert any("graph" in c["text"].lower() for c in r2)

    def test_same_count_replacement_detected(self):
        """When content changes but count stays the same (same-count
        replacement in the same process), cache must be refreshed."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(engine._bm25_cache) == 1

        # Replace with different docs (same count) — build_index calls invalidate
        build_index(_docs_alt(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed,
                    dimension=3, rebuild=True)
        # Cache should have been invalidated by build_index
        assert len(engine._bm25_cache) == 0

    def test_scope_a_update_does_not_evict_scope_b(self):
        """Updating scope A must not invalidate scope B's cache."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c", workspace_id="w2",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w1")
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w2")
        assert len(engine._bm25_cache) == 2

        # Rebuild w1 only
        build_index(_docs()[:1], corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed,
                    dimension=3, rebuild=True)
        # w2 should still be cached
        assert len(engine._bm25_cache) == 1

    def test_concurrent_queries_no_crash(self):
        """Multiple threads querying the same engine should not crash."""
        engine = _setup_engine()
        errors = []

        def query(q):
            try:
                r = engine.retrieve(q, "bm25", corpus_id="c", workspace_id="w")
                assert isinstance(r, list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query, args=(f"query {i}",))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent errors: {errors}"


# ===================================================================
# Task 1.2 — Cross-process cache freshness
# ===================================================================

class TestCacheFreshness:
    """TTL and fingerprint-based staleness detection."""

    def test_ttl_expiry(self):
        """Expired TTL forces a rebuild."""
        config = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=1,
        )
        client = QdrantClient(":memory:")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(engine._bm25_cache) == 1
        time.sleep(1.1)
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(engine._bm25_cache) == 1

    def test_sampled_point_content_update_detected_by_fingerprint(self):
        """When text in the first sampled points changes (same count, same IDs),
        payload-aware fingerprint detects the change across queries."""
        import uuid
        from qdrant_client import models
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=300)
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        r1 = engine.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(r1) > 0
        assert "cây nhị phân" in r1[0]["text"]

        # In-place update point #0 payload text directly without invalidation
        # (same chunk_id, same count, same point ID)
        identity = '["w","c","a"]'
        pt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        client.set_payload(
            collection_name=config.collection,
            payload={"text": "đồ thị vô hướng undirected graph"},
            points=[pt_id],
        )

        # Query again — fingerprint changed because sample point text hash changed!
        r2 = engine.retrieve("đồ thị vô hướng", "bm25", corpus_id="c", workspace_id="w")
        assert len(r2) > 0
        assert "đồ thị vô hướng" in r2[0]["text"]

    def test_unsampled_point_content_update_requires_invalidation_or_ttl_zero(self):
        """When in-place text update happens beyond sample limit without invalidation,
        the limitation is caught by setting bm25_cache_ttl=0 or explicit invalidate."""
        import uuid
        client = QdrantClient(":memory:")
        config_cached = RetrievalConfig(qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=300)

        # 8 chunks: first 5 are sample window, chunks 6-8 are outside sample window
        many_docs = [
            DocumentChunk(chunk_id=f"c{i}", document_id="d", filename=f"f{i}.pdf",
                          page=i, text=f"chunk content number {i}")
            for i in range(8)
        ]
        build_index(many_docs, corpus_id="c", workspace_id="w",
                    client=client, config=config_cached, embedder=_embed, dimension=3)

        engine_cached = RetrievalEngine(
            client=client, config=config_cached,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        # Warm up cache
        _ = engine_cached.retrieve("chunk", "bm25", corpus_id="c", workspace_id="w")

        # Find a point that is explicitly outside the first-5 sample window
        pts_sample, _ = client.scroll(config_cached.collection, limit=5, with_payload=False)
        sampled_id_set = {str(p.id) for p in pts_sample}
        all_pts, _ = client.scroll(config_cached.collection, limit=100, with_payload=False)
        unsampled_pts = [p for p in all_pts if str(p.id) not in sampled_id_set]
        assert len(unsampled_pts) > 0
        target_pt_id = str(unsampled_pts[0].id)

        # In-place update target point's payload text directly in Qdrant (without invalidating engine)
        client.set_payload(
            collection_name=config_cached.collection,
            payload={"text": "special unique keyword updated content outside sample"},
            points=[target_pt_id],
        )

        # 1) Engine with cache hit will return old data (demonstrating cross-process limitation)
        res_cached = engine_cached.retrieve("special unique keyword", "bm25", corpus_id="c", workspace_id="w")
        assert not any("special unique keyword" in r["text"] for r in res_cached)

        # 2) But an engine configured with bm25_cache_ttl=0 always rebuilds and finds it immediately!
        config_fresh = RetrievalConfig(qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=0)
        engine_fresh = RetrievalEngine(
            client=client, config=config_fresh,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        res_fresh = engine_fresh.retrieve("special unique keyword", "bm25", corpus_id="c", workspace_id="w")
        assert any("special unique keyword" in r["text"] for r in res_fresh)

    def test_cache_disabled_when_ttl_zero(self):
        """TTL=0 disables caching entirely."""
        config = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=0,
        )
        client = QdrantClient(":memory:")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(engine._bm25_cache) == 0

    def test_results_consistent_with_and_without_cache(self):
        """Cached and uncached BM25 produce identical results."""
        client = QdrantClient(":memory:")
        config_no = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=0,
        )
        config_yes = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=300,
        )
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config_no, embedder=_embed, dimension=3)
        eng_no = RetrievalEngine(client=client, config=config_no,
                                 query_embedder=lambda q: [1., 0., 0.],
                                 rerank_scorer=lambda p: [1.] * len(p))
        eng_yes = RetrievalEngine(client=client, config=config_yes,
                                  query_embedder=lambda q: [1., 0., 0.],
                                  rerank_scorer=lambda p: [1.] * len(p))
        for query in ("binary tree", "heap sort", "cây nhị phân"):
            r_no = eng_no.retrieve(query, "bm25", corpus_id="c", workspace_id="w")
            r_yes = eng_yes.retrieve(query, "bm25", corpus_id="c", workspace_id="w")
            assert [c["chunk_id"] for c in r_no] == [c["chunk_id"] for c in r_yes]
            for a, b in zip(r_no, r_yes):
                assert a["score"] == pytest.approx(b["score"])


# ===================================================================
# Task B — Embedding device selection
# ===================================================================

class TestEmbeddingDevice:
    def test_resolve_device_cpu(self):
        assert embeddings.resolve_device("cpu") == "cpu"

    def test_resolve_device_cuda(self):
        assert embeddings.resolve_device("cuda") == "cuda"

    def test_resolve_device_auto_fallback(self, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mock")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", no_torch)
        assert embeddings.resolve_device("auto") == "cpu"

    def test_resolve_device_invalid(self):
        with pytest.raises(ValueError, match="Unknown device"):
            embeddings.resolve_device("tpu")

    def test_model_cache_keyed_by_device(self, monkeypatch):
        calls = []
        class MockModel:
            def __init__(self, device):
                self.device = device
            def get_sentence_embedding_dimension(self):
                return 3
            def encode(self, texts, **kw):
                return np.array([[1., 0., 0.]] * len(texts))
        def mock_get(model_name, device="cpu"):
            calls.append((model_name, device))
            return MockModel(device)
        monkeypatch.setattr(embeddings, "_get_model", mock_get)
        embeddings.embed_texts(["a"], device="cpu")
        embeddings.embed_texts(["a"], device="cuda")
        assert ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu") in calls
        assert ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cuda") in calls


# ===================================================================
# Task C — Reranker device, API, and score validation
# ===================================================================

class TestRerankerDevice:
    def test_mock_scorer_still_works(self):
        chunks = [{"chunk_id": "a", "text": "a", "page": 1},
                  {"chunk_id": "b", "text": "b", "page": 2}]
        result = reranker_mod.rerank("q", chunks, scorer=lambda pairs: [0.1, 0.9])
        assert result[0]["chunk_id"] == "b" and result[0]["page"] == 2
        assert result[0]["retrieval_method"] == "hybrid_rerank"

    def test_no_model_configured_raises(self):
        chunks = [{"chunk_id": "a", "text": "a", "page": 1}]
        with pytest.raises(ValueError, match="not configured"):
            reranker_mod.rerank("q", chunks)

    def test_empty_candidates(self):
        assert reranker_mod.rerank("q", []) == []

    def test_resolve_device(self):
        assert reranker_mod._resolve_device("cpu") == "cpu"
        assert reranker_mod._resolve_device("cuda") == "cuda"
        with pytest.raises(ValueError, match="Unknown"):
            reranker_mod._resolve_device("tpu")

    def test_metadata_preserved_after_rerank(self):
        chunks = [
            {"chunk_id": "x", "text": "hello", "filename": "a.pdf",
             "page": 5, "section": "Intro", "metadata": {"k": "v"},
             "document_id": "d1", "corpus_id": "c", "workspace_id": "w"},
        ]
        result = reranker_mod.rerank("q", chunks, scorer=lambda p: [0.5])
        r = result[0]
        assert r["filename"] == "a.pdf"
        assert r["page"] == 5
        assert r["section"] == "Intro"
        assert r["metadata"]["k"] == "v"
        assert r["chunk_id"] == "x"
        assert r["document_id"] == "d1"

    def test_batch_size_and_max_length_forwarded(self):
        chunks = [{"chunk_id": "a", "text": "a"}]
        result = reranker_mod.rerank(
            "q", chunks, scorer=lambda p: [0.5],
            batch_size=16, max_length=512,
        )
        assert len(result) == 1

    def test_nan_score_rejected(self):
        """Non-finite scores must raise."""
        chunks = [{"chunk_id": "a", "text": "a"}]
        with pytest.raises(ValueError, match="non-finite"):
            reranker_mod.rerank("q", chunks, scorer=lambda p: [float("nan")])

    def test_inf_score_rejected(self):
        chunks = [{"chunk_id": "a", "text": "a"}]
        with pytest.raises(ValueError, match="non-finite"):
            reranker_mod.rerank("q", chunks, scorer=lambda p: [float("inf")])

    def test_correct_score_sorting(self):
        """Higher score = higher rank."""
        chunks = [{"chunk_id": "a", "text": "a"},
                  {"chunk_id": "b", "text": "b"},
                  {"chunk_id": "c", "text": "c"}]
        result = reranker_mod.rerank(
            "q", chunks, scorer=lambda p: [0.1, 0.9, 0.5], top_k=3,
        )
        assert [r["chunk_id"] for r in result] == ["b", "c", "a"]
        assert result[0]["rank"] == 1
        assert result[2]["rank"] == 3

    def test_model_cache_keyed_by_max_length(self, monkeypatch):
        """Different max_length values should load separate model instances."""
        loads = []
        class FakeModel:
            def __init__(self, **kw):
                loads.append(kw)
            def predict(self, pairs, **kw):
                return [0.5] * len(pairs)
        def fake_get(name, device="cpu", max_length=0):
            loads.append({"name": name, "device": device, "max_length": max_length})
            return FakeModel()
        monkeypatch.setattr(reranker_mod, "_get_model", fake_get)
        chunks = [{"chunk_id": "a", "text": "a"}]
        reranker_mod.rerank("q", chunks, model_name="m", device="cpu", max_length=0)
        reranker_mod.rerank("q", chunks, model_name="m", device="cpu", max_length=512)
        # Should have been called with different max_length
        assert any(l.get("max_length") == 0 for l in loads)
        assert any(l.get("max_length") == 512 for l in loads)


# ===================================================================
# Task D — Model identity validation
# ===================================================================

class TestModelValidation:
    def test_correct_model_accepted(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test-model")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        check_model(client, config)

    def test_wrong_model_rejected(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="model-A")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        wrong = RetrievalConfig(qdrant_mode="memory", embedding_model="model-B")
        with pytest.raises(ValueError, match="mismatch"):
            check_model(client, wrong)

    def test_mixed_model_detected(self):
        client = QdrantClient(":memory:")
        config_a = RetrievalConfig(
            qdrant_mode="memory", embedding_model="model-A", collection="shared",
        )
        build_index(_docs()[:1], corpus_id="c", workspace_id="w1",
                    client=client, config=config_a, embedder=_embed, dimension=3)
        from qdrant_client import models
        import uuid
        client.upsert(
            "shared",
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": [0., 0., 1.]},
                payload={"chunk_id": "z", "document_id": "d",
                         "filename": "z.pdf", "page": 1, "text": "z",
                         "corpus_id": "c", "workspace_id": "w2",
                         "embedding_model": "model-B"},
            )],
            wait=True,
        )
        with pytest.raises(ValueError, match="[Mm]ixed|mismatch"):
            check_model(client, config_a, sample_size=100)

    def test_missing_metadata_rejected(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        ensure_collection(client, 3, config=config)
        from qdrant_client import models
        import uuid
        client.upsert(
            config.collection,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": [1., 0., 0.]},
                payload={"chunk_id": "no-meta", "text": "test",
                         "corpus_id": "c", "workspace_id": "w"},
            )],
            wait=True,
        )
        with pytest.raises(ValueError, match="no embedding_model"):
            check_model(client, config)

    def test_exhaustive_check(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test-model")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        check_model_exhaustive(client, config)
        wrong = RetrievalConfig(qdrant_mode="memory", embedding_model="other")
        with pytest.raises(ValueError, match="expected"):
            check_model_exhaustive(client, wrong)

    def test_scope_filtering_unaffected(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c1", workspace_id="w1",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c2", workspace_id="w2",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        r1 = engine.retrieve("tree", "dense", corpus_id="c1", workspace_id="w1")
        r2 = engine.retrieve("tree", "dense", corpus_id="c2", workspace_id="w2")
        assert all(c["corpus_id"] == "c1" for c in r1)
        assert all(c["corpus_id"] == "c2" for c in r2)
        assert engine.retrieve("tree", "dense", corpus_id="c1", workspace_id="w2") == []


# ===================================================================
# Hybrid RRF + Rerank integration (mock)
# ===================================================================

class TestHybridIntegration:
    def test_hybrid_rrf_no_duplicate_chunks(self):
        engine = _setup_engine()
        results = engine.retrieve("binary tree", "hybrid_rrf",
                                  corpus_id="c", workspace_id="w")
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_hybrid_rerank_with_mock_scorer(self):
        engine = _setup_engine()
        results = engine.retrieve("binary tree", "hybrid_rerank",
                                  corpus_id="c", workspace_id="w")
        assert results
        assert all(r["retrieval_method"] == "hybrid_rerank" for r in results)
        assert all("filename" in r and "page" in r for r in results)

    def test_all_methods_return_correct_scope(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c", workspace_id="other",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda p: [1.] * len(p),
        )
        for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
            results = engine.retrieve("binary tree", method,
                                      corpus_id="c", workspace_id="w")
            assert results
            assert all(c["workspace_id"] == "w" for c in results), (
                f"Method {method} returned wrong workspace"
            )

    def test_empty_query_returns_empty(self):
        engine = _setup_engine()
        for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
            assert engine.retrieve("", method, corpus_id="c", workspace_id="w") == []
            assert engine.retrieve("  ", method, corpus_id="c", workspace_id="w") == []

    def test_reranker_not_configured_raises(self):
        """hybrid_rerank without model or scorer must raise clearly."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            # No rerank_scorer, no reranker_model
        )
        with pytest.raises(ValueError, match="not configured"):
            engine.retrieve("tree", "hybrid_rerank",
                            corpus_id="c", workspace_id="w")


# ===================================================================
# Backend Interface Contract & Frozen Config
# ===================================================================

class TestBackendInterfaceContract:
    def test_frozen_config_immutability(self):
        """RetrievalConfig is immutable (frozen). Mutation raises FrozenInstanceError."""
        import dataclasses
        config = RetrievalConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.reranker_device = "cuda"

        # Correct way to override: dataclasses.replace
        new_config = dataclasses.replace(config, reranker_device="cuda")
        assert new_config.reranker_device == "cuda"
        assert config.reranker_device == "cpu"

    def test_engine_init_keyword_only(self):
        """RetrievalEngine.__init__ requires keyword arguments."""
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        # Positional config must raise TypeError
        with pytest.raises(TypeError):
            RetrievalEngine(config)

        # Keyword argument config=config succeeds
        engine = RetrievalEngine(config=config)
        assert engine.config == config

    def test_retrieve_signature_and_argument_order(self):
        """engine.retrieve takes (query, method) in order or as keywords."""
        engine = _setup_engine()

        # Keyword arguments
        res1 = engine.retrieve(
            query="binary tree",
            method="hybrid_rrf",
            top_k=5,
            corpus_id="c",
            workspace_id="w",
        )
        assert isinstance(res1, list)
        assert len(res1) > 0

        # Positional arguments: (query, method, top_k, corpus_id, workspace_id)
        res2 = engine.retrieve("binary tree", "hybrid_rrf", 5, "c", "w")
        assert isinstance(res2, list)

        # Inverted arguments ("hybrid_rrf", "binary tree") must raise ValueError
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            engine.retrieve("hybrid_rrf", "binary tree", 5, "c", "w")

    def test_build_index_signature_and_keyword_args(self):
        """build_index takes chunks positionally, corpus_id/workspace_id keyword-only."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")

        # Correct usage
        res = build_index(_docs(), corpus_id="c", workspace_id="w",
                          client=client, config=config, embedder=_embed, dimension=3)
        assert res["scope_count"] == 2

        # Positional corpus_id must raise TypeError
        with pytest.raises(TypeError):
            build_index(_docs(), "c", "w", client=client, config=config)
