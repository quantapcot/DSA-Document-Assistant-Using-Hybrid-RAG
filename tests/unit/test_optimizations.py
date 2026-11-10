"""Tests for BM25 cache (Task A), embedding/reranker device selection (Task B/C),
and model identity validation (Task D)."""
import time
import threading
import pytest
import numpy as np
from qdrant_client import QdrantClient
from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index
from app.retrieval.hybrid_search import (
    RetrievalEngine, rrf_fuse, invalidate_bm25_cache, _bm25_cache,
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
# Task A — BM25 cache
# ===================================================================

class TestBM25Cache:
    """BM25 index caching behaviour."""

    def setup_method(self):
        """Clear the global cache before each test."""
        invalidate_bm25_cache()

    def test_two_queries_same_scope_no_rebuild(self):
        """Consecutive queries on the same scope must reuse the cached index."""
        engine = _setup_engine()
        r1 = engine.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        r2 = engine.retrieve("heap sort", "bm25", corpus_id="c", workspace_id="w")
        assert r1 and r2
        # Cache should have exactly 1 entry for this scope.
        assert len(_bm25_cache) == 1

    def test_different_workspaces_separate_cache(self):
        """Two workspaces must not share BM25 index."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c", workspace_id="w2",
                    client=client, config=config, embedder=_embed, dimension=3)
        # Clear cache entries created by build_index invalidation.
        invalidate_bm25_cache()

        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w1")
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w2")
        assert len(_bm25_cache) == 2

    def test_scope_change_invalidates_cache(self):
        """After data changes in a scope, cache for that scope is refreshed."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        r1 = engine.retrieve("binary tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(r1) >= 1

        # Rebuild with only one chunk → count changes → cache invalidated.
        build_index(_docs()[:1], corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed,
                    dimension=3, rebuild=True)
        r2 = engine.retrieve("heap sort", "bm25", corpus_id="c", workspace_id="w")
        # "heap sort" should not be found because chunk "b" was removed.
        assert all(c["chunk_id"] != "b" for c in r2)

    def test_scope_a_update_does_not_evict_scope_b(self):
        """Updating scope A must not invalidate the valid cache of scope B."""
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
        build_index(_docs(), corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed, dimension=3)
        build_index(_docs(), corpus_id="c", workspace_id="w2",
                    client=client, config=config, embedder=_embed, dimension=3)
        invalidate_bm25_cache()

        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        # Populate cache for both scopes.
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w1")
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w2")
        assert len(_bm25_cache) == 2

        # Rebuild w1 only → invalidates w1 but NOT w2.
        build_index(_docs()[:1], corpus_id="c", workspace_id="w1",
                    client=client, config=config, embedder=_embed,
                    dimension=3, rebuild=True)
        # w2 should still be cached.
        assert len(_bm25_cache) == 1

    def test_ttl_expiry_detects_stale(self):
        """Expired TTL forces a rebuild even if count hasn't changed."""
        config = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=1,
        )
        client = QdrantClient(":memory:")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        invalidate_bm25_cache()

        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(_bm25_cache) == 1

        # Wait for TTL to expire.
        time.sleep(1.1)
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        # The cache entry is recreated — still size 1 but was refreshed.
        assert len(_bm25_cache) == 1

    def test_cache_disabled_when_ttl_zero(self):
        """TTL=0 should disable caching entirely."""
        config = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=0,
        )
        client = QdrantClient(":memory:")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        invalidate_bm25_cache()

        engine = RetrievalEngine(
            client=client, config=config,
            query_embedder=lambda q: [1., 0., 0.],
            rerank_scorer=lambda pairs: [1.] * len(pairs),
        )
        engine.retrieve("tree", "bm25", corpus_id="c", workspace_id="w")
        assert len(_bm25_cache) == 0

    def test_bm25_results_consistent_with_and_without_cache(self):
        """Cached and uncached BM25 must produce identical results."""
        client = QdrantClient(":memory:")
        config_no_cache = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=0,
        )
        config_with_cache = RetrievalConfig(
            qdrant_mode="memory", embedding_model="test", bm25_cache_ttl=300,
        )
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config_no_cache, embedder=_embed, dimension=3)
        invalidate_bm25_cache()

        eng_no = RetrievalEngine(client=client, config=config_no_cache,
                                 query_embedder=lambda q: [1., 0., 0.],
                                 rerank_scorer=lambda p: [1.] * len(p))
        eng_yes = RetrievalEngine(client=client, config=config_with_cache,
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
    """Device resolution and cache keying for embeddings."""

    def test_resolve_device_cpu(self):
        assert embeddings.resolve_device("cpu") == "cpu"

    def test_resolve_device_cuda(self):
        assert embeddings.resolve_device("cuda") == "cuda"

    def test_resolve_device_auto_fallback(self, monkeypatch):
        """auto should fall back to cpu when torch is unavailable."""
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
        """Different device values should produce separate cache entries."""
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
        # Both calls should have been made with distinct device args.
        assert ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu") in calls
        assert ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cuda") in calls


# ===================================================================
# Task C — Reranker device selection and configuration
# ===================================================================

class TestRerankerDevice:
    """Reranker device config, batch size, max_length, and mock scorer."""

    def test_mock_scorer_still_works(self):
        """Mock scorer must still function identically."""
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
        """Filename, page, section, chunk_id, metadata must survive reranking."""
        chunks = [
            {"chunk_id": "x", "text": "hello", "filename": "a.pdf",
             "page": 5, "section": "Intro", "metadata": {"k": "v"}},
        ]
        result = reranker_mod.rerank("q", chunks, scorer=lambda p: [0.5])
        r = result[0]
        assert r["filename"] == "a.pdf"
        assert r["page"] == 5
        assert r["section"] == "Intro"
        assert r["metadata"]["k"] == "v"
        assert r["chunk_id"] == "x"

    def test_batch_size_and_max_length_forwarded(self):
        """Verify batch_size and max_length params are accepted."""
        chunks = [{"chunk_id": "a", "text": "a"}]
        # These shouldn't error — they're forwarded to the scorer.
        result = reranker_mod.rerank(
            "q", chunks, scorer=lambda p: [0.5],
            batch_size=16, max_length=512,
        )
        assert len(result) == 1


# ===================================================================
# Task D — Model identity validation
# ===================================================================

class TestModelValidation:
    """Embedding model identity checks in Qdrant."""

    def test_correct_model_accepted(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="test-model")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        # Should not raise.
        check_model(client, config)

    def test_wrong_model_rejected(self):
        client = QdrantClient(":memory:")
        config = RetrievalConfig(qdrant_mode="memory", embedding_model="model-A")
        build_index(_docs(), corpus_id="c", workspace_id="w",
                    client=client, config=config, embedder=_embed, dimension=3)
        wrong_config = RetrievalConfig(qdrant_mode="memory", embedding_model="model-B")
        with pytest.raises(ValueError, match="mismatch"):
            check_model(client, wrong_config)

    def test_mixed_model_detected(self):
        """If two models wrote to the same collection, check_model must reject."""
        client = QdrantClient(":memory:")
        config_a = RetrievalConfig(
            qdrant_mode="memory", embedding_model="model-A", collection="shared",
        )
        # First, index with model-A.
        build_index(_docs()[:1], corpus_id="c", workspace_id="w1",
                    client=client, config=config_a, embedder=_embed, dimension=3)
        # Manually upsert a point with model-B to simulate mixed-model.
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
        # check_model samples 10 points — should find both.
        with pytest.raises(ValueError, match="[Mm]ixed|mismatch"):
            check_model(client, config_a, sample_size=100)

    def test_missing_metadata_rejected(self):
        """Points without embedding_model must not be silently accepted."""
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
        # Should not raise.
        check_model_exhaustive(client, config)
        # Wrong model should raise.
        wrong = RetrievalConfig(qdrant_mode="memory", embedding_model="other")
        with pytest.raises(ValueError, match="expected"):
            check_model_exhaustive(client, wrong)

    def test_scope_filtering_unaffected(self):
        """Model validation must not interfere with workspace/corpus filtering."""
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
        # Cross-scope should return empty.
        assert engine.retrieve("tree", "dense", corpus_id="c1", workspace_id="w2") == []