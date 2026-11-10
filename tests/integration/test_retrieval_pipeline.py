import pytest
from qdrant_client import QdrantClient
from app.schemas.contracts import DocumentChunk
from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import ensure_collection

def embed(texts):
    return [[1., 0., 0.] if "tree" in t.lower() else [0., 1., 0.] for t in texts]

def docs():
    return [DocumentChunk(chunk_id="a", document_id="d", filename="tree.pdf", page=2, section="Tree", text="binary tree c\u00e2y nh\u1ecb ph\u00e2n", metadata={"lang":"vi"}), DocumentChunk(chunk_id="b", document_id="d", filename="heap.pdf", page=3, text="heap sort", metadata={})]

def test_end_to_end_and_isolation():
    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
    assert build_index(docs(), corpus_id="c", workspace_id="w", client=client, config=config, embedder=embed, dimension=3)["scope_count"] == 2
    build_index(docs(), corpus_id="c", workspace_id="other", client=client, config=config, embedder=embed, dimension=3)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda q: [1.,0.,0.], rerank_scorer=lambda pairs: [1.] * len(pairs))
    for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
        results = engine.retrieve("binary tree", method, corpus_id="c", workspace_id="w")
        assert results and all(c["workspace_id"] == "w" for c in results)
        first = next(c for c in results if c["chunk_id"] == "a")
        assert (first["filename"], first["page"], first["section"], first["metadata"]["lang"]) == ("tree.pdf", 2, "Tree", "vi")
    assert engine.retrieve("tree", "dense", corpus_id="other", workspace_id="w") == []
    with pytest.raises(ValueError): engine.retrieve("tree", "dense", corpus_id="c")
    build_index(docs()[:1], corpus_id="c", workspace_id="w", client=client, config=config, embedder=embed, dimension=3, rebuild=True)
    assert len(engine.retrieve("tree", "dense", corpus_id="c", workspace_id="other")) == 2
    assert len(engine.retrieve("tree", "dense", corpus_id="c", workspace_id="w")) == 1
    with pytest.raises(ValueError, match="Incompatible"): ensure_collection(client, 4, config=config)
    with pytest.raises(ValueError, match="Embedding model"): build_index(docs(), corpus_id="c", workspace_id="w", client=client, config=RetrievalConfig(embedding_model="different"), embedder=embed, dimension=3)


def test_benchmark_four_methods_and_unlabeled_rejection():
    from evaluation.scripts.run_benchmark import run
    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory", embedding_model="test")
    build_index(docs(), corpus_id="c", workspace_id="w", client=client, config=config, embedder=embed, dimension=3)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda q: [1., 0., 0.], rerank_scorer=lambda pairs: [1.] * len(pairs))
    results = run(engine, [{"query_id":"q1", "query":"binary tree", "relevant_ids":["a"]}], "c", "w")
    assert len(results) == 4
    assert all(r["recall_at_k"] == 1 for r in results)
    with pytest.raises(ValueError, match="labeled"): run(engine, [{"query":"tree", "relevant_ids":[]}], "c", "w")
