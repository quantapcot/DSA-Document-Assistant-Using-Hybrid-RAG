"""Backend Integration Smoke Test & Reference Implementation.

This script tests the exact contract and code pattern that Backend developers
should copy into their service (e.g. FastAPI startup and query handlers).

Usage:
    $env:PYTHONPATH = "backend"
    python tests/smoke_backend_integration.py
"""
import dataclasses
import logging
import sys
from pathlib import Path

# Add backend to sys.path so 'app.retrieval.*' imports cleanly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("backend_integration_smoke")


def main():
    print("=" * 80)
    print("RUNNING BACKEND INTEGRATION SMOKE TEST")
    print("=" * 80)

    # 1. Load configuration from YAML
    config_path = ROOT / "configs" / "retrieval.mvp.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = RetrievalConfig.from_yaml(config_path)
    print(f"Loaded config: embedding={config.embedding_model}, reranker={config.reranker_model}")
    print(f"Devices: embedding={config.embedding_device}, reranker={config.reranker_device}")

    # Note: RetrievalConfig is frozen (immutable dataclass).
    # If Backend needs to override parameters in code, use dataclasses.replace():
    # config = dataclasses.replace(config, reranker_device="cpu")

    # 2. Instantiate RetrievalEngine (MUST use keyword argument 'config=config')
    # Note: Create this ONCE during backend startup (e.g., in FastAPI lifespan / dependency)
    engine = RetrievalEngine(config=config)
    print("RetrievalEngine initialized successfully (reusable singleton).")

    # 3. Query: hybrid_rerank (Production primary method)
    corpus_id = "benchmark-v1"
    workspace_id = "research"
    query_en = "What is a Binary Search Tree and its time complexity?"

    print(f"\n--- Testing engine.retrieve with method='hybrid_rerank' ---")
    results_rerank = engine.retrieve(
        query=query_en,
        method="hybrid_rerank",
        top_k=3,
        corpus_id=corpus_id,
        workspace_id=workspace_id,
    )
    assert len(results_rerank) > 0, "Expected at least 1 result"
    top_result = results_rerank[0]
    print(f"Top-1 result: rank={top_result['rank']}, score={top_result['score']:.4f}")
    print(f"File: {top_result['filename']} (p.{top_result['page']})")
    print(f"Snippet: {top_result['text'][:120]}...")

    # Validate output schema contract
    required_keys = {
        "chunk_id", "document_id", "filename", "page", "section",
        "text", "metadata", "score", "rank", "retrieval_method",
        "corpus_id", "workspace_id",
    }
    for res in results_rerank:
        missing = required_keys - set(res.keys())
        assert not missing, f"Result missing keys: {missing}"
        assert res["retrieval_method"] == "hybrid_rerank"
        assert res["corpus_id"] == corpus_id
        assert res["workspace_id"] == workspace_id

    # 4. Query: hybrid_rrf (Fallback method when reranker is disabled or errored)
    query_vi = "Cây nhị phân tìm kiếm là gì?"
    print(f"\n--- Testing engine.retrieve with fallback method='hybrid_rrf' ---")
    results_rrf = engine.retrieve(
        query=query_vi,
        method="hybrid_rrf",
        top_k=3,
        corpus_id=corpus_id,
        workspace_id=workspace_id,
    )
    assert len(results_rrf) > 0
    assert results_rrf[0]["retrieval_method"] == "hybrid_rrf"
    print(f"Top-1 RRF result: rank={results_rrf[0]['rank']}, score={results_rrf[0]['score']:.4f}")
    print(f"File: {results_rrf[0]['filename']} (p.{results_rrf[0]['page']})")

    # 5. Verify Scope Isolation: non-existent workspace returns empty
    empty_res = engine.retrieve(
        query="tree",
        method="hybrid_rerank",
        top_k=3,
        corpus_id=corpus_id,
        workspace_id="unauthorized_workspace_xyz",
    )
    assert len(empty_res) == 0, "Expected empty results for unauthorized workspace"
    print("\nScope isolation check: PASS (non-existent/unauthorized workspace returns 0 results)")

    print("\n" + "=" * 80)
    print("ALL BACKEND INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()