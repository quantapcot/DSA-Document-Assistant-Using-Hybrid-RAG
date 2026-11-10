"""Real-model smoke test: BGE Reranker v2-m3 on indexed DSA corpus.

This script loads the actual reranker model, runs 5 DSA queries through
all 4 retrieval methods, and reports latencies and GPU VRAM usage.

Usage (from repo root, PowerShell):
    $env:PYTHONPATH = "backend"
    python tests/smoke_reranker.py
"""
import logging
import sys
import time
from pathlib import Path

# Ensure backend is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

QUERIES = [
    ("en_bst", "What is a Binary Search Tree and its time complexity?"),
    ("vi_bst", "Cây nhị phân tìm kiếm là gì?"),
    ("en_avl", "Explain AVL Tree rotations and balancing"),
    ("en_bfs", "How does BFS traversal work on a graph?"),
    ("en_sort", "Compare Quick Sort and Merge Sort algorithms"),
]

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")

CONFIG_PATH = ROOT / "configs" / "retrieval.example.yaml"
CORPUS_ID = "benchmark-v1"
WORKSPACE_ID = "research"


def gpu_mem_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main():
    # Load config with reranker enabled
    config = RetrievalConfig(
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_model="BAAI/bge-reranker-v2-m3",
        collection="dsa_chunks_v1",
        qdrant_mode="local",
        qdrant_path=str(ROOT / "data" / "runtime" / "qdrant_local"),
        candidate_top_k=10,
        final_top_k=5,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        accent_insensitive=False,
        batch_size=32,
        embedding_device="cuda",
        reranker_device="cuda",
        reranker_batch_size=1,
        reranker_max_length=512,
        bm25_cache_ttl=300,
        bm25_cache_max_entries=16,
    )

    print("=" * 80)
    print("BGE Reranker v2-m3 Smoke Test on Real DSA Corpus")
    print("=" * 80)
    print(f"Embedding model : {config.embedding_model}")
    print(f"Reranker model  : {config.reranker_model}")
    print(f"Embedding device: {config.embedding_device}")
    print(f"Reranker device : {config.reranker_device}")
    print(f"Reranker batch  : {config.reranker_batch_size}")
    print(f"Max length      : {config.reranker_max_length}")
    print(f"Candidate top-k : {config.candidate_top_k}")
    print(f"Final top-k     : {config.final_top_k}")
    print(f"Corpus          : {CORPUS_ID} / {WORKSPACE_ID}")
    print()

    # Reset GPU peak memory tracking
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # Create engine (model loading happens lazily)
    t0 = time.perf_counter()
    engine = RetrievalEngine(config=config)
    engine_create_ms = (time.perf_counter() - t0) * 1000
    print(f"Engine created in {engine_create_ms:.0f} ms")
    print()

    # Track model load vs warm latencies
    model_loaded = {m: False for m in METHODS}
    all_results = {}

    for qid, query in QUERIES:
        print(f"--- Query: [{qid}] {query}")
        print()
        for method in METHODS:
            t_start = time.perf_counter()
            try:
                results = engine.retrieve(
                    query, method, top_k=5,
                    corpus_id=CORPUS_ID, workspace_id=WORKSPACE_ID,
                )
            except Exception as exc:
                print(f"  [{method}] ERROR: {exc}")
                continue
            latency_ms = (time.perf_counter() - t_start) * 1000

            load_note = ""
            if not model_loaded[method]:
                load_note = " (includes model loading)"
                model_loaded[method] = True

            print(f"  [{method}] {len(results)} results, {latency_ms:.0f} ms{load_note}")
            for r in results[:3]:
                text_snip = r["text"][:80].replace("\n", " ")
                print(f"    #{r['rank']} score={r['score']:.4f} "
                      f"chunk={r['chunk_id'][:20]} "
                      f"file={r['filename']} p.{r['page']} "
                      f"| {text_snip}...")
            if len(results) > 3:
                print(f"    ... and {len(results) - 3} more")

            all_results[(qid, method)] = results
        print()

    # GPU memory summary
    peak_vram = gpu_mem_mb()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"GPU Peak VRAM: {peak_vram:.0f} MB")
    print(f"Total queries: {len(QUERIES)} × {len(METHODS)} methods = {len(QUERIES) * len(METHODS)}")

    # Sanity checks
    errors = []
    for (qid, method), results in all_results.items():
        if not results:
            errors.append(f"{qid}/{method}: empty results")
            continue
        # Check no duplicates
        ids = [r["chunk_id"] for r in results]
        if len(ids) != len(set(ids)):
            errors.append(f"{qid}/{method}: duplicate chunk_ids")
        # Check scope
        for r in results:
            if r.get("workspace_id") != WORKSPACE_ID:
                errors.append(f"{qid}/{method}: wrong workspace_id")
                break
        # Check required fields
        for r in results:
            for field in ("chunk_id", "document_id", "filename", "page", "text",
                          "score", "rank", "retrieval_method"):
                if field not in r:
                    errors.append(f"{qid}/{method}: missing field '{field}'")
                    break

    if errors:
        print("\nSANITY CHECK FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("\n✓ All sanity checks passed")

    # Check if reranker actually changed rankings
    for qid, query in QUERIES:
        rrf_ids = [r["chunk_id"] for r in all_results.get((qid, "hybrid_rrf"), [])]
        rerank_ids = [r["chunk_id"] for r in all_results.get((qid, "hybrid_rerank"), [])]
        if rrf_ids and rerank_ids:
            changed = rrf_ids != rerank_ids
            print(f"  [{qid}] Reranker changed order: {'YES' if changed else 'NO'}")

    print()
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())