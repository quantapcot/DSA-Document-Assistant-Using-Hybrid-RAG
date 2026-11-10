# Retrieval engine (C1)

Four methods: `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank`. BM25 uses rank-bm25 and PyVi with NFC and casefold; accent folding is opt-in and may create false positives. Dense vectors use cosine similarity in Qdrant. RRF combines ranks, not raw scores. Scores are method-specific relevance scores, not probabilities.

## Install and configure

Use Python 3.11 in the `dsa-rag` environment: `python -m pip install -r requirements.txt` and `python -m pip install pytest` for tests. This installs sentence-transformers and its PyTorch dependencies. The embedding model is loaded lazily on first dense indexing/search and may need a download. Existing embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; the Proposal names `sentence-transformers/all-MiniLM-L6-v2`. Changing models requires a new collection and rebuilding the index, including when dimensions happen to match. This implementation checks model identity stored in payloads and rejects an unknown/incompatible collection. Do not share this collection with other writers.

Copy `configs/retrieval.example.yaml` and pass `--config PATH` to the CLI or `RetrievalConfig.from_yaml(PATH)` in code. By default it uses a local Qdrant folder under `data/runtime/qdrant_local`; `qdrant_mode: memory` is for tests, and `qdrant_mode: server` requires `qdrant_url`. Without a YAML config, `QDRANT_MODE`, `QDRANT_PATH`, and `QDRANT_URL` select the connection. An existing QDRANT_URL never activates HTTP in local mode.

## CPU / GPU device selection

Both the embedding model and the reranker support configurable devices via `embedding_device` and `reranker_device`:

| Value  | Behaviour |
|--------|-----------|
| `cpu`  | Always use CPU (default, backwards-compatible) |
| `cuda` | Force CUDA; fails if CUDA unavailable |
| `auto` | Use CUDA if `torch.cuda.is_available()`, else CPU |

Set in `configs/retrieval.example.yaml` or pass to `RetrievalConfig`:

```yaml
embedding_device: auto   # Use GPU if available
reranker_device: auto
```

Models are cached by `(model_name, device)` — switching device does not reload a model that is already loaded on the target device. GPU and CPU produce the same-dimension vectors from the same model; existing Qdrant indexes do not need rebuilding.

## BM25 cache

BM25 indexes are cached per scope `(collection, corpus_id, workspace_id)` and BM25 parameters (`k1`, `b`, `accent_insensitive`). This avoids rebuilding the index from Qdrant payloads on every query.

**Invalidation:**

- **Same process:** `build_index()` automatically calls `invalidate_bm25_cache()` after upserting. Manual callers who modify Qdrant directly should call `invalidate_bm25_cache(collection, corpus_id, workspace_id)`.
- **Cross-process:** When indexing and retrieval run in separate processes (e.g. background worker + API server), in-process invalidation is insufficient. The cache uses a **TTL** (`bm25_cache_ttl`, default 300 seconds) and a **version check** (scope point count) to detect staleness. The TTL caps maximum staleness; the count check catches most additions/deletions immediately.
- **Disable caching:** Set `bm25_cache_ttl: 0` to rebuild on every query (original behaviour).

**Memory control:** `bm25_cache_max_entries` (default 16) limits the number of scope caches; LRU eviction prevents unbounded growth.

**Limitation:** If a document is replaced with a same-count update in a different process, the change may not be detected until the TTL expires. For a course project deployment where indexing and retrieval share a process, this is not an issue.

## Model identity validation

`check_model()` now samples up to 10 points (configurable via `sample_size`) instead of checking only the first point. It:

- Detects mixed-model collections (different `embedding_model` values).
- Rejects points with missing `embedding_model` metadata (fail-closed).
- Reports a clear error distinguishing mismatch vs. mixed vs. missing.

For one-time audits, `check_model_exhaustive()` scans every point.

## Index

Run from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = "backend"
python -m app.retrieval.indexing data/corpus/dataset_clean/chunks_all.jsonl --corpus-id benchmark-v1 --workspace-id research --config configs/retrieval.example.yaml
```

Assign distinct corpus and workspace identifiers to benchmark and runtime data. Upsert replaces matching stable scope/chunk IDs, leaving other chunks intact. `--rebuild` replaces only the specified scope, but is not transactional: interruption after deletion requires retry. Input is never changed. Supply new/changed chunks to add/update; removed source chunks remain until a scope rebuild. Duplicate IDs within input are rejected. BM25 is now cached per scope with TTL-based invalidation (see above).

## Backend use

```python
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

config = RetrievalConfig.from_yaml("configs/retrieval.example.yaml")
engine = RetrievalEngine(config=config)
for method in ("bm25", "dense", "hybrid_rrf"):
    evidence = engine.retrieve("binary tree", method, top_k=5,
                               corpus_id="benchmark-v1", workspace_id="research")
    print(method, evidence)
# Once a cross-encoder has been approved and reranker_model configured:
# print(engine.retrieve("binary tree", "hybrid_rerank", corpus_id="benchmark-v1", workspace_id="research"))
```

The engine requires both scope IDs for every method and applies both filters at storage. The caller must authenticate the user and authorize those IDs; IDs alone are not an access-control system. Returned dictionaries retain DocumentChunk fields, metadata, score, rank, and retrieval_method. Connection/model errors propagate, rather than silently returning an empty corpus. An empty corpus or query returns no results. Do not open multiple local clients against the same path concurrently; reuse the client/engine.

The reranker interface and mock scorer work; no real reranker is selected or downloaded. Proposal originally points to English-oriented MS MARCO. A potential multilingual choice, `BAAI/bge-reranker-v2-m3`, has approximately 2.27 GB FP32 weights and needs additional CPU RAM and inference time; it is not approved or benchmarked. See https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main. Set `reranker_model` only after explicit model approval. Test with a mock scorer to avoid downloads. The reranker now supports `reranker_batch_size`, `reranker_max_length`, and GPU via `reranker_device`.

## Tests and evaluation

```powershell
python -m pytest tests/unit tests/integration -q --basetemp "$env:USERPROFILE\pytest-dsa-temp" -p no:cacheprovider
python evaluation/scripts/run_benchmark.py --queries labeled.jsonl --corpus data/corpus/dataset_clean/chunks_all.jsonl --corpus-id benchmark-v1 --workspace-id research --config configs/retrieval.example.yaml --split held-out-test --output evaluation/results/run.json
```

Each JSONL query needs `query_id`, `query`, and a nonempty list of manually validated `relevant_ids` (chunk IDs). The default benchmark requires an approved reranker; pass `--methods bm25 dense hybrid_rrf` to omit it. `run_ablation.py` has the same arguments. Results include per-query and mean Precision@5, Recall@5, reciprocal rank within the returned top 5, and latency; JSON or CSV plus a metadata sidecar are supported. Record a real split name and do not present tuning queries as independent test data. Corpus SHA-256 records the supplied file; the runner does not verify it matches the index. Cold-start/model loading and in-memory BM25 rebuilding affect latency. Do not publish metrics without validated ground truth and a controlled protocol.
