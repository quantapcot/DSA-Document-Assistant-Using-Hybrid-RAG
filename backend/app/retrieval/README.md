# Retrieval Engine — DSA Document Assistant

Unified Hybrid Retrieval Engine supporting four retrieval methods: `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank`.

---

## Quick Start for Backend Engineers

```python
import dataclasses
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

# 1. Load config (retrieval.mvp.yaml has BGE reranker + CUDA enabled)
config = RetrievalConfig.from_yaml("configs/retrieval.mvp.yaml")

# If you need to override fields programmatically (RetrievalConfig is frozen/immutable):
# config = dataclasses.replace(config, reranker_device="cpu")

# 2. Instantiate engine ONCE during app startup (keyword argument required)
engine = RetrievalEngine(config=config)

# 3. Retrieve evidence (query and method are the first two arguments)
results = engine.retrieve(
    query="Cây AVL là gì?",
    method="hybrid_rerank",   # Primary: "hybrid_rerank"; Fallback: "hybrid_rrf"
    top_k=5,
    corpus_id="benchmark-v1",
    workspace_id="research",
)

for r in results:
    print(f"#{r['rank']} [{r['filename']} p.{r['page']}] score={r['score']:.4f}")
    print(f"  {r['text'][:120]}...")
```

> [!TIP]
> A complete, copy-pasteable reference script is available at `tests/smoke_backend_integration.py`. Run it directly to verify your environment.

---

## Architecture Pipeline

```
User Query
    ├── 1. BM25 Lexical Retrieval (PyVi Vietnamese tokenizer + rank-bm25)
    │       ↓
    ├── 2. Dense Semantic Retrieval (MiniLM-L12-v2 + Qdrant Cosine)
    │       ↓
    └───→ 3. Reciprocal Rank Fusion (RRF, k=60)
              ↓
         Candidate Top-K (default k=10)
              ↓
         4. BGE Cross-Encoder Reranking (BAAI/bge-reranker-v2-m3)
              ↓
         Final Top-K Evidence Chunks
```

---

## Output Schema (Evidence Contract)

Every dictionary in the returned list contains the following standardized fields:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Unique chunk identifier |
| `document_id` | `str` | Parent document ID |
| `filename` | `str` | Source filename (PDF, DOCX, etc.) |
| `page` | `int` | 1-based page number |
| `section` | `str \| None` | Section heading or None |
| `text` | `str` | Chunk text content |
| `metadata` | `dict` | Document metadata (e.g. `lang`, custom tags) |
| `score` | `float` | Method-specific relevance score |
| `rank` | `int` | Position in result set (1 to top_k) |
| `retrieval_method` | `str` | `bm25`, `dense`, `hybrid_rrf`, or `hybrid_rerank` |
| `corpus_id` | `str` | Scope corpus identifier |
| `workspace_id` | `str` | Scope workspace identifier |

> [!WARNING]
> **Score interpretation:**
> Scores are relevance metrics, **not** probabilities or answer confidence:
> - `bm25`: BM25 term matching scores (positive unbounded).
> - `dense`: Cosine similarity in range `[-1.0, 1.0]`.
> - `hybrid_rrf`: Reciprocal Rank Fusion score $\sum \frac{1}{60 + r_i} \in (0.0, 0.033]$.
> - `hybrid_rerank`: Cross-Encoder relevance score/logit (higher = more relevant).

---

## Configuration & Tuning

### Config Files

- `configs/retrieval.example.yaml`: CPU-only safe default (no reranker model loaded).
- `configs/retrieval.mvp.yaml`: MVP configuration (CUDA GPU, `BAAI/bge-reranker-v2-m3` reranker, `candidate_top_k: 10`, `reranker_batch_size: 4`).

### Measured RTX 5060 Benchmark (8 GB VRAM)

Tested with `candidate_top_k: 10` (10 candidate pairs reranked):

| `reranker_batch_size` | Latency (10 pairs) | Peak VRAM | Notes |
|---|---|---|---|
| `1` | 128.75 ms | 2,177.1 MB | Ultra-conservative |
| `4` | 66.95 ms | 2,184.2 MB | **Recommended default** (~50% faster, +7 MB VRAM) |
| `10` | 52.84 ms | 2,196.0 MB | Single forward pass for top-10 |
| `16` | 57.90 ms | 2,196.0 MB | Fast batch inference |

Total GPU peak VRAM for entire system (Embedding + Reranker): **~2,634 MB** out of 8,192 MB capacity (~32% utilization).

---

## Device Handling & CUDA Verification

### Device Selection

`embedding_device` and `reranker_device` accept:
- `"cpu"`: Force CPU execution.
- `"cuda"`: Force CUDA GPU execution (fails with clear `RuntimeError` if unavailable).
- `"auto"`: Selects `"cuda"` if `torch.cuda.is_available()`, otherwise falls back to `"cpu"`.

### Transparency & Device Verification

The engine **never silently switches** from CUDA to CPU during inference.
When models are loaded, the actual PyTorch device is logged explicitly:
```
INFO app.retrieval.embeddings: Embedding model ... initialized successfully on actual device: cuda:0
INFO app.retrieval.reranker: Reranker model ... initialized successfully on actual device: cuda:0
```

### CUDA OOM Handling

If GPU runs out of VRAM during model loading or prediction:
1. A clear `RuntimeError` is raised indicating CUDA OOM.
2. The error message suggests reducing `reranker_batch_size`, decreasing `candidate_top_k`, or switching to `reranker_device: cpu`.

---

## BM25 Cache Invalidation & Multi-Process Freshness

BM25 indexes are heavy to compute from Qdrant payloads (~9 seconds for 5,941 chunks). They are cached **per-RetrievalEngine instance**.

### Same-Process (FastAPI app or monolithic script)
`build_index()` automatically calls `invalidate_all_bm25_caches()` upon completing indexing. No manual cache flushing is required.

### Cross-Process (Separate Indexing Worker / Background Process)
When indexing runs in a separate process from the API service:
1. **Dynamic Fingerprint:** On each query, `_scope_fingerprint()` computes a SHA-256 hash of the point count + sampled point IDs + sampled payload text hashes. Additions, deletions, and updates within the sample window trigger automatic rebuilds.
2. **TTL Expiry:** `bm25_cache_ttl` (default `300` seconds) ensures stale caches expire within 5 minutes even if point count and sample points remain unchanged.
3. **Guaranteed Freshness Mode:** For ingestion workflows requiring zero-staleness, set `bm25_cache_ttl: 0`. This disables caching entirely and rebuilds the BM25 index on every query.

---

## Fallback Strategy for Backend

If the Reranker model encounters an error or CUDA is temporarily unavailable:

```python
try:
    results = engine.retrieve(
        query=user_query,
        method="hybrid_rerank",
        top_k=5,
        corpus_id=corpus_id,
        workspace_id=workspace_id,
    )
except Exception as exc:
    logger.warning("Reranker failed (%s); falling back to hybrid_rrf", exc)
    results = engine.retrieve(
        query=user_query,
        method="hybrid_rrf",
        top_k=5,
        corpus_id=corpus_id,
        workspace_id=workspace_id,
    )
```

---

## Indexing API Reference

To index new documents programmatically:

```python
from app.retrieval.indexing import build_index
from app.retrieval.config import RetrievalConfig

config = RetrievalConfig.from_yaml("configs/retrieval.mvp.yaml")

# chunks: list of DocumentChunk dicts or path to JSONL file
result = build_index(
    chunks="data/corpus/dataset_clean/chunks_all.jsonl",
    corpus_id="benchmark-v1",
    workspace_id="research",
    config=config,
    rebuild=False,  # Set rebuild=True to replace existing scope
)
print(f"Indexed {result['indexed']} chunks; total in scope: {result['scope_count']}")
```

---

## Verification Commands

```powershell
# 1. Run unit & integration test suite (52 tests)
cd d:\Project\DSA-Document-Assistant-Using-Hybrid-RAG
$env:PYTHONPATH = "backend"
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe -m pytest tests/unit tests/integration -v `
    --basetemp "$env:USERPROFILE\pytest-dsa-temp-gpu" -p no:cacheprovider

# 2. Run backend integration smoke test (verifies YAML loading, engine init, real inference)
$env:HF_HUB_DISABLE_XET = "1"
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe tests/smoke_backend_integration.py

# 3. Run full 5-query reranker benchmark on real corpus
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe tests/smoke_reranker.py
```
