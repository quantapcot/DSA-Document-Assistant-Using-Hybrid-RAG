"""Stage 02 eight-combination (A1-A4 & B1-B4) smoke-test verification on data/runtime/qdrant_eval."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import torch
from app.retrieval import embeddings, reranker, vector_store
from app.retrieval.hybrid_search import RetrievalEngine

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.benchmark_builder import (
    EVAL_WORKSPACE_ID,
    SCOPE_BY_PROFILE,
    build_eval_retrieval_config,
)

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
COMBO_LABELS = {
    (PROFILE_A_ID, "bm25"): "A1",
    (PROFILE_A_ID, "dense"): "A2",
    (PROFILE_A_ID, "hybrid_rrf"): "A3",
    (PROFILE_A_ID, "hybrid_rerank"): "A4",
    (PROFILE_B_ID, "bm25"): "B1",
    (PROFILE_B_ID, "dense"): "B2",
    (PROFILE_B_ID, "hybrid_rrf"): "B3",
    (PROFILE_B_ID, "hybrid_rerank"): "B4",
}

SMOKE_QUERIES = [
    {"query_id": "smoke_vi_avl", "category": "vi_theory", "query": "Cây AVL là gì và các thao tác quay cây cân bằng hoạt động như thế nào?"},
    {"query_id": "smoke_en_bst", "category": "en_definition", "query": "What is a Binary Search Tree and how does search work?"},
    {"query_id": "smoke_code_sort", "category": "code_algorithm", "query": "Thuật toán Merge Sort chia để trị và cài đặt hàm merge như thế nào?"},
    {"query_id": "smoke_complexity", "category": "complexity", "query": "Độ phức tạp thời gian trường hợp xấu nhất của QuickSort là bao nhiêu?"},
]


def run_eight_combo_smoke(*, top_k: int = 15, candidate_pool: int = 30) -> dict:
    assert torch.cuda.is_available(), "RTX 5060 CUDA device must be available for GPU smoke test"
    cfg = build_eval_retrieval_config(candidate_top_k=candidate_pool)

    # Load expected chunk ID sets for strict cross-profile isolation checks
    corpus_ids_by_profile: dict[str, set[str]] = {}
    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        rows = [
            json.loads(line)
            for line in (ROOT / "evaluation" / "artifacts" / pid / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        corpus_ids_by_profile[pid] = {r["chunk_id"] for r in rows}

    # Pre-warm embedding & reranker on actual config devices (verify no OOM / silent CPU fallback)
    emb_model = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
    rer_model = reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
    actual_emb_dev = str(emb_model.device)
    actual_rer_dev = str(next(rer_model.model.parameters()).device)
    assert actual_emb_dev.startswith("cuda"), f"Unexpected CPU fallback for embeddings: {actual_emb_dev}"
    assert actual_rer_dev.startswith("cuda"), f"Unexpected CPU fallback for reranker: {actual_rer_dev}"

    engine = RetrievalEngine(config=cfg)
    combo_results = []
    try:
        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            corpus_id = SCOPE_BY_PROFILE[pid]
            other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
            for method in METHODS:
                combo = COMBO_LABELS[(pid, method)]
                latencies_ms = []
                query_summaries = []
                for item in SMOKE_QUERIES:
                    if method != "bm25":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    hits = engine.retrieve(
                        item["query"],
                        method,
                        top_k=top_k,
                        corpus_id=corpus_id,
                        workspace_id=EVAL_WORKSPACE_ID,
                    )
                    if method != "bm25":
                        torch.cuda.synchronize()
                    dt_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)
                    latencies_ms.append(dt_ms)

                    assert len(hits) > 0, f"Unexpected empty results for {combo} query={item['query_id']}"
                    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
                    for h in hits:
                        cid = h["chunk_id"]
                        assert cid in corpus_ids_by_profile[pid], f"{combo}: chunk {cid} missing from {pid} corpus!"
                        assert cid not in corpus_ids_by_profile[other_pid], f"{combo}: cross-profile leak {cid}!"
                        assert h.get("corpus_id") == corpus_id
                        assert h.get("workspace_id") == EVAL_WORKSPACE_ID
                        assert (h.get("metadata") or {}).get("profile_id") == pid
                        assert math.isfinite(float(h["score"]))
                        assert (h.get("metadata") or {}).get("source_spans")

                    query_summaries.append({
                        "query_id": item["query_id"],
                        "returned_count": len(hits),
                        "latency_ms": dt_ms,
                        "top1_chunk_id": hits[0]["chunk_id"],
                        "top1_score": round(float(hits[0]["score"]), 4),
                        "top1_filename": hits[0]["filename"],
                        "top1_page": hits[0]["page"],
                        "top1_content_type": (hits[0].get("metadata") or {}).get("content_type"),
                    })

                combo_results.append({
                    "combo": combo,
                    "profile_id": pid,
                    "corpus_id": corpus_id,
                    "method": method,
                    "mean_smoke_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
                    "queries": query_summaries,
                    "status": "PASS",
                })
    finally:
        engine.client.close()
        vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    return {
        "note": "Smoke test only (not official evaluation benchmark; no P/R/MRR/nDCG computed without Ground Truth).",
        "qdrant_path": cfg.qdrant_path,
        "collection": cfg.collection,
        "workspace_id": EVAL_WORKSPACE_ID,
        "top_k": top_k,
        "candidate_top_k": candidate_pool,
        "actual_devices": {"embedding": actual_emb_dev, "reranker": actual_rer_dev},
        "embedding_cache_info": embeddings._get_model.cache_info()._asdict(),
        "reranker_cache_info": reranker._get_model.cache_info()._asdict(),
        "combos": combo_results,
    }


def main() -> int:
    out = run_eight_combo_smoke()
    out_path = ROOT / "evaluation" / "artifacts" / "smoke_test_8_combos.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
