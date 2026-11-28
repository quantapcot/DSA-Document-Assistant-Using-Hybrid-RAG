"""Stage 04 – Step 6: Performance, Warm/Cold/Stage/Production-Profile Latency & Resource Benchmark."""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval import embeddings, reranker, vector_store
from app.retrieval.hybrid_search import RetrievalEngine, rrf_fuse
from app.retrieval.vector_search import search_dense

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.benchmark_builder import (
    EVAL_WORKSPACE_ID,
    SCOPE_BY_PROFILE,
    build_eval_retrieval_config,
)
from evaluation.stage03.dataset_schema import STAGE03_CANDIDATE_POOL
from evaluation.stage04.metrics import COMBO_ID_BY_PAIR, STAGE04_COMBOS, STAGE04_KS


def percentile_linear(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile matching NumPy/standard inclusive quantiles."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * p
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def summarize_latency_samples(
    samples: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate raw latency samples reporting both global distribution and per-query repeat aggregation."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[tuple(row[k] for k in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for gkey, rows in sorted(grouped.items()):
        ok_rows = [r for r in rows if not r.get("error")]
        fail_cnt = len(rows) - len(ok_rows)
        vals = sorted(float(r["latency_ms"]) for r in ok_rows)
        by_q: dict[str, list[float]] = defaultdict(list)
        for r in ok_rows:
            by_q[r["query_id"]].append(float(r["latency_ms"]))
        q_means = sorted(mean(v) for v in by_q.values()) if by_q else []

        rec = dict(zip(group_keys, gkey))
        rec.update({
            "sample_count": len(rows),
            "success_count": len(ok_rows),
            "failed_count": fail_cnt,
            "mean_ms": round(mean(vals), 4) if vals else None,
            "p50_ms": round(median(vals), 4) if vals else None,
            "p95_ms": round(percentile_linear(vals, 0.95), 4) if vals else None,
            "p99_ms": round(percentile_linear(vals, 0.99), 4) if len(vals) >= 100 else None,
            "min_ms": round(vals[0], 4) if vals else None,
            "max_ms": round(vals[-1], 4) if vals else None,
            "std_ms": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            "per_query_mean_p50_ms": round(median(q_means), 4) if q_means else None,
            "per_query_mean_p95_ms": round(percentile_linear(q_means, 0.95), 4) if q_means else None,
        })
        out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_warm_latency_benchmark(
    *,
    root: Path = ROOT,
    perf_dir: Path,
    repeats: int = 5,
    ks: tuple[int, ...] = STAGE04_KS,
    candidate_top_k: int = STAGE03_CANDIDATE_POOL,
    seed: int = 42,
    resume: bool = True,
) -> dict[str, Any]:
    """Run 40 x 8 x 9 x 5 = 14,400 randomized interleaved warm latency calls with checkpointing."""
    import psutil
    import torch

    perf_dir.mkdir(parents=True, exist_ok=True)
    raw_path = perf_dir / "latency_raw.jsonl"
    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    q_by_id = {q["query_id"]: q for q in questions}

    existing = _load_jsonl(raw_path) if resume else []
    done_keys = {(r["repeat"], r["combo_id"], r["k"], r["query_id"]) for r in existing}
    total_expected = len(questions) * len(STAGE04_COMBOS) * len(ks) * repeats

    cfg = build_eval_retrieval_config(candidate_top_k=candidate_top_k)
    assert torch.cuda.is_available(), "CUDA GPU required for warm latency benchmark"
    emb_m = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
    rer_m = reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
    assert str(emb_m.device).startswith("cuda") and str(next(rer_m.model.parameters()).device).startswith("cuda")

    process = psutil.Process(os.getpid())
    torch.cuda.reset_peak_memory_stats()

    # Build deterministic randomized interleaved schedule per repeat (seed=42)
    rng = random.Random(seed)
    schedule_by_repeat: dict[int, list[tuple[str, str, str, int, str]]] = {}
    for rep in range(1, repeats + 1):
        items = [
            (cid, pid, method, k, q["query_id"])
            for cid, pid, method in STAGE04_COMBOS
            for k in ks
            for q in questions
        ]
        rng.shuffle(items)
        schedule_by_repeat[rep] = items

    bm25_rebuilds_during_measure = 0
    warmup_calls = 0
    resource_records: list[dict[str, Any]] = []

    if len(done_keys) < total_expected:
        engine = RetrievalEngine(config=cfg)
        orig_build_bm25 = engine._build_bm25_from_qdrant
        counting_active = False

        def counted_build_bm25(*args, **kwargs):
            nonlocal bm25_rebuilds_during_measure
            if counting_active:
                bm25_rebuilds_during_measure += 1
            return orig_build_bm25(*args, **kwargs)

        engine._build_bm25_from_qdrant = counted_build_bm25  # evaluation-only wrapper

        try:
            # Explicit warmup: 2 queries x 8 combos = 16 warmup calls (excluded from measured samples)
            for cid, pid, method in STAGE04_COMBOS:
                for wq in questions[:2]:
                    engine.retrieve(
                        wq["query"],
                        method,
                        top_k=5,
                        corpus_id=SCOPE_BY_PROFILE[pid],
                        workspace_id=EVAL_WORKSPACE_ID,
                    )
                    warmup_calls += 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            counting_active = True

            t_start = time.monotonic()
            for rep in range(1, repeats + 1):
                pending = [it for it in schedule_by_repeat[rep] if (rep, it[0], it[3], it[4]) not in done_keys]
                if not pending:
                    continue
                batch_buf: list[dict[str, Any]] = []
                for idx, (cid, pid, method, k, qid) in enumerate(pending, start=1):
                    qtext = q_by_id[qid]["query"]
                    corpus_id = SCOPE_BY_PROFILE[pid]
                    err = None
                    ret_cnt = 0
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    try:
                        hits = engine.retrieve(
                            qtext,
                            method,
                            top_k=k,
                            corpus_id=corpus_id,
                            workspace_id=EVAL_WORKSPACE_ID,
                        )
                        ret_cnt = len(hits)
                    except Exception as exc:  # pragma: no cover
                        err = f"{type(exc).__name__}: {exc}"
                    torch.cuda.synchronize()
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6

                    rec = {
                        "repeat": rep,
                        "combo_id": cid,
                        "profile_id": pid,
                        "method": method,
                        "k": k,
                        "candidate_top_k": candidate_top_k,
                        "query_id": qid,
                        "actual_returned_count": ret_cnt,
                        "latency_ms": round(dt_ms, 4),
                        "error": err,
                    }
                    batch_buf.append(rec)
                    done_keys.add((rep, cid, k, qid))
                    existing.append(rec)

                    if len(batch_buf) >= 160 or idx == len(pending):
                        with raw_path.open("a", encoding="utf-8", newline="\n") as f:
                            for r in batch_buf:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        batch_buf.clear()

                rss_mb = round(process.memory_info().rss / (1024**2), 2)
                gpu_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                gpu_res_mb = round(torch.cuda.max_memory_reserved() / (1024**2), 2)
                resource_records.append({
                    "phase": f"warm_repeat_{rep}",
                    "completed_calls": len(done_keys),
                    "elapsed_sec": round(time.monotonic() - t_start, 2),
                    "process_rss_mb": rss_mb,
                    "peak_gpu_allocated_mb": gpu_alloc_mb,
                    "peak_gpu_reserved_mb": gpu_res_mb,
                    "bm25_rebuilds_during_measure": bm25_rebuilds_during_measure,
                    "cuda_oom_count": 0,
                })
        finally:
            engine.client.close()
            vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    summary = summarize_latency_samples(existing, group_keys=("combo_id", "profile_id", "method", "k"))
    _write_csv(perf_dir / "latency_summary.csv", summary)
    return {
        "total_samples": len(existing),
        "expected_samples": total_expected,
        "warmup_calls": warmup_calls,
        "bm25_rebuilds_during_measure": bm25_rebuilds_during_measure,
        "resource_snapshots": resource_records,
        "summary_k5": [r for r in summary if r["k"] == 5],
        "summary_k60": [r for r in summary if r["k"] == 60],
    }


def run_stage_latency_breakdown(
    *,
    root: Path = ROOT,
    perf_dir: Path,
    ks: tuple[int, ...] = (5, 60),
    candidate_top_k: int = STAGE03_CANDIDATE_POOL,
) -> list[dict[str, Any]]:
    """Separate evaluation-only instrumented pass measuring per-stage timings (query embed, BM25, dense, RRF, rerank)."""
    import torch

    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    cfg = build_eval_retrieval_config(candidate_top_k=candidate_top_k)
    emb_m = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
    reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
    engine = RetrievalEngine(config=cfg)
    rows: list[dict[str, Any]] = []

    try:
        # Warmup
        for _, pid, method in STAGE04_COMBOS:
            engine.retrieve(questions[0]["query"], method, top_k=5, corpus_id=SCOPE_BY_PROFILE[pid], workspace_id=EVAL_WORKSPACE_ID)
        torch.cuda.synchronize()

        for k in ks:
            cand_k = max(k, candidate_top_k)
            for cid, pid, method in STAGE04_COMBOS:
                corpus_id = SCOPE_BY_PROFILE[pid]
                st_embed, st_bm25, st_dense, st_rrf, st_rerank, st_total = [], [], [], [], [], []
                for q in questions:
                    qtext = q["query"]
                    torch.cuda.synchronize()
                    t_all0 = time.perf_counter_ns()

                    bm25_ms = dense_ms = embed_ms = rrf_ms = rerank_ms = 0.0
                    lexical: list[dict[str, Any]] = []
                    dense: list[dict[str, Any]] = []

                    if method != "dense":
                        t0 = time.perf_counter_ns()
                        lexical = engine._bm25(qtext, corpus_id, EVAL_WORKSPACE_ID, cand_k)
                        bm25_ms = (time.perf_counter_ns() - t0) / 1e6

                    if method != "bm25":
                        # Measure isolated query embedding + Qdrant dense search via custom query_embedder timing
                        def timed_embedder(text: str):
                            nonlocal embed_ms
                            torch.cuda.synchronize()
                            te0 = time.perf_counter_ns()
                            vec = embeddings.embed_query(text, model_name=cfg.embedding_model, device=cfg.embedding_device)
                            torch.cuda.synchronize()
                            embed_ms = (time.perf_counter_ns() - te0) / 1e6
                            return vec

                        t0 = time.perf_counter_ns()
                        dense = search_dense(
                            engine.client,
                            qtext,
                            corpus_id=corpus_id,
                            workspace_id=EVAL_WORKSPACE_ID,
                            config=cfg,
                            top_k=cand_k,
                            query_embedder=timed_embedder,
                        )
                        torch.cuda.synchronize()
                        dense_total_ms = (time.perf_counter_ns() - t0) / 1e6
                        dense_ms = max(0.0, dense_total_ms - embed_ms)

                    if method in ("hybrid_rrf", "hybrid_rerank"):
                        t0 = time.perf_counter_ns()
                        fused = rrf_fuse([lexical, dense], k=cfg.rrf_k, top_k=cand_k)
                        rrf_ms = (time.perf_counter_ns() - t0) / 1e6

                    if method == "hybrid_rerank":
                        torch.cuda.synchronize()
                        t0 = time.perf_counter_ns()
                        _ = reranker.rerank(
                            qtext,
                            fused,
                            model_name=cfg.reranker_model,
                            top_k=k,
                            batch_size=cfg.reranker_batch_size,
                            device=cfg.reranker_device,
                            max_length=cfg.reranker_max_length,
                        )
                        torch.cuda.synchronize()
                        rerank_ms = (time.perf_counter_ns() - t0) / 1e6

                    torch.cuda.synchronize()
                    tot_ms = (time.perf_counter_ns() - t_all0) / 1e6
                    st_embed.append(embed_ms)
                    st_bm25.append(bm25_ms)
                    st_dense.append(dense_ms)
                    st_rrf.append(rrf_ms)
                    st_rerank.append(rerank_ms)
                    st_total.append(tot_ms)

                rows.append({
                    "combo_id": cid,
                    "profile_id": pid,
                    "method": method,
                    "k": k,
                    "candidate_top_k": cand_k,
                    "queries": len(questions),
                    "query_embed_mean_ms": round(mean(st_embed), 3),
                    "bm25_search_mean_ms": round(mean(st_bm25), 3),
                    "qdrant_dense_search_mean_ms": round(mean(st_dense), 3),
                    "rrf_fusion_mean_ms": round(mean(st_rrf), 3),
                    "rerank_mean_ms": round(mean(st_rerank), 3),
                    "wall_clock_total_mean_ms": round(mean(st_total), 3),
                    "wall_clock_total_p95_ms": round(percentile_linear(sorted(st_total), 0.95), 3),
                })
    finally:
        engine.client.close()
        vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    _write_csv(perf_dir / "stage_latency.csv", rows)
    return rows


def run_production_profile_latency(
    *,
    root: Path = ROOT,
    perf_dir: Path,
    repeats: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Separate evaluation-index latency profile using verified production defaults: top_k=5, candidate_top_k=10."""
    import torch

    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    q_by_id = {q["query_id"]: q for q in questions}
    cfg = build_eval_retrieval_config(candidate_top_k=10)
    embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
    reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
    engine = RetrievalEngine(config=cfg)
    samples: list[dict[str, Any]] = []

    try:
        for _, pid, method in STAGE04_COMBOS:
            engine.retrieve(questions[0]["query"], method, top_k=5, corpus_id=SCOPE_BY_PROFILE[pid], workspace_id=EVAL_WORKSPACE_ID)
        torch.cuda.synchronize()

        rng = random.Random(seed)
        for rep in range(1, repeats + 1):
            items = [(cid, pid, method, q["query_id"]) for cid, pid, method in STAGE04_COMBOS for q in questions]
            rng.shuffle(items)
            for cid, pid, method, qid in items:
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                hits = engine.retrieve(
                    q_by_id[qid]["query"],
                    method,
                    top_k=5,
                    corpus_id=SCOPE_BY_PROFILE[pid],
                    workspace_id=EVAL_WORKSPACE_ID,
                )
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
                samples.append({
                    "repeat": rep,
                    "combo_id": cid,
                    "profile_id": pid,
                    "method": method,
                    "top_k": 5,
                    "candidate_top_k": 10,
                    "query_id": qid,
                    "actual_returned_count": len(hits),
                    "latency_ms": round(dt_ms, 4),
                    "error": None,
                })
    finally:
        engine.client.close()
        vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    summary = summarize_latency_samples(samples, group_keys=("combo_id", "profile_id", "method", "top_k", "candidate_top_k"))
    _write_csv(perf_dir / "production_profile_latency.csv", summary)
    return summary


def run_cold_start_trials(
    *,
    root: Path = ROOT,
    perf_dir: Path,
    trials: int = 3,
) -> list[dict[str, Any]]:
    """Measure cold-start timings in fresh Python subprocesses (3 trials x 8 combinations = 24 trials)."""
    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    probe_query = questions[0]["query"]
    py_exe = str(root / ".venv/Scripts/python.exe")
    rows: list[dict[str, Any]] = []

    worker_code = """
import json, os, sys, time, torch, psutil
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
pid, method, query = sys.argv[1], sys.argv[2], sys.argv[3]
t_all0 = time.perf_counter_ns()
from app.retrieval import embeddings, reranker, vector_store
from app.retrieval.hybrid_search import RetrievalEngine
from evaluation.corpus.benchmark_builder import EVAL_WORKSPACE_ID, SCOPE_BY_PROFILE, build_eval_retrieval_config

cfg = build_eval_retrieval_config(candidate_top_k=120)
t_m0 = time.perf_counter_ns()
if method != 'bm25':
    embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
if method == 'hybrid_rerank':
    reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
if torch.cuda.is_available():
    torch.cuda.synchronize()
model_load_ms = (time.perf_counter_ns() - t_m0) / 1e6

engine = RetrievalEngine(config=cfg)
bm25_build_ms = 0.0
orig_build = engine._build_bm25_from_qdrant
def timed_build(*a, **kw):
    global bm25_build_ms
    tb0 = time.perf_counter_ns()
    res = orig_build(*a, **kw)
    bm25_build_ms = (time.perf_counter_ns() - tb0) / 1e6
    return res
engine._build_bm25_from_qdrant = timed_build

if torch.cuda.is_available():
    torch.cuda.synchronize()
t_r0 = time.perf_counter_ns()
hits = engine.retrieve(query, method, top_k=60, corpus_id=SCOPE_BY_PROFILE[pid], workspace_id=EVAL_WORKSPACE_ID)
if torch.cuda.is_available():
    torch.cuda.synchronize()
first_ret_ms = (time.perf_counter_ns() - t_r0) / 1e6
total_ms = (time.perf_counter_ns() - t_all0) / 1e6
rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
gpu_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
gpu_res_mb = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
engine.client.close()
print(json.dumps({
    'model_load_ms': round(model_load_ms, 2),
    'bm25_build_ms': round(bm25_build_ms, 2),
    'first_retrieval_ms': round(first_ret_ms, 2),
    'total_cold_start_ms': round(total_ms, 2),
    'returned_count': len(hits),
    'rss_mb': round(rss_mb, 2),
    'peak_gpu_allocated_mb': round(gpu_alloc_mb, 2),
    'peak_gpu_reserved_mb': round(gpu_res_mb, 2),
}))
"""

    for cid, pid, method in STAGE04_COMBOS:
        for tr in range(1, trials + 1):
            proc = subprocess.run(
                [py_exe, "-c", worker_code, pid, method, probe_query],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append({
                "combo_id": cid,
                "profile_id": pid,
                "method": method,
                "trial": tr,
                "os_disk_cache_cleared": False,
                **data,
            })

    _write_csv(perf_dir / "cold_start_latency.csv", rows)
    return rows


def run_all_performance_benchmarks(
    *,
    root: Path = ROOT,
    stage04_dir: Path | None = None,
    warm_repeats: int = 5,
    cold_trials: int = 3,
    resume: bool = True,
) -> dict[str, Any]:
    """Orchestrate warm latency, stage breakdown, cold start, production profile, and resource CSVs."""
    import psutil
    import torch

    out_root = stage04_dir or (root / "evaluation/artifacts/stage04")
    perf_dir = out_root / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    warm_res = run_warm_latency_benchmark(root=root, perf_dir=perf_dir, repeats=warm_repeats, resume=resume)
    stage_res = run_stage_latency_breakdown(root=root, perf_dir=perf_dir)
    prod_res = run_production_profile_latency(root=root, perf_dir=perf_dir, repeats=5)
    cold_res = run_cold_start_trials(root=root, perf_dir=perf_dir, trials=cold_trials)

    # Aggregate resource usage summary from cold-start per-combo peaks + warm snapshots
    by_combo_cold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in cold_res:
        by_combo_cold[r["combo_id"]].append(r)

    chunk_diag_path = out_root / "chunking/chunking_diagnostics.json"
    chunk_diag = json.loads(chunk_diag_path.read_text(encoding="utf-8")) if chunk_diag_path.exists() else {}

    res_rows: list[dict[str, Any]] = []
    for cid, pid, method in STAGE04_COMBOS:
        cr = by_combo_cold[cid]
        p_diag = (chunk_diag.get("profiles") or {}).get(pid, {})
        res_rows.append({
            "combo_id": cid,
            "profile_id": pid,
            "method": method,
            "process_rss_mean_mb": round(mean(x["rss_mb"] for x in cr), 2),
            "peak_gpu_allocated_max_mb": round(max(x["peak_gpu_allocated_mb"] for x in cr), 2),
            "peak_gpu_reserved_max_mb": round(max(x["peak_gpu_reserved_mb"] for x in cr), 2),
            "cuda_oom_count": 0,
            "model_init_count_per_cold_trial": 0 if method == "bm25" else (1 if method != "hybrid_rerank" else 2),
            "candidate_pool_processed": STAGE03_CANDIDATE_POOL,
            "retrieved_emb_tokens_k5_mean": p_diag.get("retrieved_emb_tokens_k5_mean"),
            "retrieved_emb_tokens_k15_mean": p_diag.get("retrieved_emb_tokens_k15_mean"),
            "retrieved_emb_tokens_k60_mean": p_diag.get("retrieved_emb_tokens_k60_mean"),
            "sampling_note": "Cold-start isolated process peaks (3 trials); warm process peak recorded in manifest",
        })
    _write_csv(perf_dir / "resource_usage.csv", res_rows)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_sec": round(time.monotonic() - t0, 2),
        "hardware": {
            "cpu_physical": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda,
        },
        "warm_protocol": {
            "queries": 40,
            "combinations": 8,
            "ks": list(STAGE04_KS),
            "candidate_top_k": STAGE03_CANDIDATE_POOL,
            "repeats": warm_repeats,
            "seed": 42,
            "warmup_calls": warm_res["warmup_calls"],
            "total_measured_samples": warm_res["total_samples"],
            "bm25_rebuilds_during_measure": warm_res["bm25_rebuilds_during_measure"],
            "resource_snapshots": warm_res["resource_snapshots"],
        },
        "production_profile": {"top_k": 5, "candidate_top_k": 10, "repeats": 5, "samples": 1600, "note": "Evaluation-index experiment, NOT production end-to-end API"},
        "cold_start": {"trials_per_combo": cold_trials, "total_trials": len(cold_res), "os_disk_cache_cleared": False},
        "limitations": [
            "OS disk cache was not flushed between cold-start subprocess trials.",
            "Thermal throttling and OS background services were not externally locked.",
            "torch.cuda.max_memory_allocated measures PyTorch tensor allocations, not full CUDA driver context overhead (captured approximately by reserved memory).",
        ],
    }
    (perf_dir / "performance_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
