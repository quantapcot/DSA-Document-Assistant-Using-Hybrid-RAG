"""Stage 04 – Step 4 (Retrieval), Step 5 (Evidence), and Step 12.2 (Candidate Pool Sensitivity) Runners."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval import embeddings, reranker, vector_store
from app.retrieval.hybrid_search import RetrievalEngine

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.benchmark_builder import (
    EVAL_WORKSPACE_ID,
    SCOPE_BY_PROFILE,
    build_eval_retrieval_config,
)
from evaluation.corpus.provenance import canonical_json, sha256_bytes, sha256_text
from evaluation.stage03.dataset_schema import FROZEN_CORPORA, STAGE03_CANDIDATE_POOL, STAGE03_POOLING_TOP_K
from evaluation.stage03.pooling import verify_frozen_corpora_and_index
from evaluation.stage04.evidence_metrics import (
    EVIDENCE_OVERLAP_POLICY,
    aggregate_evidence_metrics,
    evaluate_query_evidence,
)
from evaluation.stage04.metrics import (
    COMBO_ID_BY_PAIR,
    STAGE04_COMBOS,
    STAGE04_KS,
    STAGE04_METHODS,
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
    compute_ranking_drift,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def load_official_ground_truth(root: Path = ROOT) -> dict[tuple[str, str], dict[str, Any]]:
    gt_a = _load_jsonl(root / "evaluation/datasets/ground_truth_recursive.jsonl")
    gt_b = _load_jsonl(root / "evaluation/datasets/ground_truth_adaptive.jsonl")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in gt_a + gt_b:
        out[(row["query_id"], row["profile_id"])] = row
    return out


def run_retrieval_and_evidence_benchmark(
    *,
    root: Path = ROOT,
    stage04_dir: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute 320 official Stage 04 Top-60/Pool-120 retrieval calls, ranking drift, retrieval & evidence metrics."""
    import torch

    out_root = stage04_dir or (root / "evaluation/artifacts/stage04")
    ret_dir = out_root / "retrieval"
    ev_dir = out_root / "evidence"
    ret_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)

    verified = verify_frozen_corpora_and_index(root)
    corpus_by_id = verified["corpus_by_id"]
    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    evidences = _load_jsonl(root / "evaluation/datasets/source_evidence.jsonl")
    gt_by_qp = load_official_ground_truth(root)
    q_by_id = {q["query_id"]: q for q in questions}
    ev_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in evidences:
        ev_by_q[ev["query_id"]].append(ev)

    # Load Stage 03 final annotations for optional semantic KU distinction
    s3_final_path = root / "evaluation/artifacts/stage03/ai_annotations_final.jsonl"
    sem_ku_map: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    if s3_final_path.exists():
        for rec in _load_jsonl(s3_final_path):
            sem_ku_map[(rec["query_id"], rec["profile_id"])][rec["chunk_id"]] = rec.get(
                "supported_knowledge_unit_ids", []
            )

    rankings_path = ret_dir / "retrieval_rankings.jsonl"
    if resume and rankings_path.exists():
        raw_rankings = _load_jsonl(rankings_path)
        if len(raw_rankings) != 320:
            raw_rankings = []
    else:
        raw_rankings = []

    if not raw_rankings:
        cfg = build_eval_retrieval_config(candidate_top_k=STAGE03_CANDIDATE_POOL)
        cfg_dict = {k: v for k, v in asdict(cfg).items() if k not in {"qdrant_path", "qdrant_url"}}
        cfg_fp = sha256_text(canonical_json(cfg_dict))[:16]

        assert torch.cuda.is_available(), "CUDA GPU required for Stage 04 official retrieval"
        emb_m = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
        rer_m = reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
        actual_emb_dev = str(emb_m.device)
        actual_rer_dev = str(next(rer_m.model.parameters()).device)
        if not actual_emb_dev.startswith("cuda") or not actual_rer_dev.startswith("cuda"):
            raise RuntimeError(f"Unintended CPU fallback: emb={actual_emb_dev}, rer={actual_rer_dev}")

        engine = RetrievalEngine(config=cfg)
        try:
            for combo_id, pid, method in STAGE04_COMBOS:
                corpus_id = SCOPE_BY_PROFILE[pid]
                other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
                corpus_sha = FROZEN_CORPORA[pid]["corpus_sha256"]
                for q in questions:
                    qid = q["query_id"]
                    hits = engine.retrieve(
                        q["query"],
                        method,
                        top_k=STAGE03_POOLING_TOP_K,
                        corpus_id=corpus_id,
                        workspace_id=EVAL_WORKSPACE_ID,
                    )
                    seen_ids: set[str] = set()
                    formatted: list[dict[str, Any]] = []
                    for expected_rank, h in enumerate(hits, start=1):
                        cid = h["chunk_id"]
                        if cid in seen_ids:
                            raise RuntimeError(f"Duplicate chunk {cid} in {combo_id}/{qid}")
                        seen_ids.add(cid)
                        if cid not in corpus_by_id[pid] or cid in corpus_by_id[other_pid]:
                            raise RuntimeError(f"Scope leak/invalid chunk {cid} in {combo_id}/{qid}")
                        score = float(h["score"])
                        if not math.isfinite(score):
                            raise RuntimeError(f"Non-finite score in {combo_id}/{qid}")
                        actual_method = h.get("retrieval_method", method)
                        if actual_method != method:
                            raise RuntimeError(f"Silent fallback: {method} -> {actual_method}")
                        formatted.append({"rank": expected_rank, "chunk_id": cid, "score": round(score, 6)})

                    raw_rankings.append({
                        "combo_id": combo_id,
                        "query_id": qid,
                        "profile_id": pid,
                        "corpus_id": corpus_id,
                        "corpus_sha256": corpus_sha,
                        "method": method,
                        "requested_method": method,
                        "actual_method": method,
                        "requested_top_k": STAGE03_POOLING_TOP_K,
                        "candidate_top_k": STAGE03_CANDIDATE_POOL,
                        "actual_returned_count": len(formatted),
                        "retrieval_config_fingerprint": cfg_fp,
                        "embedding_device": actual_emb_dev,
                        "reranker_device": actual_rer_dev,
                        "error": None,
                        "fallback_occurred": False,
                        "results": formatted,
                    })
        finally:
            engine.client.close()
            vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

        _write_jsonl(rankings_path, raw_rankings)

    # Ranking drift vs Stage 03 pooling rankings
    s3_rankings = _load_jsonl(root / "evaluation/artifacts/stage03/raw_retrieval_results.jsonl")
    drift = compute_ranking_drift(s3_rankings, raw_rankings, gt_by_qp)
    (ret_dir / "ranking_drift.json").write_text(json.dumps(drift, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compute Primary Retrieval Metrics, Strict Grade-2 Sensitivity, Judgment Coverage, and Evidence Metrics
    per_query_primary: list[dict[str, Any]] = []
    per_query_strict: list[dict[str, Any]] = []
    judgment_cov_rows: list[dict[str, Any]] = []
    ev_per_query: list[dict[str, Any]] = []
    ku_trace_records: list[dict[str, Any]] = []
    ev_redundancy_rows: list[dict[str, Any]] = []

    for rec in raw_rankings:
        qid = rec["query_id"]
        pid = rec["profile_id"]
        method = rec["method"]
        combo_id = rec.get("combo_id") or COMBO_ID_BY_PAIR[(pid, method)]
        q = q_by_id[qid]
        gt = gt_by_qp[(qid, pid)]
        retrieved_ids = [h["chunk_id"] for h in rec["results"]]

        prim_rows = compute_query_retrieval_metrics(
            retrieved_ids, gt["relevance_grades"], ks=STAGE04_KS, min_positive_grade=1
        )
        strict_rows = compute_query_retrieval_metrics(
            retrieved_ids, gt["relevance_grades"], ks=STAGE04_KS, min_positive_grade=2
        )
        base_meta = {
            "combo_id": combo_id,
            "query_id": qid,
            "profile_id": pid,
            "method": method,
            "topic": q["topic"],
            "language": q["language"],
            "difficulty": q["difficulty"],
            "question_type": q["question_type"],
            "evidence_count": len(ev_by_q[qid]),
            "ku_count": len(q["required_knowledge_units"]),
        }
        for pr in prim_rows:
            per_query_primary.append({**base_meta, **pr})
            judgment_cov_rows.append({
                "combo_id": combo_id,
                "query_id": qid,
                "profile_id": pid,
                "method": method,
                "k": pr["k"],
                "actual_returned_count": pr["actual_returned_count"],
                "judged_count": pr["judged_count"],
                "unjudged_count": pr["unjudged_count"],
                "judged_at_k": pr["judged_at_k"],
            })
        for sr in strict_rows:
            per_query_strict.append({**base_meta, **sr})

        retrieved_chunks = [corpus_by_id[pid][cid] for cid in retrieved_ids]
        ev_rows, ku_trace = evaluate_query_evidence(
            question=q,
            evidences=ev_by_q[qid],
            retrieved_chunks=retrieved_chunks,
            ks=STAGE04_KS,
            semantic_ku_by_chunk=sem_ku_map.get((qid, pid)),
        )
        for er in ev_rows:
            ev_per_query.append({**base_meta, **er})
            ev_redundancy_rows.append({
                "combo_id": combo_id,
                "query_id": qid,
                "profile_id": pid,
                "method": method,
                "k": er["k"],
                "matched_evidence_chunks": er["matched_evidence_chunks"],
                "sum_matched_evidence_chars": er["sum_matched_evidence_chars"],
                "unique_matched_evidence_chars": er["unique_matched_evidence_chars"],
                "evidence_redundancy_at_k": er["evidence_redundancy_at_k"],
            })
        ku_trace_records.append({
            "combo_id": combo_id,
            "query_id": qid,
            "profile_id": pid,
            "method": method,
            "overlap_policy": EVIDENCE_OVERLAP_POLICY,
            "ku_coverage_by_k": ku_trace,
        })

    summary_primary = aggregate_retrieval_metrics(per_query_primary)
    summary_strict = aggregate_retrieval_metrics(per_query_strict)
    summary_ev = aggregate_evidence_metrics(ev_per_query)

    _write_csv(ret_dir / "retrieval_metrics_per_query.csv", per_query_primary)
    _write_csv(ret_dir / "retrieval_metrics_summary.csv", summary_primary)
    _write_csv(ret_dir / "strict_relevance_metrics.csv", summary_strict)
    _write_csv(ret_dir / "judgment_coverage.csv", judgment_cov_rows)

    _write_csv(ev_dir / "evidence_metrics_per_query.csv", ev_per_query)
    _write_csv(ev_dir / "evidence_metrics_summary.csv", summary_ev)
    _write_jsonl(ev_dir / "knowledge_unit_coverage.jsonl", ku_trace_records)
    _write_csv(ev_dir / "evidence_redundancy.csv", ev_redundancy_rows)

    return {
        "rankings_count": len(raw_rankings),
        "ranking_drift": {
            "exact_ranking_matches": drift["exact_ranking_matches"],
            "total_calls_compared": drift["total_calls_compared"],
            "exact_ranking_agreement_pct": drift["exact_ranking_agreement_pct"],
            "unique_unjudged_candidates": drift["unique_unjudged_candidates"],
        },
        "retrieval_summary_k5": [r for r in summary_primary if r["k"] == 5],
        "retrieval_summary_k15": [r for r in summary_primary if r["k"] == 15],
        "evidence_summary_k5": [r for r in summary_ev if r["k"] == 5],
        "evidence_summary_k15": [r for r in summary_ev if r["k"] == 15],
    }


def run_candidate_pool_sensitivity(
    *,
    root: Path = ROOT,
    stage04_dir: Path | None = None,
    pools: tuple[int, ...] = (10, 30, 60, 120),
    top_k: int = 5,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Step 12.2: Controlled candidate_top_k sensitivity at top_k=5 for hybrid_rrf & hybrid_rerank on both profiles."""
    import time
    import torch

    out_root = stage04_dir or (root / "evaluation/artifacts/stage04")
    sens_dir = out_root / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    verified = verify_frozen_corpora_and_index(root)
    corpus_by_id = verified["corpus_by_id"]
    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    evidences = _load_jsonl(root / "evaluation/datasets/source_evidence.jsonl")
    gt_by_qp = load_official_ground_truth(root)
    ev_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in evidences:
        ev_by_q[ev["query_id"]].append(ev)

    raw_path = sens_dir / "candidate_pool_rankings.jsonl"
    if resume and raw_path.exists():
        raw_rows = _load_jsonl(raw_path)
        expected = len(questions) * 2 * 2 * len(pools)
        if len(raw_rows) != expected:
            raw_rows = []
    else:
        raw_rows = []

    if not raw_rows:
        assert torch.cuda.is_available(), "CUDA GPU required for candidate pool sensitivity"
        # Share one local client/models across pools cleanly
        base_cfg = build_eval_retrieval_config(candidate_top_k=120)
        embeddings._get_model(base_cfg.embedding_model, base_cfg.embedding_device)
        reranker._get_model(base_cfg.reranker_model, base_cfg.reranker_device, base_cfg.reranker_max_length)
        client = vector_store.get_client(config=base_cfg)
        try:
            for pool_k in pools:
                cfg = build_eval_retrieval_config(candidate_top_k=pool_k)
                engine = RetrievalEngine(client=client, config=cfg)
                # Warmup BM25 + GPU once per pool
                for pid in (PROFILE_A_ID, PROFILE_B_ID):
                    engine.retrieve(questions[0]["query"], "hybrid_rerank", top_k=top_k, corpus_id=SCOPE_BY_PROFILE[pid], workspace_id=EVAL_WORKSPACE_ID)
                torch.cuda.synchronize()

                for pid in (PROFILE_A_ID, PROFILE_B_ID):
                    corpus_id = SCOPE_BY_PROFILE[pid]
                    for method in ("hybrid_rrf", "hybrid_rerank"):
                        combo_id = COMBO_ID_BY_PAIR[(pid, method)]
                        for q in questions:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter_ns()
                            hits = engine.retrieve(
                                q["query"],
                                method,
                                top_k=top_k,
                                corpus_id=corpus_id,
                                workspace_id=EVAL_WORKSPACE_ID,
                            )
                            torch.cuda.synchronize()
                            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                            raw_rows.append({
                                "combo_id": combo_id,
                                "profile_id": pid,
                                "method": method,
                                "top_k": top_k,
                                "candidate_top_k": pool_k,
                                "query_id": q["query_id"],
                                "latency_ms": round(elapsed_ms, 4),
                                "chunk_ids": [h["chunk_id"] for h in hits],
                            })
        finally:
            client.close()
            vector_store._LOCAL_CLIENTS.pop(str(Path(base_cfg.qdrant_path).resolve()), None)

        _write_jsonl(raw_path, raw_rows)

    q_by_id = {q["query_id"]: q for q in questions}
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        qid, pid = row["query_id"], row["profile_id"]
        gt = gt_by_qp[(qid, pid)]
        m = compute_query_retrieval_metrics(row["chunk_ids"], gt["relevance_grades"], ks=(top_k,), min_positive_grade=1)[0]
        ev_m = evaluate_query_evidence(
            question=q_by_id[qid],
            evidences=ev_by_q[qid],
            retrieved_chunks=[corpus_by_id[pid][c] for c in row["chunk_ids"]],
            ks=(top_k,),
        )[0][0]
        grouped[(row["combo_id"], pid, row["method"], int(row["candidate_top_k"]))].append({
            **m,
            "evidence_recall_at_k": ev_m["evidence_recall_at_k"],
            "complete_evidence_rate_at_k": ev_m["complete_evidence_rate_at_k"],
            "latency_ms": row["latency_ms"],
        })

    from statistics import mean as _mean, median as _median

    summary_rows: list[dict[str, Any]] = []
    for (combo_id, pid, method, pool_k), items in sorted(grouped.items()):
        lats = sorted(x["latency_ms"] for x in items)
        summary_rows.append({
            "combo_id": combo_id,
            "profile_id": pid,
            "method": method,
            "top_k": top_k,
            "candidate_top_k": pool_k,
            "queries": len(items),
            "precision_at_5": round(_mean(x["precision_at_k"] for x in items), 6),
            "pooled_recall_at_5": round(_mean(x["pooled_recall_at_k"] for x in items), 6),
            "mrr_at_5": round(_mean(x["mrr_at_k"] for x in items), 6),
            "ndcg_at_5": round(_mean(x["ndcg_at_k"] for x in items), 6),
            "evidence_recall_at_5": round(_mean(x["evidence_recall_at_k"] for x in items), 6),
            "complete_evidence_rate_at_5": round(_mean(x["complete_evidence_rate_at_k"] for x in items), 6),
            "judged_at_5": round(_mean(x["judged_at_k"] for x in items), 6),
            "total_unjudged_slots": sum(x["unjudged_count"] for x in items),
            "mean_latency_ms": round(_mean(lats), 3),
            "p50_latency_ms": round(_median(lats), 3),
            "p95_latency_ms": round(lats[min(len(lats) - 1, int(round((len(lats) - 1) * 0.95)))], 3),
        })

    _write_csv(sens_dir / "candidate_pool_sensitivity.csv", summary_rows)
    return summary_rows
