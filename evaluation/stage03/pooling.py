"""Stage 03 – 320 Official Pooling Calls, Deduplication, Evidence Augmentation, and Traceability."""
from __future__ import annotations

import hashlib
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
from app.retrieval.vector_store import check_model_exhaustive, scope_filter

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.benchmark_builder import (
    EVAL_COLLECTION,
    EVAL_WORKSPACE_ID,
    SCOPE_BY_PROFILE,
    build_eval_retrieval_config,
)
from evaluation.corpus.provenance import canonical_json, sha256_bytes, sha256_text
from evaluation.stage03.dataset_schema import (
    CANONICAL_FINGERPRINT_SHA256,
    FROZEN_CORPORA,
    STAGE03_CANDIDATE_POOL,
    STAGE03_METHODS,
    STAGE03_POOLING_TOP_K,
    chunk_matches_evidence,
    extract_chunk_source_backed_intervals,
)


def blind_sort_key(query_id: str, profile_id: str, chunk_id: str) -> str:
    """Deterministic randomized order per (query_id, profile_id), independent of method/rank/score."""
    return hashlib.sha256(f"stage03_blind|{query_id}|{profile_id}|{chunk_id}".encode("utf-8")).hexdigest()


def verify_frozen_corpora_and_index(root: Path = ROOT) -> dict[str, Any]:
    """Verify frozen Corpus A & B JSONLs, manifests, and isolated Qdrant index without modifying them."""
    corpus_rows: dict[str, list[dict[str, Any]]] = {}
    corpus_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    verified_info: dict[str, Any] = {}

    for pid, spec in FROZEN_CORPORA.items():
        cpath = root / spec["rel_chunks_path"]
        mpath = root / spec["rel_manifest_path"]
        actual_sha = sha256_bytes(cpath.read_bytes())
        if actual_sha != spec["corpus_sha256"]:
            raise RuntimeError(f"Frozen corpus SHA-256 mismatch for {pid}: {actual_sha} != {spec['corpus_sha256']}")
        rows = [json.loads(ln) for ln in cpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(rows) != spec["chunk_count"]:
            raise RuntimeError(f"Frozen corpus count mismatch for {pid}: {len(rows)} != {spec['chunk_count']}")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest["profile"]["corpus_sha256"] != actual_sha or manifest["profile"]["chunk_count"] != len(rows):
            raise RuntimeError(f"Frozen manifest mismatch for {pid}")
        if manifest["preprocessor"]["canonical_fingerprint_sha256"] != CANONICAL_FINGERPRINT_SHA256:
            raise RuntimeError(f"Canonical fingerprint mismatch in manifest for {pid}")

        id_map = {r["chunk_id"]: r for r in rows}
        if len(id_map) != len(rows):
            raise RuntimeError(f"Duplicate chunk_id in frozen corpus {pid}")
        corpus_rows[pid] = rows
        corpus_by_id[pid] = id_map
        verified_info[pid] = {
            "corpus_id": spec["corpus_id"],
            "chunk_count": len(rows),
            "corpus_sha256": actual_sha,
        }

    if set(corpus_by_id[PROFILE_A_ID]) & set(corpus_by_id[PROFILE_B_ID]):
        raise RuntimeError("Cross-profile chunk_id overlap detected between frozen corpora")

    cfg = build_eval_retrieval_config(candidate_top_k=STAGE03_CANDIDATE_POOL)
    client = vector_store.get_client(config=cfg)
    try:
        check_model_exhaustive(client, cfg)
        total_pts = client.count(EVAL_COLLECTION, exact=True).count
        cnt_a = client.count(
            EVAL_COLLECTION,
            count_filter=scope_filter(SCOPE_BY_PROFILE[PROFILE_A_ID], EVAL_WORKSPACE_ID),
            exact=True,
        ).count
        cnt_b = client.count(
            EVAL_COLLECTION,
            count_filter=scope_filter(SCOPE_BY_PROFILE[PROFILE_B_ID], EVAL_WORKSPACE_ID),
            exact=True,
        ).count
        if cnt_a != FROZEN_CORPORA[PROFILE_A_ID]["chunk_count"] or cnt_b != FROZEN_CORPORA[PROFILE_B_ID]["chunk_count"]:
            raise RuntimeError(f"Qdrant scope count mismatch: A={cnt_a}, B={cnt_b}")
        if total_pts != cnt_a + cnt_b:
            raise RuntimeError(f"Qdrant total points mismatch: {total_pts} != {cnt_a + cnt_b}")
    finally:
        client.close()
        vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    return {
        "canonical_fingerprint_sha256": CANONICAL_FINGERPRINT_SHA256,
        "profiles": verified_info,
        "qdrant_total_points": total_pts,
        "corpus_rows": corpus_rows,
        "corpus_by_id": corpus_by_id,
    }


def build_deduplicated_pools(
    *,
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    corpus_rows_by_profile: dict[str, list[dict[str, Any]]],
    corpus_by_id: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Deduplicate raw pooling results strictly by (query_id, profile_id, chunk_id), apply
    uniform source-backed evidence augmentation, preserve separate method traceability, and
    sort candidates in deterministic blind-method order."""
    ev_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in evidences:
        ev_by_query[ev["query_id"]].append(ev)

    # Index chunks by (profile_id, document_id, page) for uniform evidence augmentation
    chunks_by_doc_page: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for pid, rows in corpus_rows_by_profile.items():
        for ch in rows:
            pages = {iv[0] for iv in extract_chunk_source_backed_intervals(ch)}
            for pg in pages:
                chunks_by_doc_page[(pid, ch["document_id"], pg)].append(ch)

    # Group raw results by (query_id, profile_id)
    raw_by_qp: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in raw_results:
        raw_by_qp[(rec["query_id"], rec["profile_id"])].append(rec)

    pools_by_profile: dict[str, list[dict[str, Any]]] = {PROFILE_A_ID: [], PROFILE_B_ID: []}
    method_mappings: list[dict[str, Any]] = []
    per_query_stats: list[dict[str, Any]] = []

    profile_totals: dict[str, dict[str, Any]] = {
        pid: {
            "raw_candidates": 0,
            "unique_retrieval_candidates": 0,
            "duplicates_removed": 0,
            "evidence_added_candidates": 0,
            "final_annotation_workload": 0,
            "actual_retrieval_calls": 0,
            "fewer_than_60_calls": [],
        }
        for pid in (PROFILE_A_ID, PROFILE_B_ID)
    }

    for q in questions:
        qid = q["query_id"]
        q_evs = ev_by_query.get(qid, [])
        q_stat: dict[str, Any] = {"query_id": qid, "profiles": {}}

        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
            prefix = "A" if pid == PROFILE_A_ID else "B"
            qp_raws = raw_by_qp.get((qid, pid), [])

            raw_count = 0
            # chunk_id -> list of method appearances
            appearances: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in qp_raws:
                profile_totals[pid]["actual_retrieval_calls"] += 1
                method = rec["method"]
                returned = rec["results"]
                if len(returned) < rec["requested_top_k"]:
                    profile_totals[pid]["fewer_than_60_calls"].append({
                        "query_id": qid,
                        "method": method,
                        "actual_returned_count": len(returned),
                    })
                seen_in_ranking: set[str] = set()
                for hit in returned:
                    cid = hit["chunk_id"]
                    if cid in seen_in_ranking:
                        raise ValueError(f"Duplicate chunk_id {cid} inside single ranking {qid}/{pid}/{method}")
                    seen_in_ranking.add(cid)
                    if cid not in corpus_by_id[pid]:
                        raise ValueError(f"Invalid chunk_id {cid} not in frozen corpus {pid}")
                    if cid in corpus_by_id[other_pid]:
                        raise ValueError(f"Cross-profile chunk_id leak {cid} in {pid}")
                    raw_count += 1
                    appearances[cid].append({
                        "method": method,
                        "rank": hit["rank"],
                        "score": hit["score"],
                    })

            unique_retrieval_ids = set(appearances.keys())
            duplicates_removed = raw_count - len(unique_retrieval_ids)

            # Uniform evidence-based pool augmentation across all source-backed segments
            evidence_added_ids: set[str] = set()
            matched_ev_ids_by_chunk: dict[str, list[str]] = defaultdict(list)
            checked_candidates: dict[str, dict[str, Any]] = {}
            for ev in q_evs:
                for ch in chunks_by_doc_page.get((pid, ev["document_id"], ev["canonical_page"]), []):
                    checked_candidates[ch["chunk_id"]] = ch

            for cid, ch in checked_candidates.items():
                matched_eids = [ev["evidence_id"] for ev in q_evs if chunk_matches_evidence(ch, ev)]
                if matched_eids:
                    matched_ev_ids_by_chunk[cid] = sorted(matched_eids)
                    if cid not in unique_retrieval_ids:
                        evidence_added_ids.add(cid)

            all_unique_ids = unique_retrieval_ids | evidence_added_ids
            blind_ordered_ids = sorted(all_unique_ids, key=lambda cid: blind_sort_key(qid, pid, cid))

            for idx, cid in enumerate(blind_ordered_ids, start=1):
                ann_id = f"{prefix}_{qid}_{idx:04d}"
                ch = corpus_by_id[pid][cid]
                origin = "retrieval_pool" if cid in unique_retrieval_ids else "evidence_augmentation"
                meta = ch.get("metadata") or {}
                spans = meta.get("source_spans") or []
                span_desc = (
                    f"p.{ch['page']} [{spans[0]['char_start']}:{spans[-1]['char_end']}]"
                    if spans
                    else f"p.{ch['page']}"
                )
                section = ch.get("section")
                loc_str = f"{span_desc} | {section}" if section else span_desc

                pools_by_profile[pid].append({
                    "annotation_id": ann_id,
                    "query_id": qid,
                    "profile_id": pid,
                    "chunk_id": cid,
                    "chunk_text": ch["text"],
                    "document_id": ch["document_id"],
                    "filename": ch["filename"],
                    "source_path": meta.get("source_path", ch["filename"]),
                    "page": ch["page"],
                    "source_location": loc_str,
                    "content_type": meta.get("content_type", "unknown"),
                    "candidate_origin": origin,
                    "matched_evidence_ids": matched_ev_ids_by_chunk.get(cid, []),
                    "relevance_grade": None,  # ALWAYS blank/unjudged before human annotation
                })

                method_mappings.append({
                    "annotation_id": ann_id,
                    "query_id": qid,
                    "profile_id": pid,
                    "chunk_id": cid,
                    "blind_order_index": idx,
                    "candidate_origin": origin,
                    "methods": sorted({a["method"] for a in appearances.get(cid, [])}),
                    "method_ranks": {a["method"]: a["rank"] for a in appearances.get(cid, [])},
                    "method_scores": {a["method"]: a["score"] for a in appearances.get(cid, [])},
                    "matched_evidence_ids": matched_ev_ids_by_chunk.get(cid, []),
                })

            final_n = len(blind_ordered_ids)
            dedup_rate = round(100.0 * duplicates_removed / max(1, raw_count), 2)
            profile_totals[pid]["raw_candidates"] += raw_count
            profile_totals[pid]["unique_retrieval_candidates"] += len(unique_retrieval_ids)
            profile_totals[pid]["duplicates_removed"] += duplicates_removed
            profile_totals[pid]["evidence_added_candidates"] += len(evidence_added_ids)
            profile_totals[pid]["final_annotation_workload"] += final_n

            q_stat["profiles"][pid] = {
                "raw_candidates": raw_count,
                "unique_retrieval_candidates": len(unique_retrieval_ids),
                "duplicates_removed": duplicates_removed,
                "deduplication_rate_pct": dedup_rate,
                "evidence_added_candidates": len(evidence_added_ids),
                "final_annotation_workload": final_n,
            }

        per_query_stats.append(q_stat)

    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        pt = profile_totals[pid]
        pt["deduplication_rate_pct"] = round(100.0 * pt["duplicates_removed"] / max(1, pt["raw_candidates"]), 2)
        pt["fewer_than_60_count"] = len(pt["fewer_than_60_calls"])

    combined_totals = {
        "raw_candidates": sum(profile_totals[p]["raw_candidates"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
        "unique_retrieval_candidates": sum(profile_totals[p]["unique_retrieval_candidates"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
        "duplicates_removed": sum(profile_totals[p]["duplicates_removed"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
        "evidence_added_candidates": sum(profile_totals[p]["evidence_added_candidates"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
        "final_annotation_workload": sum(profile_totals[p]["final_annotation_workload"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
        "actual_retrieval_calls": sum(profile_totals[p]["actual_retrieval_calls"] for p in (PROFILE_A_ID, PROFILE_B_ID)),
    }
    combined_totals["deduplication_rate_pct"] = round(
        100.0 * combined_totals["duplicates_removed"] / max(1, combined_totals["raw_candidates"]), 2
    )

    return {
        "pool_recursive": pools_by_profile[PROFILE_A_ID],
        "pool_adaptive": pools_by_profile[PROFILE_B_ID],
        "method_mapping": method_mappings,
        "diagnostics": {
            "requested_top_k": STAGE03_POOLING_TOP_K,
            "candidate_pool": STAGE03_CANDIDATE_POOL,
            "methods": list(STAGE03_METHODS),
            "totals_by_profile": profile_totals,
            "totals_combined": combined_totals,
            "per_query": per_query_stats,
        },
    }


def run_official_pooling(
    *,
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    artifacts_dir: Path | None = None,
    root: Path = ROOT,
    engine: RetrievalEngine | None = None,
) -> dict[str, Any]:
    """Execute 320 official pooling calls (40 queries x 2 profiles x 4 methods, top_k=60, pool=120)."""
    import torch

    verified = verify_frozen_corpora_and_index(root)
    corpus_rows = verified["corpus_rows"]
    corpus_by_id = verified["corpus_by_id"]

    cfg = build_eval_retrieval_config(candidate_top_k=STAGE03_CANDIDATE_POOL)
    cfg_dict = {k: v for k, v in asdict(cfg).items() if k not in {"qdrant_path", "qdrant_url"}}
    cfg_fp = sha256_text(canonical_json(cfg_dict))[:16]

    owns_engine = engine is None
    if owns_engine:
        assert torch.cuda.is_available(), "CUDA GPU required for Stage 03 official pooling"
        emb_m = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
        rer_m = reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
        actual_emb_dev = str(emb_m.device)
        actual_rer_dev = str(next(rer_m.model.parameters()).device)
        if not actual_emb_dev.startswith("cuda") or not actual_rer_dev.startswith("cuda"):
            raise RuntimeError(f"Unintended CPU fallback: emb={actual_emb_dev}, rer={actual_rer_dev}")
        active_engine = RetrievalEngine(config=cfg)
    else:
        actual_emb_dev = cfg.embedding_device
        actual_rer_dev = cfg.reranker_device
        active_engine = engine

    raw_results: list[dict[str, Any]] = []
    try:
        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            corpus_id = SCOPE_BY_PROFILE[pid]
            other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
            corpus_sha = FROZEN_CORPORA[pid]["corpus_sha256"]

            for method in STAGE03_METHODS:
                for q in questions:
                    qid = q["query_id"]
                    hits = active_engine.retrieve(
                        q["query"],
                        method,
                        top_k=STAGE03_POOLING_TOP_K,
                        corpus_id=corpus_id,
                        workspace_id=EVAL_WORKSPACE_ID,
                    )
                    seen_ids: set[str] = set()
                    formatted_hits: list[dict[str, Any]] = []
                    for expected_rank, h in enumerate(hits, start=1):
                        cid = h["chunk_id"]
                        if cid in seen_ids:
                            raise RuntimeError(f"Duplicate chunk_id {cid} returned in {qid}/{pid}/{method}")
                        seen_ids.add(cid)
                        if cid not in corpus_by_id[pid] or cid in corpus_by_id[other_pid]:
                            raise RuntimeError(f"Scope contamination or invalid chunk {cid} in {qid}/{pid}/{method}")
                        if h.get("rank") != expected_rank:
                            raise RuntimeError(f"Invalid rank sequence in {qid}/{pid}/{method}")
                        score = float(h["score"])
                        if not math.isfinite(score):
                            raise RuntimeError(f"Non-finite score in {qid}/{pid}/{method}: {score}")
                        actual_method = h.get("retrieval_method", method)
                        if actual_method != method:
                            raise RuntimeError(
                                f"Silent method fallback detected: requested={method}, actual={actual_method}"
                            )
                        formatted_hits.append({
                            "rank": expected_rank,
                            "chunk_id": cid,
                            "score": round(score, 6),
                        })

                    raw_results.append({
                        "query_id": qid,
                        "profile_id": pid,
                        "corpus_id": corpus_id,
                        "corpus_sha256": corpus_sha,
                        "requested_method": method,
                        "actual_method": method,
                        "method": method,
                        "requested_top_k": STAGE03_POOLING_TOP_K,
                        "candidate_top_k": STAGE03_CANDIDATE_POOL,
                        "actual_returned_count": len(formatted_hits),
                        "retrieval_config_fingerprint": cfg_fp,
                        "error": None,
                        "fallback_occurred": False,
                        "results": formatted_hits,
                    })
    finally:
        if owns_engine:
            active_engine.client.close()
            vector_store._LOCAL_CLIENTS.pop(str(Path(cfg.qdrant_path).resolve()), None)

    pooled = build_deduplicated_pools(
        questions=questions,
        evidences=evidences,
        raw_results=raw_results,
        corpus_rows_by_profile=corpus_rows,
        corpus_by_id=corpus_by_id,
    )
    pooled["diagnostics"]["execution_environment"] = {
        "embedding_device": actual_emb_dev,
        "reranker_device": actual_rer_dev,
        "reranker_batch_size": cfg.reranker_batch_size,
        "reranker_max_length": cfg.reranker_max_length,
        "retrieval_config_fingerprint": cfg_fp,
        "corpus_sha256": {pid: FROZEN_CORPORA[pid]["corpus_sha256"] for pid in (PROFILE_A_ID, PROFILE_B_ID)},
    }

    out_dir = artifacts_dir or (root / "evaluation" / "artifacts" / "stage03")
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
        data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
        path.write_bytes(data)
        return sha256_bytes(data)

    raw_sha = write_jsonl(out_dir / "raw_retrieval_results.jsonl", raw_results)
    pool_a_sha = write_jsonl(out_dir / "candidate_pool_recursive.jsonl", pooled["pool_recursive"])
    pool_b_sha = write_jsonl(out_dir / "candidate_pool_adaptive.jsonl", pooled["pool_adaptive"])
    map_sha = write_jsonl(out_dir / "method_candidate_mapping.jsonl", pooled["method_mapping"])

    pooled["diagnostics"]["artifact_sha256"] = {
        "raw_retrieval_results.jsonl": raw_sha,
        "candidate_pool_recursive.jsonl": pool_a_sha,
        "candidate_pool_adaptive.jsonl": pool_b_sha,
        "method_candidate_mapping.jsonl": map_sha,
    }
    diag_path = out_dir / "pool_diagnostics.json"
    diag_path.write_text(json.dumps(pooled["diagnostics"], ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "raw_results": raw_results,
        "pool_recursive": pooled["pool_recursive"],
        "pool_adaptive": pooled["pool_adaptive"],
        "method_mapping": pooled["method_mapping"],
        "diagnostics": pooled["diagnostics"],
        "artifacts_dir": str(out_dir),
    }
