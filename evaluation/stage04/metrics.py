"""Stage 04 – Profile-Aware Top-60 Retrieval Metrics, Strict Grade-2 Sensitivity, and Ranking Drift."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.metrics.retrieval_metrics import (
    average_precision_at_k,
    f1_at_k,
    hit_rate_at_k,
    judged_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

STAGE04_KS: tuple[int, ...] = (1, 3, 5, 10, 15, 20, 30, 45, 60)
STAGE04_METHODS: tuple[str, ...] = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
STAGE04_COMBOS: tuple[tuple[str, str, str], ...] = (
    ("A1", PROFILE_A_ID, "bm25"),
    ("A2", PROFILE_A_ID, "dense"),
    ("A3", PROFILE_A_ID, "hybrid_rrf"),
    ("A4", PROFILE_A_ID, "hybrid_rerank"),
    ("B1", PROFILE_B_ID, "bm25"),
    ("B2", PROFILE_B_ID, "dense"),
    ("B3", PROFILE_B_ID, "hybrid_rrf"),
    ("B4", PROFILE_B_ID, "hybrid_rerank"),
)
COMBO_ID_BY_PAIR: dict[tuple[str, str], str] = {(pid, m): cid for cid, pid, m in STAGE04_COMBOS}

PRIMARY_METRIC_NAMES: tuple[str, ...] = (
    "precision_at_k",
    "pooled_recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "map_at_k",
    "f1_at_k",
    "judged_at_k",
)


def compute_query_retrieval_metrics(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    *,
    ks: tuple[int, ...] = STAGE04_KS,
    min_positive_grade: int = 1,
    labels_complete: bool = True,
) -> list[dict[str, Any]]:
    """Compute all 8 Stage 04 retrieval metrics for a single (query, profile, method) ranking across *ks*.

    - Primary relevance uses ``min_positive_grade=1`` (Grades 1 & 2 = relevant).
    - Strict sensitivity analysis uses ``min_positive_grade=2`` (Grade 2 only = relevant).
    - Zero-positive queries return ``None`` for recall/AP/F1 (never raising or faking labels).
    - Unjudged retrieved IDs are NOT treated as grade 0 in ``judged_at_k``; they are reported
      explicitly via ``judged_at_k``, ``judged_count``, and ``unjudged_count``.
    """
    relevant_ids = {cid for cid, g in relevance_grades.items() if g >= min_positive_grade}
    judged_ids = set(relevance_grades.keys())
    # For strict Grade-2-only sensitivity, graded nDCG still reflects binary Grade-2 gain or standard graded
    eff_grades = (
        relevance_grades
        if min_positive_grade == 1
        else {cid: (2 if g >= 2 else 0) for cid, g in relevance_grades.items()}
    )

    rows: list[dict[str, Any]] = []
    for k in ks:
        prec = precision_at_k(retrieved_ids, relevant_ids, k)
        rec = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
            labels_complete=labels_complete,
            zero_positive_policy="none",
        )
        rr = reciprocal_rank(retrieved_ids, relevant_ids, k=k)
        ndcg = ndcg_at_k(retrieved_ids, eff_grades, k, labels_complete=labels_complete)
        hr = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        ap = average_precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
            labels_complete=labels_complete,
            zero_positive_policy="none",
        )
        f1 = f1_at_k(prec, rec)
        j_rate, j_cnt, ret_cnt = judged_at_k(retrieved_ids, judged_ids, k)
        rows.append({
            "k": k,
            "precision_at_k": prec,
            "pooled_recall_at_k": rec,
            "mrr_at_k": rr,
            "ndcg_at_k": ndcg,
            "hit_rate_at_k": hr,
            "map_at_k": ap,
            "f1_at_k": f1,
            "judged_at_k": j_rate,
            "judged_count": j_cnt,
            "unjudged_count": ret_cnt - j_cnt,
            "actual_returned_count": ret_cnt,
            "positive_pool_count": len(relevant_ids),
            "zero_positive": len(relevant_ids) == 0,
        })
    return rows


def aggregate_retrieval_metrics(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average per-query retrieval metrics by (combo_id, profile_id, method, k) with explicit valid counts."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        key = (row["combo_id"], row["profile_id"], row["method"], int(row["k"]))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    combo_order = {cid: idx for idx, (cid, _, _) in enumerate(STAGE04_COMBOS)}
    for (combo_id, pid, method, k), rows in sorted(
        grouped.items(), key=lambda x: (combo_order.get(x[0][0], 99), x[0][3])
    ):
        rec: dict[str, Any] = {
            "combo_id": combo_id,
            "profile_id": pid,
            "method": method,
            "k": k,
            "total_queries": len(rows),
            "zero_positive_queries": sum(1 for r in rows if r["zero_positive"]),
            "mean_returned_count": round(mean(r["actual_returned_count"] for r in rows), 4),
            "total_unjudged_slots": sum(r["unjudged_count"] for r in rows),
        }
        for mname in PRIMARY_METRIC_NAMES:
            vals = [r[mname] for r in rows if r[mname] is not None]
            rec[mname] = round(mean(vals), 6) if vals else None
            rec[f"{mname}_valid_n"] = len(vals)
        summary.append(rec)
    return summary


def compute_ranking_drift(
    stage03_rankings: list[dict[str, Any]],
    stage04_rankings: list[dict[str, Any]],
    gt_by_qp: dict[tuple[str, str], dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 5, 10, 15, 30, 60),
) -> dict[str, Any]:
    """Compare Stage 04 fresh rankings against Stage 03 pooling rankings across all 320 calls."""
    s3_map = {(r["query_id"], r["profile_id"], r["method"]): r for r in stage03_rankings}
    s4_map = {(r["query_id"], r["profile_id"], r["method"]): r for r in stage04_rankings}

    if set(s3_map.keys()) != set(s4_map.keys()):
        raise ValueError("Stage 03 and Stage 04 ranking key sets do not match")

    exact_matches = 0
    total_calls = len(s4_map)
    newly_retrieved_unjudged: list[dict[str, Any]] = []
    drift_by_combo: dict[str, dict[str, Any]] = {}
    per_call_diffs: list[dict[str, Any]] = []

    for cid, pid, method in STAGE04_COMBOS:
        combo_keys = sorted(k for k in s4_map if k[1] == pid and k[2] == method)
        c_exact = 0
        c_overlaps: dict[int, list[float]] = {k: [] for k in ks}
        c_rank_changes = 0
        c_new_ids = 0
        c_unjudged = 0
        c_tie_changes = 0

        for key in combo_keys:
            qid = key[0]
            r3 = s3_map[key]["results"]
            r4 = s4_map[key]["results"]
            ids3 = [h["chunk_id"] for h in r3]
            ids4 = [h["chunk_id"] for h in r4]
            scores3 = {h["chunk_id"]: h["score"] for h in r3}
            scores4 = {h["chunk_id"]: h["score"] for h in r4}
            judged_ids = set(gt_by_qp[(qid, pid)]["relevance_grades"].keys())

            is_exact = ids3 == ids4
            if is_exact:
                exact_matches += 1
                c_exact += 1

            for k in ks:
                s3_k = set(ids3[:k])
                s4_k = set(ids4[:k])
                overlap = len(s3_k & s4_k) / max(1, min(k, max(len(s3_k), len(s4_k))))
                c_overlaps[k].append(overlap)

            rank3_map = {ch: idx for idx, ch in enumerate(ids3, start=1)}
            changed_ranks = 0
            tie_related = 0
            for idx, ch in enumerate(ids4, start=1):
                if ch not in rank3_map:
                    c_new_ids += 1
                    if ch not in judged_ids:
                        c_unjudged += 1
                        newly_retrieved_unjudged.append({
                            "combo_id": cid, "query_id": qid, "profile_id": pid,
                            "method": method, "rank": idx, "chunk_id": ch, "score": scores4[ch],
                        })
                elif rank3_map[ch] != idx:
                    changed_ranks += 1
                    # Check if swapped neighbor had virtually tied score (<= 1e-5)
                    if abs(scores3.get(ch, 0.0) - scores4.get(ch, 0.0)) <= 1e-5:
                        tie_related += 1
            c_rank_changes += changed_ranks
            c_tie_changes += tie_related

            if not is_exact:
                per_call_diffs.append({
                    "combo_id": cid,
                    "query_id": qid,
                    "profile_id": pid,
                    "method": method,
                    "changed_positions": changed_ranks,
                    "new_chunk_ids": sorted(set(ids4) - set(ids3)),
                    "dropped_chunk_ids": sorted(set(ids3) - set(ids4)),
                    "unjudged_in_top60": sorted(set(ids4) - judged_ids),
                    "top5_overlap": round(c_overlaps[5][-1], 4),
                    "top60_overlap": round(c_overlaps[60][-1], 4),
                })

        drift_by_combo[cid] = {
            "combo_id": cid,
            "profile_id": pid,
            "method": method,
            "queries": len(combo_keys),
            "exact_ranking_matches": c_exact,
            "exact_ranking_agreement_pct": round(100.0 * c_exact / max(1, len(combo_keys)), 2),
            "mean_topk_overlap": {str(k): (round(mean(v), 6) if v else None) for k, v in c_overlaps.items()},
            "total_rank_changes": c_rank_changes,
            "tie_related_rank_changes": c_tie_changes,
            "newly_retrieved_chunk_ids_count": c_new_ids,
            "missing_judgments_count": c_unjudged,
        }

    return {
        "total_calls_compared": total_calls,
        "exact_ranking_matches": exact_matches,
        "exact_ranking_agreement_pct": round(100.0 * exact_matches / max(1, total_calls), 2),
        "total_unjudged_retrieved_slots": len(newly_retrieved_unjudged),
        "unique_unjudged_candidates": len({(x["query_id"], x["profile_id"], x["chunk_id"]) for x in newly_retrieved_unjudged}),
        "by_combo": drift_by_combo,
        "non_exact_calls_count": len(per_call_diffs),
        "non_exact_calls_sample": per_call_diffs[:40],
        "unjudged_candidates": newly_retrieved_unjudged,
    }
