"""Stage 04 – Source-Backed Evidence Evaluation across Shared Canonical Knowledge Units (91 KUs / 107 Spans)."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from evaluation.stage03.dataset_schema import (
    chunk_matches_evidence,
    extract_chunk_source_backed_intervals,
)
from evaluation.stage04.metrics import STAGE04_COMBOS, STAGE04_KS

EVIDENCE_OVERLAP_POLICY = "overlap_chars >= min(20, max(8, floor(0.25 * evidence_span_len)))"


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted disjoint union of half-open [start, end) intervals."""
    valid = sorted((s, e) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for s, e in valid[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def interval_total_len(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def intersect_intervals(
    a_intervals: list[tuple[int, int]],
    b_intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return disjoint intersection intervals between two interval lists."""
    a_m = merge_intervals(a_intervals)
    b_m = merge_intervals(b_intervals)
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a_m) and j < len(b_m):
        s = max(a_m[i][0], b_m[j][0])
        e = min(a_m[i][1], b_m[j][1])
        if e > s:
            out.append((s, e))
        if a_m[i][1] < b_m[j][1]:
            i += 1
        else:
            j += 1
    return out


def _evidence_intervals_by_page(ev: dict[str, Any]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    by_dp: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    doc_id = ev["document_id"]
    for sp in ev.get("source_spans") or []:
        by_dp[(doc_id, int(sp["page"]))].append((int(sp["char_start"]), int(sp["char_end"])))
    return {k: merge_intervals(v) for k, v in by_dp.items()}


def _chunk_intervals_by_page(chunk: dict[str, Any]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    by_dp: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    doc_id = chunk["document_id"]
    for iv in extract_chunk_source_backed_intervals(chunk):
        pg, s, e = iv[0], iv[1], iv[2]
        by_dp[(doc_id, int(pg))].append((int(s), int(e)))
    return {k: merge_intervals(v) for k, v in by_dp.items()}


def _ev_coverage_threshold(ev_len: int) -> int:
    return min(20, max(8, math.floor(0.25 * ev_len)))


def evaluate_query_evidence(
    *,
    question: dict[str, Any],
    evidences: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]],
    ks: tuple[int, ...] = STAGE04_KS,
    semantic_ku_by_chunk: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute source-backed Evidence metrics and KU coverage across *ks* for one (query, profile, method).

    Supports:
    - ``single``: KU covered when its evidence span meets the source-backed overlap threshold over the Top-K union.
    - ``alternative``: KU covered when ANY registered alternative evidence span meets the threshold.
    - ``composite``: KU covered ONLY when ALL registered component evidence spans meet the threshold
      (allowing multiple retrieved chunks in Top-K to jointly satisfy the composite KU via union intervals).
    - Also tracks partial evidence coverage, source-backed evidence redundancy (without double-counting
      identical coordinates or KUs), and optional Stage 03 semantic KU support distinction.
    """
    kus = question["required_knowledge_units"]
    ev_by_ku: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ev_dp_map: dict[str, dict[tuple[str, int], list[tuple[int, int]]]] = {}
    ev_total_len: dict[str, int] = {}

    for ev in evidences:
        eid = ev["evidence_id"]
        ev_by_ku[ev["knowledge_unit_id"]].append(ev)
        dp = _evidence_intervals_by_page(ev)
        ev_dp_map[eid] = dp
        ev_total_len[eid] = sum(interval_total_len(ivs) for ivs in dp.values())

    # Precompute per-chunk source-backed intervals & direct evidence matches
    chunk_dp_list = [_chunk_intervals_by_page(ch) for ch in retrieved_chunks]
    chunk_has_valid_ev = [
        any(chunk_matches_evidence(ch, ev) for ev in evidences)
        for ch in retrieved_chunks
    ]
    # Precompute per-chunk matched chars against the union of ALL registered evidences of this query
    all_ev_dp: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for dp in ev_dp_map.values():
        for k_dp, ivs in dp.items():
            all_ev_dp[k_dp].extend(ivs)
    all_ev_dp = {k_dp: merge_intervals(ivs) for k_dp, ivs in all_ev_dp.items()}

    chunk_ev_intersections: list[dict[tuple[str, int], list[tuple[int, int]]]] = []
    for c_dp in chunk_dp_list:
        inter_dp: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for k_dp, c_ivs in c_dp.items():
            if k_dp in all_ev_dp:
                inter = intersect_intervals(c_ivs, all_ev_dp[k_dp])
                if inter:
                    inter_dp[k_dp] = inter
        chunk_ev_intersections.append(inter_dp)

    per_k_rows: list[dict[str, Any]] = []
    ku_trace_by_k: dict[str, Any] = {}

    for k in ks:
        top_n = min(k, len(retrieved_chunks))
        # Union of all source-backed intervals across Top-K retrieved chunks
        union_dp: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for idx in range(top_n):
            for k_dp, ivs in chunk_dp_list[idx].items():
                union_dp[k_dp].extend(ivs)
        union_dp = {k_dp: merge_intervals(ivs) for k_dp, ivs in union_dp.items()}

        # Evaluate each evidence span against the Top-K source-backed union
        ev_satisfied: dict[str, bool] = {}
        ev_cov_ratio: dict[str, float] = {}
        for ev in evidences:
            eid = ev["evidence_id"]
            tot_len = ev_total_len[eid]
            ov_chars = 0
            for k_dp, e_ivs in ev_dp_map[eid].items():
                if k_dp in union_dp:
                    ov_chars += interval_total_len(intersect_intervals(e_ivs, union_dp[k_dp]))
            thresh = _ev_coverage_threshold(tot_len)
            ev_satisfied[eid] = ov_chars >= thresh if tot_len > 0 else False
            ev_cov_ratio[eid] = (ov_chars / tot_len) if tot_len > 0 else 0.0

        # Evaluate each KU according to its evidence_mode
        ku_complete_count = 0
        ku_partial_scores: list[float] = []
        ku_any_supported_count = 0
        ku_details: dict[str, dict[str, Any]] = {}

        for ku in kus:
            ku_id = ku["ku_id"]
            mode = ku.get("evidence_mode", "single")
            ku_evs = ev_by_ku.get(ku_id, [])
            if not ku_evs:
                ku_details[ku_id] = {"mode": mode, "complete": False, "partial_score": 0.0}
                ku_partial_scores.append(0.0)
                continue

            if mode == "composite":
                sat_cnt = sum(1 for ev in ku_evs if ev_satisfied[ev["evidence_id"]])
                is_complete = sat_cnt == len(ku_evs)
                # Partial score = average of component satisfaction (1.0 if satisfied else char coverage ratio)
                part_score = mean(
                    1.0 if ev_satisfied[ev["evidence_id"]] else ev_cov_ratio[ev["evidence_id"]]
                    for ev in ku_evs
                )
            elif mode == "alternative":
                is_complete = any(ev_satisfied[ev["evidence_id"]] for ev in ku_evs)
                part_score = 1.0 if is_complete else max(ev_cov_ratio[ev["evidence_id"]] for ev in ku_evs)
            else:  # single
                is_complete = any(ev_satisfied[ev["evidence_id"]] for ev in ku_evs)
                part_score = 1.0 if is_complete else max(ev_cov_ratio[ev["evidence_id"]] for ev in ku_evs)

            if is_complete:
                ku_complete_count += 1
                ku_any_supported_count += 1
            elif part_score > 0.0:
                ku_any_supported_count += 1

            ku_partial_scores.append(part_score)
            ku_details[ku_id] = {
                "mode": mode,
                "complete": is_complete,
                "partial_score": round(part_score, 4),
            }

        total_kus = len(kus)
        ev_recall = (ku_complete_count / total_kus) if total_kus > 0 else 0.0
        complete_ev_rate = 1.0 if (total_kus > 0 and ku_complete_count == total_kus) else 0.0
        partial_ev_cov = mean(ku_partial_scores) if ku_partial_scores else 0.0
        ku_cov_any = (ku_any_supported_count / total_kus) if total_kus > 0 else 0.0

        # Evidence Precision@K: fraction of retrieved Top-K chunks having valid source-backed evidence overlap
        matched_chunks_k = sum(1 for idx in range(top_n) if chunk_has_valid_ev[idx])
        ev_precision = (matched_chunks_k / top_n) if top_n > 0 else 0.0

        # Evidence Redundancy@K: excess matched source-backed chars beyond unique union across Top-K
        sum_matched_chars = 0
        union_matched_dp: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for idx in range(top_n):
            for k_dp, ivs in chunk_ev_intersections[idx].items():
                sum_matched_chars += interval_total_len(ivs)
                union_matched_dp[k_dp].extend(ivs)
        unique_matched_chars = sum(interval_total_len(ivs) for ivs in union_matched_dp.values())
        redundancy = (
            (sum_matched_chars - unique_matched_chars) / sum_matched_chars
            if sum_matched_chars > 0
            else 0.0
        )

        # Optional semantic KU support from Stage 03 LLM-as-a-Judge annotations (distinct from span provenance!)
        sem_ku_recall = None
        if semantic_ku_by_chunk is not None and total_kus > 0:
            sem_supported: set[str] = set()
            valid_ku_ids = {ku["ku_id"] for ku in kus}
            for idx in range(top_n):
                cid = retrieved_chunks[idx]["chunk_id"]
                sem_supported.update(k_id for k_id in semantic_ku_by_chunk.get(cid, ()) if k_id in valid_ku_ids)
            sem_ku_recall = len(sem_supported) / total_kus

        per_k_rows.append({
            "k": k,
            "actual_returned_count": top_n,
            "total_kus": total_kus,
            "covered_kus": ku_complete_count,
            "evidence_recall_at_k": ev_recall,
            "evidence_precision_at_k": ev_precision,
            "ku_coverage_at_k": ku_cov_any,
            "complete_evidence_rate_at_k": complete_ev_rate,
            "partial_evidence_coverage_at_k": partial_ev_cov,
            "evidence_redundancy_at_k": redundancy,
            "matched_evidence_chunks": matched_chunks_k,
            "sum_matched_evidence_chars": sum_matched_chars,
            "unique_matched_evidence_chars": unique_matched_chars,
            "semantic_ku_recall_at_k": sem_ku_recall,
        })
        if k in (5, 15, 60):
            ku_trace_by_k[str(k)] = ku_details

    return per_k_rows, ku_trace_by_k


def aggregate_evidence_metrics(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average per-query evidence metrics by (combo_id, profile_id, method, k)."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        grouped[(row["combo_id"], row["profile_id"], row["method"], int(row["k"]))].append(row)

    metric_fields = (
        "evidence_recall_at_k",
        "evidence_precision_at_k",
        "ku_coverage_at_k",
        "complete_evidence_rate_at_k",
        "partial_evidence_coverage_at_k",
        "evidence_redundancy_at_k",
        "semantic_ku_recall_at_k",
    )
    combo_order = {cid: idx for idx, (cid, _, _) in enumerate(STAGE04_COMBOS)}
    summary: list[dict[str, Any]] = []
    for (combo_id, pid, method, k), rows in sorted(
        grouped.items(), key=lambda x: (combo_order.get(x[0][0], 99), x[0][3])
    ):
        rec: dict[str, Any] = {
            "combo_id": combo_id,
            "profile_id": pid,
            "method": method,
            "k": k,
            "total_queries": len(rows),
        }
        for mf in metric_fields:
            vals = [r[mf] for r in rows if r.get(mf) is not None]
            rec[mf] = round(mean(vals), 6) if vals else None
        summary.append(rec)
    return summary
