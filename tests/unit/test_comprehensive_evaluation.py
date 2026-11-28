"""Unit tests for Stage 04 Comprehensive Evaluation framework (metrics, evidence, drift, latency, stats, generation helpers)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics.retrieval_metrics import (
    average_precision_at_k,
    f1_at_k,
    hit_rate_at_k,
    judged_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.stage04.analysis import (
    bootstrap_ci,
    compute_pareto_frontier,
    holm_bonferroni_adjust,
    paired_bootstrap_test,
)
from evaluation.stage04.evidence_metrics import (
    evaluate_query_evidence,
    intersect_intervals,
    merge_intervals,
)
from evaluation.stage04.generation_eval import (
    build_independent_robustness_dataset,
    pack_context_by_token_budget,
    validate_answer_citations,
)
from evaluation.stage04.latency_benchmark import percentile_linear, summarize_latency_samples
from evaluation.stage04.metrics import (
    STAGE04_KS,
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
    compute_ranking_drift,
)


def test_retrieval_metrics_manual_and_edge_cases():
    # Perfect ranking at K=3 with graded relevance 2, 1, 0
    grades = {"c1": 2, "c2": 1, "c3": 0, "c4": 0}
    retrieved = ["c1", "c2", "c3"]
    assert precision_at_k(retrieved, {"c1", "c2"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, {"c1", "c2"}, 3) == pytest.approx(1.0)
    assert hit_rate_at_k(retrieved, {"c1", "c2"}, 3) == 1.0
    assert average_precision_at_k(retrieved, {"c1", "c2"}, 3) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, grades, 3) == pytest.approx(1.0)
    assert f1_at_k(2 / 3, 1.0) == pytest.approx(0.8)

    # Worst ranking (no relevant in top-3)
    assert hit_rate_at_k(["c3", "c4"], {"c1", "c2"}, 3) == 0.0
    assert average_precision_at_k(["c3", "c4"], {"c1", "c2"}, 3) == 0.0
    assert f1_at_k(0.0, 0.0) == 0.0

    # Partial relevance & AP cutoff normalization min(K, total_rel)
    # Relevant={"a","b","c","d"}, retrieved=["x","a","b"], K=3 -> hits at rank 2 (1/2) & rank 3 (2/3), denom=min(3,4)=3
    assert average_precision_at_k(["x", "a", "b"], {"a", "b", "c", "d"}, 3) == pytest.approx((0.5 + 2 / 3) / 3)

    # Duplicate chunk IDs must NOT inflate AP
    assert average_precision_at_k(["a", "a", "b"], {"a", "b"}, 3) == pytest.approx((1.0 + 2 / 3) / 2)

    # Empty ranking & fewer-than-K judged_at_k
    assert judged_at_k([], {"a"}, 5) == (0.0, 0, 0)
    assert judged_at_k(["a", "u1", "b"], {"a", "b"}, 5) == (pytest.approx(2 / 3), 2, 3)

    # Zero-positive query handling: default raises, policy="none" returns None
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    assert recall_at_k(["a"], set(), 5, zero_positive_policy="none") is None
    assert average_precision_at_k(["a"], set(), 5, zero_positive_policy="none") is None
    assert f1_at_k(0.0, None) is None


def test_stage04_top60_zero_positive_unjudged_and_macro_aggregation():
    # Q1 has positive grades, Q2 is zero-positive, Q1 has an unjudged chunk "u_new"
    q1_rows = compute_query_retrieval_metrics(
        ["c1", "u_new", "c2"] + [f"z{i}" for i in range(57)],
        {"c1": 2, "c2": 1, "c3": 0},
        ks=STAGE04_KS,
    )
    assert len(q1_rows) == 9 and q1_rows[-1]["k"] == 60
    r3 = next(r for r in q1_rows if r["k"] == 3)
    assert r3["judged_count"] == 2 and r3["unjudged_count"] == 1 and r3["actual_returned_count"] == 3
    assert r3["judged_at_k"] == pytest.approx(2 / 3)

    q2_rows = compute_query_retrieval_metrics(["c9"], {"c9": 0}, ks=STAGE04_KS)
    r5_q2 = next(r for r in q2_rows if r["k"] == 5)
    assert r5_q2["zero_positive"] is True
    assert r5_q2["pooled_recall_at_k"] is None
    assert r5_q2["map_at_k"] is None
    assert r5_q2["f1_at_k"] is None
    assert r5_q2["actual_returned_count"] == 1

    # Strict Grade-2 sensitivity treats Grade-1 as 0
    strict_rows = compute_query_retrieval_metrics(["c2", "c1"], {"c1": 2, "c2": 1}, ks=(1, 2), min_positive_grade=2)
    assert strict_rows[0]["precision_at_k"] == 0.0
    assert strict_rows[1]["pooled_recall_at_k"] == 1.0

    # Macro aggregation reports valid_n explicitly (1 valid query for recall/map/f1, 2 for precision)
    tagged = [
        {"combo_id": "A1", "profile_id": "recursive_512_100", "method": "bm25", **r}
        for r in (q1_rows + q2_rows)
    ]
    agg = aggregate_retrieval_metrics(tagged)
    agg_k5 = next(r for r in agg if r["k"] == 5)
    assert agg_k5["total_queries"] == 2
    assert agg_k5["zero_positive_queries"] == 1
    assert agg_k5["precision_at_k_valid_n"] == 2
    assert agg_k5["pooled_recall_at_k_valid_n"] == 1


def test_evidence_intervals_single_alternative_composite_and_redundancy():
    assert merge_intervals([(0, 10), (5, 20), (25, 30)]) == [(0, 20), (25, 30)]
    assert intersect_intervals([(0, 15), (20, 30)], [(10, 25)]) == [(10, 15), (20, 25)]

    question = {
        "query_id": "Q1",
        "required_knowledge_units": [
            {"ku_id": "KU1", "evidence_mode": "single"},
            {"ku_id": "KU2", "evidence_mode": "alternative"},
            {"ku_id": "KU3", "evidence_mode": "composite"},
        ],
    }
    evidences = [
        {"evidence_id": "E1", "knowledge_unit_id": "KU1", "document_id": "d1", "source_spans": [{"page": 1, "char_start": 0, "char_end": 50}]},
        {"evidence_id": "E2a", "knowledge_unit_id": "KU2", "document_id": "d1", "source_spans": [{"page": 1, "char_start": 100, "char_end": 150}]},
        {"evidence_id": "E2b", "knowledge_unit_id": "KU2", "document_id": "d2", "source_spans": [{"page": 1, "char_start": 0, "char_end": 50}]},
        {"evidence_id": "E3a", "knowledge_unit_id": "KU3", "document_id": "d1", "source_spans": [{"page": 2, "char_start": 0, "char_end": 40}]},
        {"evidence_id": "E3b", "knowledge_unit_id": "KU3", "document_id": "d1", "source_spans": [{"page": 2, "char_start": 60, "char_end": 100}]},
    ]
    # Chunks: c1 covers E1 + E3a; c2 overlaps E1 (creates redundancy!) + covers E2b via overlap_segment;
    # c3 has ONLY non-source-backed synthetic for E3b (must NOT count!); c4 covers E3b validly.
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "metadata": {"source_spans": [{"page": 1, "char_start": 0, "char_end": 50}, {"page": 2, "char_start": 0, "char_end": 40}]}},
        {"chunk_id": "c2", "document_id": "d1", "metadata": {"source_spans": [{"page": 1, "char_start": 0, "char_end": 50}]}},
        {"chunk_id": "c2b", "document_id": "d2", "metadata": {"source_spans": [], "overlap_segments": [{"page": 1, "char_start": 0, "char_end": 50}]}},
        {"chunk_id": "c3_fake", "document_id": "d1", "metadata": {"source_spans": [], "synthetic_segments": [{"source_backed": False, "page": 2, "char_start": 60, "char_end": 100}]}},
        {"chunk_id": "c4", "document_id": "d1", "metadata": {"source_spans": [{"page": 2, "char_start": 60, "char_end": 100}]}},
    ]
    rows, _ = evaluate_query_evidence(question=question, evidences=evidences, retrieved_chunks=chunks, ks=(1, 3, 4, 5))
    r1, r3, r4, r5 = rows
    assert r1["covered_kus"] == 1  # KU1 complete, KU3 only partial (E3a)
    assert r1["evidence_recall_at_k"] == pytest.approx(1 / 3)
    assert r1["complete_evidence_rate_at_k"] == 0.0
    assert r1["partial_evidence_coverage_at_k"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    assert r3["covered_kus"] == 2  # KU1 + KU2 (via E2b overlap)
    assert r3["evidence_redundancy_at_k"] > 0.0  # c1 and c2 both cover E1 [0,50)

    # At K=4, c3_fake is NOT source-backed so KU3 still incomplete!
    assert r4["covered_kus"] == 2
    assert r4["complete_evidence_rate_at_k"] == 0.0

    # At K=5, c1 + c4 jointly cover composite KU3!
    assert r5["covered_kus"] == 3
    assert r5["evidence_recall_at_k"] == pytest.approx(1.0)
    assert r5["complete_evidence_rate_at_k"] == 1.0


def test_ranking_drift_latency_stats_and_generation_helpers():
    s3 = [{"query_id": "Q1", "profile_id": "recursive_512_100", "method": "bm25", "results": [{"chunk_id": "a", "score": 2.0}, {"chunk_id": "b", "score": 1.0}]}]
    s4 = [{"query_id": "Q1", "profile_id": "recursive_512_100", "method": "bm25", "results": [{"chunk_id": "a", "score": 2.0}, {"chunk_id": "c_new", "score": 1.0}]}]
    gt = {("Q1", "recursive_512_100"): {"relevance_grades": {"a": 2, "b": 0}}}
    drift = compute_ranking_drift(s3, s4, gt, ks=(1, 2, 5, 60))
    assert drift["exact_ranking_matches"] == 0
    assert drift["unique_unjudged_candidates"] == 1

    # Latency percentiles & aggregation
    assert percentile_linear([10.0, 20.0, 30.0, 40.0, 50.0], 0.5) == pytest.approx(30.0)
    samples = [
        {"combo_id": "A1", "profile_id": "recursive_512_100", "method": "bm25", "k": 5, "query_id": "Q1", "latency_ms": 10.0, "error": None},
        {"combo_id": "A1", "profile_id": "recursive_512_100", "method": "bm25", "k": 5, "query_id": "Q1", "latency_ms": 20.0, "error": None},
        {"combo_id": "A1", "profile_id": "recursive_512_100", "method": "bm25", "k": 5, "query_id": "Q2", "latency_ms": 30.0, "error": None},
    ]
    lsum = summarize_latency_samples(samples, group_keys=("combo_id", "k"))
    assert lsum[0]["sample_count"] == 3 and lsum[0]["mean_ms"] == pytest.approx(20.0)

    # Bootstrap reproducibility, Holm adjustment & Pareto frontier
    ci1 = bootstrap_ci([0.2, 0.5, 0.8, 0.9], n_resamples=500, seed=42)
    ci2 = bootstrap_ci([0.2, 0.5, 0.8, 0.9], n_resamples=500, seed=42)
    assert ci1 == ci2 and ci1[1] <= ci1[0] <= ci1[2]

    pb = paired_bootstrap_test([0.9, 0.8, 0.7, 0.85], [0.4, 0.5, 0.3, 0.45], n_resamples=500, seed=42)
    assert pb["wins"] == 4 and pb["losses"] == 0 and pb["abs_diff"] > 0.3
    assert holm_bonferroni_adjust([0.01, 0.04, 0.20]) == [0.03, 0.08, 0.2]

    pf = compute_pareto_frontier(
        [
            {"combo_id": "A", "ndcg": 0.9, "lat": 100.0},
            {"combo_id": "B", "ndcg": 0.7, "lat": 20.0},
            {"combo_id": "C", "ndcg": 0.6, "lat": 50.0},  # dominated by B
        ],
        maximize_keys=("ndcg",),
        minimize_keys=("lat",),
    )
    assert {r["combo_id"]: r["is_pareto_optimal"] for r in pf} == {"A": True, "B": True, "C": False}

    # Context packing & citation validation
    class _SpaceTok:
        def encode(self, s, add_special_tokens=False):
            return s.split()
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    packed = pack_context_by_token_budget(
        [{"chunk_id": "c1", "filename": "f.pdf", "page": 1, "text": "one two three"},
         {"chunk_id": "c2", "filename": "f.pdf", "page": 2, "text": "four five six seven eight nine ten"}],
        tokenizer=_SpaceTok(),
        token_budget=14,
    )
    assert packed["selected_chunk_ids"] == ["c1"] and packed["truncated_last_chunk"] is False
    cit = validate_answer_citations("QuickSort is O(n log n) [1] and [9] invalid.", 3)
    assert cit["valid_citations"] == [1] and cit["invalid_citations"] == [9] and cit["citation_validity"] == 0.5

    # Robustness dataset has exact 20 cases with required category counts
    rob = build_independent_robustness_dataset()
    assert len(rob) == 20
