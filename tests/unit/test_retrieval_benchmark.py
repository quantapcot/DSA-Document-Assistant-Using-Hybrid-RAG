"""Benchmark protocol and summary tests using only synthetic judgments."""
from math import log2
from unittest.mock import Mock

import pytest

from app.retrieval.config import RetrievalConfig
from evaluation.retrieval_results import KS, METHODS, summarize
from evaluation.scripts.run_benchmark import evaluation_config, run


def test_each_method_retrieves_top_15_once_without_changing_base_config():
    base = RetrievalConfig(candidate_top_k=10)
    config = evaluation_config(base)
    engine = Mock(config=config)
    engine.retrieve.return_value = [{"chunk_id": str(i)} for i in range(1, 16)]
    rows = run(engine, [{
        "query_id": "q1", "query": "fixture", "relevant_ids": ["10", "15"],
        "relevance_grades": {"10": 2, "15": 1}, "labels_complete": True,
    }], "corpus", "workspace")
    assert base.candidate_top_k == 10 and config.candidate_top_k == 30
    assert [call.args[1] for call in engine.retrieve.call_args_list] == list(METHODS)
    assert all(call.kwargs == {"top_k": 15, "corpus_id": "corpus", "workspace_id": "workspace"}
               for call in engine.retrieve.call_args_list)
    metrics = {row["k"]: row for row in summarize(rows, METHODS, KS)["metrics"] if row["method"] == "bm25"}
    assert tuple(metrics) == (1, 3, 5, 10, 15)
    assert metrics[5]["mrr_at_k"] == 0
    assert metrics[10]["precision_at_k"] == pytest.approx(1 / 10)
    assert metrics[10]["recall_at_k"] == 0.5
    assert metrics[10]["mrr_at_k"] == pytest.approx(1 / 10)
    assert metrics[15]["recall_at_k"] == 1
    assert metrics[15]["ndcg_at_k"] == pytest.approx((3 / log2(11) + 1 / log2(16)) / (3 + 1 / log2(3)))


def test_pool_validation_and_benchmark_uses_selected_override():
    base = RetrievalConfig(candidate_top_k=12)
    with pytest.raises(ValueError, match="at least 15"):
        evaluation_config(base, 14)
    override = evaluation_config(base, 35)
    engine = Mock(config=override)
    engine.retrieve.return_value = []
    query = [{
        "query_id": "q", "query": "fixture", "relevant_ids": ["a"],
        "relevance_grades": {"a": 2}, "labels_complete": True,
    }]
    with pytest.raises(ValueError, match="selected candidate pool"):
        run(engine, query, "c", "w", methods=["bm25"])
    assert not engine.retrieve.called
    assert len(run(engine, query, "c", "w", methods=["bm25"], candidate_pool=35)) == 1


def test_latency_summary_counts_rankings_not_cutoffs():
    rows = [{
        "query_id": str(i), "method": "dense", "retrieved_ids": ["a"],
        "relevance_grades": {"a": 2}, "relevant_ids": ["a"], "labels_complete": True,
        "latency_ms": latency,
    } for i, latency in enumerate([10, 20, 100])]
    summary = summarize(rows, ["dense"], KS)
    assert len(summary["metrics"]) == 5
    assert summary["latency"] == [{
        "method": "dense", "samples": 3, "mean_ms": pytest.approx(130 / 3),
        "p50_ms": 20, "p95_ms": pytest.approx(92),
    }]
    singleton = summarize(rows[:1], ["dense"], KS)["latency"][0]
    assert singleton["mean_ms"] == singleton["p50_ms"] == singleton["p95_ms"] == 10


@pytest.mark.parametrize("extra", [
    {"labels_complete": False}, {"relevance_grades": {"a": True}},
    {"relevance_grades": {"b": 2}}, {"relevant_ids": ["a", "a"]},
])
def test_incomplete_or_inconsistent_ground_truth_never_calls_retrieval(extra):
    engine = Mock(config=evaluation_config(RetrievalConfig()))
    queries = [{
        "query_id": "q", "query": "fixture", "relevant_ids": ["a"],
        "relevance_grades": {"a": 2}, "labels_complete": True, **extra,
    }]
    with pytest.raises(ValueError):
        run(engine, queries, "c", "w")
    assert not engine.retrieve.called
