"""Benchmark protocol and summary tests using only synthetic judgments."""
from math import log2
from unittest.mock import Mock

import pytest

from app.retrieval.config import RetrievalConfig
from evaluation.retrieval_results import KS, METHODS, summarize, summarize_latency
from evaluation.scripts import run_benchmark
from evaluation.scripts.run_benchmark import evaluation_config, measure_latency, run


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
        "query_id": "q", "method": "dense", "repeat": repeat, "latency_ms": latency,
    } for repeat, latency in enumerate([10, 20, 100], start=1)]
    summary = summarize_latency(rows, ["dense"], 3)
    assert summary == [{
        "method": "dense", "samples": 3, "mean_ms": pytest.approx(130 / 3),
        "p50_ms": 20, "p95_ms": pytest.approx(92),
    }]
    singleton = summarize_latency(rows[:1], ["dense"], 1)[0]
    assert singleton["mean_ms"] == singleton["p50_ms"] == singleton["p95_ms"] == 10
    with pytest.raises(ValueError, match="configured number"):
        summarize_latency(rows[:2], ["dense"], 3)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_latency([*rows, rows[0]], ["dense"], 3)


def test_warmup_uses_full_workload_and_is_not_timed(monkeypatch):
    engine = Mock()
    queries = [{"query_id": str(i), "query": f"fixture {i}"} for i in range(2)]
    round_size = len(queries) * len(METHODS)
    now = 0

    def retrieve(*args, **kwargs):
        nonlocal now
        # Warmup is deliberately much slower than any measured call.
        now += 700_000_000 if engine.retrieve.call_count <= round_size else 11_000_000

    engine.retrieve.side_effect = retrieve
    clock = Mock(side_effect=lambda: now)
    monkeypatch.setattr(run_benchmark.time, "perf_counter_ns", clock)
    samples = measure_latency(engine, queries, "c", "w", seed=42)
    assert len(samples) == round_size * 5
    assert clock.call_count == 2 * len(samples) and engine.retrieve.call_count == round_size * 6
    expected = {(query["query"], method) for query in queries for method in METHODS}
    calls = engine.retrieve.call_args_list
    rounds = [calls[i:i + round_size] for i in range(0, len(calls), round_size)]
    assert all({call.args for call in batch} == expected for batch in rounds)
    assert len({tuple(call.args for call in batch) for batch in rounds[1:]}) > 1
    assert all(call.kwargs == {"top_k": 15, "corpus_id": "c", "workspace_id": "w"} for call in calls)
    assert {sample["repeat"] for sample in samples} == {1, 2, 3, 4, 5}
    assert all(sample["latency_ms"] == 11 for sample in samples)


def test_latency_pass_does_not_supply_rankings_to_quality_metrics():
    engine = Mock(config=evaluation_config(RetrievalConfig()))
    query = [{
        "query_id": "q", "query": "fixture", "relevant_ids": ["good"],
        "relevance_grades": {"good": 2}, "labels_complete": True,
    }]
    methods = ["dense"]
    engine.retrieve.side_effect = [
        [{"chunk_id": "good"}],                          # quality baseline
        *[[{"chunk_id": "wrong"}] for _ in range(5)],   # 2 warmups and 3 timed
        [{"chunk_id": "good"}],                          # separate quality pass
    ]
    baseline = summarize(run(engine, query, "c", "w", methods), methods, KS)
    samples = measure_latency(engine, query, "c", "w", methods, repeats=3, warmup_rounds=2)
    after = summarize(run(engine, query, "c", "w", methods), methods, KS)
    assert len(samples) == 3 and all("retrieved_ids" not in sample for sample in samples)
    assert engine.retrieve.call_count == 7
    assert after == baseline and after["metrics"][0]["recall_at_k"] == 1


@pytest.mark.parametrize("options", [{"repeats": 0}, {"repeats": True}, {"warmup_rounds": 0}])
def test_invalid_latency_options_fail_before_retrieval(options):
    engine = Mock()
    with pytest.raises(ValueError, match="positive integer"):
        measure_latency(engine, [{"query_id": "q", "query": "fixture"}], "c", "w", **options)
    assert not engine.retrieve.called


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
