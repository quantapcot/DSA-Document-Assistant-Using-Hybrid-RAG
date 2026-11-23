"""Verify the CLI handoff using an in-memory data fixture, never benchmark data."""
import csv
import json

import pytest
from qdrant_client import QdrantClient

from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.indexing import build_index
from evaluation.retrieval_results import KS, METHODS, load_results
from evaluation.scripts import run_benchmark


@pytest.mark.parametrize("extension", ["json", "csv"])
def test_benchmark_output_can_be_plotted_directly(tmp_path, monkeypatch, extension):
    from evaluation.scripts import plot_retrieval_metrics

    corpus = tmp_path / "fixture-corpus.jsonl"
    chunks = [
        {"chunk_id": "tree", "document_id": "d1", "filename": "tree.pdf", "page": 1, "text": "binary tree"},
        {"chunk_id": "queue", "document_id": "d2", "filename": "queue.pdf", "page": 2, "text": "FIFO queue"},
    ]
    corpus.write_text("\n".join(map(json.dumps, chunks)), encoding="utf-8")
    queries = tmp_path / "fixture-queries.jsonl"
    queries.write_text(json.dumps({
        "query_id": "fixture-1", "query": "binary tree", "relevant_ids": ["tree"],
        "relevance_grades": {"tree": 2, "queue": 0}, "labels_complete": True,
    }), encoding="utf-8")
    config_file = tmp_path / "retrieval.yaml"
    config_file.write_text("qdrant_mode: memory\nreranker_model: fixture\n", encoding="utf-8")

    def engine_factory(config):
        assert config.candidate_top_k == 35
        client = QdrantClient(":memory:")
        build_index(corpus, corpus_id="fixture", workspace_id="test", config=config, client=client,
                    embedder=lambda texts: [[1., 0.] if "tree" in t else [0., 1.] for t in texts], dimension=2)
        return RetrievalEngine(config=config, client=client, query_embedder=lambda _: [1., 0.],
                               rerank_scorer=lambda pairs: [float("tree" in text) for _, text in pairs])

    monkeypatch.setattr(run_benchmark, "RetrievalEngine", engine_factory)
    result = tmp_path / "results" / f"run.{extension}"
    run_benchmark.main([
        "--queries", str(queries), "--corpus", str(corpus), "--workspace-id", "test", "--corpus-id", "fixture",
        "--config", str(config_file), "--candidate-pool", "35", "--split", "fixture", "--output", str(result),
    ])
    data = load_results(result)
    assert data["provenance"]["ks"] == list(KS)
    assert data["provenance"]["candidate_pool"] == 35
    assert data["provenance"]["retrieved_top_k"] == 15
    assert len(data["queries"]) == 4

    output = tmp_path / "plots"
    plot_retrieval_metrics.main(["--input", str(result), "--output-dir", str(output)])
    assert {path.name for path in output.glob("*.png")} == {
        "metric_comparison_k15.png", "metrics_by_k.png", "latency_comparison.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output.glob("*.png"))
    with (output / "metrics_summary.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    with (output / "latency_summary.csv").open(encoding="utf-8", newline="") as stream:
        latency = list(csv.DictReader(stream))
    assert {(row["method"], int(row["k"])) for row in metrics} == {
        (method, k) for method in METHODS for k in KS
    }
    assert all(row["samples"] == "1" and float(row["p95_ms"]) >= 0 for row in latency)
