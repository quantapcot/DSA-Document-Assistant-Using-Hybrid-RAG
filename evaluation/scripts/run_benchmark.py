"""Evaluate a pre-indexed scope using complete graded JSONL queries."""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from evaluation.retrieval_results import (
    DEFAULT_CANDIDATE_POOL, KS, METHODS, RETRIEVE_TOP_K,
    summarize, validate_labels, validate_options, write_results,
)


def validate_candidate_pool(candidate_pool):
    if type(candidate_pool) is not int or candidate_pool < RETRIEVE_TOP_K:
        raise ValueError(f"Candidate pool must be at least {RETRIEVE_TOP_K}")


def evaluation_config(config, candidate_pool=DEFAULT_CANDIDATE_POOL):
    validate_candidate_pool(candidate_pool)
    return replace(config, candidate_top_k=candidate_pool)


def validate_queries(queries):
    if not queries:
        raise ValueError("No labeled queries found; evaluation requires existing ground truth")
    seen = set()
    for row in queries:
        validate_labels(row)
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"{row['query_id']}: missing query text")
        if row["query_id"] in seen:
            raise ValueError(f"Duplicate query_id: {row['query_id']}")
        seen.add(row["query_id"])


def load_queries(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Queries line {line_number}: invalid JSON") from exc
    validate_queries(rows)
    return rows


def run(engine, queries, corpus_id, workspace_id, methods=METHODS, *,
        candidate_pool=DEFAULT_CANDIDATE_POOL):
    """Collect one Top-15 ranking and elapsed time per query and method."""
    validate_options(methods, KS)
    validate_queries(queries)
    validate_candidate_pool(candidate_pool)
    if engine.config.candidate_top_k != candidate_pool:
        raise ValueError("Build the benchmark engine with evaluation_config and the selected candidate pool")
    rows = []
    for row in queries:
        for method in methods:
            start = time.perf_counter()
            results = engine.retrieve(row["query"], method, top_k=RETRIEVE_TOP_K,
                                      corpus_id=corpus_id, workspace_id=workspace_id)
            latency = (time.perf_counter() - start) * 1000
            rows.append({
                "query_id": row["query_id"],
                "method": method,
                "retrieved_ids": [result["chunk_id"] for result in results],
                "relevant_ids": row["relevant_ids"],
                "relevance_grades": row["relevance_grades"],
                "labels_complete": True,
                "latency_ms": latency,
            })
    return rows


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Existing JSONL with complete graded relevance judgments")
    parser.add_argument("--corpus", required=True, help="Original indexed JSONL, used for SHA-256 provenance")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--config", help="Retrieval YAML; candidate pool is overridden for this run")
    parser.add_argument("--output", required=True, help="Result .json or .csv path")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--ks", nargs="+", type=int, choices=KS, default=KS)
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL,
                        help=f"Candidates per search stage (at least {RETRIEVE_TOP_K}; default: %(default)s)")
    parser.add_argument("--split", required=True, help="Label for the evaluation split")
    args = parser.parse_args(argv)
    try:
        validate_options(args.methods, args.ks)
        validate_candidate_pool(args.candidate_pool)
        if Path(args.output).suffix.lower() not in {".json", ".csv"}:
            raise ValueError("Output must end in .json or .csv")
        try:
            queries = load_queries(args.queries)
        except FileNotFoundError:
            raise ValueError(f"Ground-truth JSONL not found: {args.queries}") from None
        hashes = {"corpus_sha256": sha256(args.corpus), "queries_sha256": sha256(args.queries)}
        base = RetrievalConfig.from_yaml(args.config) if args.config else RetrievalConfig.from_env()
        config = evaluation_config(base, args.candidate_pool)
        if "hybrid_rerank" in args.methods and not config.reranker_model:
            raise ValueError("hybrid_rerank requires a configured reranker; use --config or choose other methods")
        engine = RetrievalEngine(config=config)
        try:
            rows = run(engine, queries, args.corpus_id, args.workspace_id, args.methods,
                       candidate_pool=args.candidate_pool)
        finally:
            engine.client.close()
        summary = summarize(rows, args.methods, args.ks)
        provenance = {
            **hashes, "config": {key: value for key, value in asdict(config).items()
                                 if key not in {"qdrant_url", "qdrant_path"}},
            "tokenizer": "PyVi + Unicode NFC + casefold",
            "corpus_id": args.corpus_id, "workspace_id": args.workspace_id, "split": args.split,
            "python": platform.python_version(), "ks": sorted(args.ks), "retrieved_top_k": RETRIEVE_TOP_K,
            "candidate_pool": args.candidate_pool, "methods": list(args.methods),
            "latency_unit": "ms", "warmup": False,
        }
        write_results(args.output, provenance, rows, summary)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(rows)} rankings to {args.output}")


if __name__ == "__main__":
    main()
