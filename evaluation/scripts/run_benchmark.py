"""Evaluate ranking quality and warmed latency on a pre-indexed scope."""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from evaluation.retrieval_results import (
    DEFAULT_CANDIDATE_POOL, KS, METHODS, RETRIEVE_TOP_K,
    PRODUCTION_CANDIDATE_POOL, PRODUCTION_TOP_K, validate_labels, validate_options, write_results,
)

DEFAULT_REPEATS = 5
DEFAULT_WARMUP_ROUNDS = 1
DEFAULT_SEED = 42


def validate_candidate_pool(candidate_pool):
    if type(candidate_pool) is not int or candidate_pool < RETRIEVE_TOP_K:
        raise ValueError(f"Candidate pool must be at least {RETRIEVE_TOP_K}")


def evaluation_config(config, candidate_pool=DEFAULT_CANDIDATE_POOL):
    validate_candidate_pool(candidate_pool)
    return replace(config, candidate_top_k=candidate_pool)


def validate_queries(queries, *, graded=True):
    if not queries:
        raise ValueError("No queries found; provide an existing query file")
    seen = set()
    for row in queries:
        if graded:
            validate_labels(row)
        elif not isinstance(row, dict) or not isinstance(row.get("query_id"), str) or not row["query_id"].strip():
            raise ValueError("Each query needs a nonempty query_id")
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"{row['query_id']}: missing query text")
        if row["query_id"] in seen:
            raise ValueError(f"Duplicate query_id: {row['query_id']}")
        seen.add(row["query_id"])


def load_queries(path, *, graded=True):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Queries line {line_number}: invalid JSON") from exc
    validate_queries(rows, graded=graded)
    return rows


def run(engine, queries, corpus_id, workspace_id, methods=METHODS, *,
        candidate_pool=DEFAULT_CANDIDATE_POOL):
    """Collect exactly one untimed Top-15 quality ranking per query and method."""
    validate_options(methods, KS)
    validate_queries(queries)
    validate_candidate_pool(candidate_pool)
    if engine.config.candidate_top_k != candidate_pool:
        raise ValueError("Build the benchmark engine with evaluation_config and the selected candidate pool")
    rows = []
    for row in queries:
        for method in methods:
            results = engine.retrieve(row["query"], method, top_k=RETRIEVE_TOP_K,
                                      corpus_id=corpus_id, workspace_id=workspace_id)
            rows.append({
                "query_id": row["query_id"],
                "method": method,
                "retrieved_ids": [result["chunk_id"] for result in results],
                "relevant_ids": row["relevant_ids"],
                "relevance_grades": row["relevance_grades"],
                "labels_complete": True,
            })
    return rows


def validate_latency_counts(repeats, warmup_rounds):
    for name, value in (("Repeats", repeats), ("Warmup rounds", warmup_rounds)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def measure_latency(engine, queries, corpus_id, workspace_id, methods=METHODS, *,
                    top_k=RETRIEVE_TOP_K, repeats=DEFAULT_REPEATS,
                    warmup_rounds=DEFAULT_WARMUP_ROUNDS, seed=DEFAULT_SEED):
    """Warm the complete workload, then time sequential interleaved calls."""
    validate_options(methods, KS)
    validate_queries(queries, graded=False)
    validate_latency_counts(repeats, warmup_rounds)
    random = Random(seed)
    method_order = list(methods)
    random.shuffle(method_order)
    samples = []
    for round_index in range(warmup_rounds + repeats):
        shuffled = list(queries)
        random.shuffle(shuffled)
        repeat = round_index - warmup_rounds + 1
        for query_index, row in enumerate(shuffled):
            offset = (query_index + round_index) % len(method_order)
            for method in method_order[offset:] + method_order[:offset]:
                if repeat > 0:
                    start = time.perf_counter_ns()
                engine.retrieve(row["query"], method, top_k=top_k,
                                corpus_id=corpus_id, workspace_id=workspace_id)
                if repeat > 0:
                    samples.append({
                        "query_id": row["query_id"], "method": method, "repeat": repeat,
                        "latency_ms": (time.perf_counter_ns() - start) / 1_000_000,
                    })
    return samples


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Existing JSONL; evaluation profile requires complete graded judgments")
    parser.add_argument("--corpus", required=True, help="Original indexed JSONL, used for SHA-256 provenance")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--config", help="Retrieval YAML; candidate pool is overridden for this run")
    parser.add_argument("--output", required=True, help="Result .json or .csv path")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--ks", nargs="+", type=int, choices=KS)
    parser.add_argument("--latency-profile", choices=("evaluation", "production"), default="evaluation",
                        help="evaluation measures Top-15 and reports quality; production measures only Top-5/pool 10")
    parser.add_argument("--candidate-pool", type=int, help=f"Evaluation candidate pool (default: {DEFAULT_CANDIDATE_POOL})")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timed rounds (default: %(default)s)")
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS,
                        help="Untimed full rounds before measuring (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for query and method ordering")
    parser.add_argument("--split", required=True, help="Label for the evaluation split")
    args = parser.parse_args(argv)
    try:
        production = args.latency_profile == "production"
        ks = [] if production else args.ks or KS
        validate_options(args.methods, ks or KS)
        validate_latency_counts(args.repeats, args.warmup_rounds)
        if production and (args.ks or args.candidate_pool not in (None, PRODUCTION_CANDIDATE_POOL)):
            raise ValueError("Production is latency-only with a fixed pool of 10; do not supply --ks or a different pool")
        pool = PRODUCTION_CANDIDATE_POOL if production else (
            DEFAULT_CANDIDATE_POOL if args.candidate_pool is None else args.candidate_pool
        )
        if not production:
            validate_candidate_pool(pool)
        if Path(args.output).suffix.lower() not in {".json", ".csv"}:
            raise ValueError("Output must end in .json or .csv")
        try:
            queries = load_queries(args.queries, graded=not production)
        except FileNotFoundError:
            kind = "Query" if production else "Ground-truth"
            raise ValueError(f"{kind} JSONL not found: {args.queries}") from None
        hashes = {"corpus_sha256": sha256(args.corpus), "queries_sha256": sha256(args.queries)}
        base = RetrievalConfig.from_yaml(args.config) if args.config else RetrievalConfig.from_env()
        config = replace(base, candidate_top_k=pool) if production else evaluation_config(base, pool)
        if "hybrid_rerank" in args.methods and not config.reranker_model:
            raise ValueError("hybrid_rerank requires a configured reranker; use --config or choose other methods")
        engine = RetrievalEngine(config=config)
        latency_top_k = PRODUCTION_TOP_K if production else RETRIEVE_TOP_K
        try:
            samples = measure_latency(
                engine, queries, args.corpus_id, args.workspace_id, args.methods, top_k=latency_top_k,
                repeats=args.repeats, warmup_rounds=args.warmup_rounds, seed=args.seed,
            )
            rows = [] if production else run(engine, queries, args.corpus_id, args.workspace_id, args.methods,
                                             candidate_pool=pool)
        finally:
            engine.client.close()
        provenance = {
            **hashes, "config": {key: value for key, value in asdict(config).items()
                                 if key not in {"qdrant_url", "qdrant_path"}},
            "tokenizer": "PyVi + Unicode NFC + casefold",
            "corpus_id": args.corpus_id, "workspace_id": args.workspace_id, "split": args.split,
            "python": platform.python_version(), "ks": sorted(ks),
            "retrieved_top_k": None if production else RETRIEVE_TOP_K,
            "candidate_pool": pool, "methods": list(args.methods), "query_count": len(queries),
            "latency_profile": args.latency_profile, "latency_top_k": latency_top_k,
            "latency_unit": "ms", "latency_repeats": args.repeats,
            "latency_warmup_rounds": args.warmup_rounds, "latency_seed": args.seed,
        }
        write_results(args.output, provenance, rows, samples)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(rows)} quality rankings and {len(samples)} {args.latency_profile} latency samples to {args.output}")


if __name__ == "__main__":
    main()
