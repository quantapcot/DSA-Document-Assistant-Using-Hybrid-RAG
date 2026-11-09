"""Evaluate a pre-indexed scope using labeled JSONL queries."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from evaluation.metrics.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")

def run(engine, queries, corpus_id, workspace_id, methods=METHODS, k=5):
    if not queries: raise ValueError("query set is empty")
    rows = []
    for row in queries:
        if not isinstance(row.get("query"), str) or not row["query"].strip() or not isinstance(row.get("relevant_ids"), list) or not row["relevant_ids"] or not all(isinstance(x, str) and x for x in row["relevant_ids"]):
            raise ValueError("Each query needs nonempty query and labeled relevant_ids")
        for method in methods:
            start = time.perf_counter()
            results = engine.retrieve(row["query"], method, top_k=k, corpus_id=corpus_id, workspace_id=workspace_id)
            latency = (time.perf_counter() - start) * 1000
            ids = [r["chunk_id"] for r in results]
            rows.append({"query_id": row.get("query_id", ""), "method": method, "precision_at_k": precision_at_k(ids, row["relevant_ids"], k), "recall_at_k": recall_at_k(ids, row["relevant_ids"], k), "reciprocal_rank": reciprocal_rank(ids, row["relevant_ids"]), "latency_ms": latency})
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, help="JSONL: query_id, query, relevant_ids")
    p.add_argument("--corpus", required=True, help="Original indexed JSONL, used for SHA-256 provenance")
    p.add_argument("--corpus-id", required=True); p.add_argument("--workspace-id", required=True)
    p.add_argument("--config"); p.add_argument("--output", required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--split", required=True, help="e.g. development, held-out test")
    args = p.parse_args()
    config = RetrievalConfig.from_yaml(args.config) if args.config else RetrievalConfig.from_env()
    with open(args.queries, encoding="utf-8") as f: queries = [json.loads(line) for line in f if line.strip()]
    rows = run(RetrievalEngine(config=config), queries, args.corpus_id, args.workspace_id, args.methods)
    summary = {}
    for method in args.methods:
        group = [r for r in rows if r["method"] == method]
        summary[method] = {key: sum(r[key] for r in group) / len(group) for key in ("precision_at_k", "recall_at_k", "reciprocal_rank", "latency_ms")}
    provenance = {"config": asdict(config), "tokenizer": "PyVi + Unicode NFC + casefold", "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(), "queries_sha256": hashlib.sha256(Path(args.queries).read_bytes()).hexdigest(), "corpus_id": args.corpus_id, "workspace_id": args.workspace_id, "split": args.split, "python": platform.python_version(), "k": 5}
    output = Path(args.output)
    if output.suffix.lower() == ".csv":
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        output.with_suffix(".metadata.json").write_text(json.dumps({"provenance":provenance, "summary":summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output.suffix.lower() == ".json":
        output.write_text(json.dumps({"provenance":provenance, "summary":summary, "queries":rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    else: raise ValueError("Output must end in .json or .csv")

if __name__ == "__main__": main()
