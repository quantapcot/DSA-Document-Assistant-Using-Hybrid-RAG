"""Result schema shared by the offline benchmark and plotter."""
import csv
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean, median, quantiles

from evaluation.metrics.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
KS = (1, 3, 5, 10, 15)
RETRIEVE_TOP_K = KS[-1]
DEFAULT_CANDIDATE_POOL = 30
SCHEMA_VERSION = 1
METRIC_KEYS = ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
JSON_FIELDS = ("relevant_ids", "relevance_grades", "retrieved_ids")
ROW_FIELDS = ("query_id", "method", *JSON_FIELDS, "labels_complete", "latency_ms")


def validate_options(methods, ks):
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Select distinct, supported retrieval methods")
    if not ks or any(type(k) is not int or k not in KS for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"Select distinct K values from {KS}")


def validate_labels(row):
    """Require complete graded judgments rather than inferring missing labels."""
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    qid = row.get("query_id")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("Each query needs a nonempty query_id")
    relevant = row.get("relevant_ids")
    if not isinstance(relevant, list) or not all(isinstance(i, str) and i.strip() for i in relevant):
        raise ValueError(f"{qid}: expected labeled relevant_ids")
    grades = row.get("relevance_grades")
    if not isinstance(grades, dict) or any(
        not isinstance(i, str) or not i.strip() or type(grade) is not int or grade not in (0, 1, 2)
        for i, grade in grades.items()
    ):
        raise ValueError(f"{qid}: relevance_grades must map chunk IDs to integer grades 0, 1 or 2")
    positive = {i for i, grade in grades.items() if grade > 0}
    if not positive or set(relevant) != positive or len(relevant) != len(positive):
        raise ValueError(f"{qid}: relevant_ids must contain each positively graded chunk once")
    if row.get("labels_complete") is not True:
        raise ValueError(f"{qid}: labels_complete must be true")


def summarize(rows, methods, ks):
    """Build macro averages over the same queries and one latency per ranking."""
    validate_options(methods, ks)
    by_method = defaultdict(list)
    labels_by_query = {}
    seen = set()
    for row in rows:
        validate_labels(row)
        qid, method = row["query_id"], row.get("method")
        key = (qid, method)
        if method not in methods or key in seen:
            raise ValueError(f"Unexpected or duplicate query/method: {key}")
        seen.add(key)
        grades = row["relevance_grades"]
        if labels_by_query.setdefault(qid, grades) != grades:
            raise ValueError(f"{qid}: inconsistent labels across methods")
        ranking, latency = row.get("retrieved_ids"), row.get("latency_ms")
        if not isinstance(ranking, list) or len(ranking) > RETRIEVE_TOP_K or not all(
            isinstance(item, str) and item.strip() for item in ranking
        ):
            raise ValueError(f"{key}: invalid Top-{RETRIEVE_TOP_K} ranking")
        if type(latency) not in (int, float) or not isfinite(latency) or latency < 0:
            raise ValueError(f"{key}: invalid latency")
        by_method[method].append(row)
    if not labels_by_query or any(len(by_method[method]) != len(labels_by_query) for method in methods):
        raise ValueError("Every selected method must use the same nonempty query set")

    metrics, latencies = [], []
    for method in methods:
        group = by_method[method]
        for k in sorted(ks):
            scores = [
                (
                    precision_at_k(row["retrieved_ids"], row["relevant_ids"], k),
                    recall_at_k(row["retrieved_ids"], row["relevant_ids"], k),
                    reciprocal_rank(row["retrieved_ids"], row["relevant_ids"], k),
                    ndcg_at_k(row["retrieved_ids"], row["relevance_grades"], k),
                )
                for row in group
            ]
            metrics.append({"method": method, "k": k, **dict(zip(METRIC_KEYS, map(mean, zip(*scores))))})
        samples = [row["latency_ms"] for row in group]
        p95 = quantiles(samples, n=20, method="inclusive")[-1] if len(samples) > 1 else samples[0]
        latencies.append({"method": method, "samples": len(samples),
                          "mean_ms": mean(samples), "p50_ms": median(samples), "p95_ms": p95})
    return {"metrics": metrics, "latency": latencies}


def write_results(output, provenance, rows, summary):
    output = Path(output)
    if output.suffix.lower() not in {".json", ".csv"}:
        raise ValueError("Output must end in .json or .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "provenance": provenance, "summary": summary}
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps({**header, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            packed = {key: json.dumps(row[key], ensure_ascii=False) for key in (*JSON_FIELDS, "labels_complete")}
            writer.writerow({**row, **packed})
    output.with_suffix(".metadata.json").write_text(
        json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def load_results(path):
    """Read benchmark JSON or its CSV variant with the metadata sidecar."""
    path = Path(path)
    if path.suffix.lower() not in {".json", ".csv"}:
        raise ValueError("Use benchmark .json or .csv output as input")
    header_path = path.with_suffix(".metadata.json") if path.suffix.lower() == ".csv" else path
    data = json.loads(header_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema; run the benchmark with the current runner")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or set(reader.fieldnames) != set(ROW_FIELDS):
                raise ValueError("Invalid benchmark CSV columns")
            data["queries"] = [
                {**row, **{key: json.loads(row[key]) for key in (*JSON_FIELDS, "labels_complete")},
                 "latency_ms": float(row["latency_ms"])}
                for row in reader
            ]
    return data
