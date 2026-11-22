
"""Plot EVAL-02 metrics from labeled retrieval results."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
)

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
KS = (1, 3, 5)
METRICS = (
    ("precision_at_k", "Precision"),
    ("recall_at_k", "Recall"),
    ("mrr_at_k", "MRR"),
    ("ndcg_at_k", "nDCG"),
)

LINE_STYLES = {
    "bm25": {
        "marker": "o",
        "linestyle": "-",
        "x_offset": -0.18,
    },
    "dense": {
        "marker": "s",
        "linestyle": "--",
        "x_offset": -0.06,
    },
    "hybrid_rrf": {
        "marker": "^",
        "linestyle": "-.",
        "x_offset": 0.06,
    },
    "hybrid_rerank": {
        "marker": "D",
        "linestyle": ":",
        "x_offset": 0.18,
    },
}


def load_data(path):
    """Load and validate labeled JSONL retrieval results."""
    rows = []
    seen = set()

    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            required = (
                "query_id", "method", "retrieved_ids",
                "relevant_ids", "relevance_grades",
                "labels_complete",
            )
            if any(key not in row for key in required):
                raise ValueError(f"Line {line_no}: missing fields")

            qid = row["query_id"]
            method = row["method"]

            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"Line {line_no}: invalid query_id")

            if method not in METHODS:
                raise ValueError(f"Line {line_no}: unknown method")

            key = (qid, method)
            if key in seen:
                raise ValueError(f"Duplicate query/method: {key}")
            seen.add(key)

            for field in ("retrieved_ids", "relevant_ids"):
                ids = row[field]
                if (
                    not isinstance(ids, list)
                    or any(not isinstance(x, str) or not x for x in ids)
                ):
                    raise ValueError(f"Line {line_no}: invalid {field}")

            grades = row["relevance_grades"]
            if not isinstance(grades, dict):
                raise ValueError(f"Line {line_no}: invalid grades")

            if any(
                type(v) is not int or v not in (0, 1, 2)
                for v in grades.values()
            ):
                raise ValueError(f"Line {line_no}: invalid grade value")

            positive_ids = {
                chunk_id for chunk_id, grade in grades.items()
                if grade > 0
            }

            if set(row["relevant_ids"]) != positive_ids:
                raise ValueError(
                    f"Line {line_no}: relevant_ids and grades disagree"
                )

            if not positive_ids:
                raise ValueError(
                    f"Line {line_no}: no relevant chunks"
                )

            if row["labels_complete"] is not True:
                raise ValueError(
                    f"Line {line_no}: incomplete labels"
                )

            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    if set(grouped) != set(METHODS):
        raise ValueError("All four retrieval methods are required")

    expected_queries = {r["query_id"] for r in grouped[METHODS[0]]}

    for method in METHODS:
        current = {r["query_id"] for r in grouped[method]}
        if current != expected_queries:
            raise ValueError("Methods have different query sets")

    # Ground truth must be identical for the same query.
    labels_by_query = {}
    for row in rows:
        qid = row["query_id"]
        labels = (
            set(row["relevant_ids"]),
            row["relevance_grades"],
        )
        if qid in labels_by_query and labels != labels_by_query[qid]:
            raise ValueError(f"Inconsistent labels for {qid}")
        labels_by_query[qid] = labels

    return grouped


def calculate_summary(grouped):
    summary = []

    for method in METHODS:
        rows = grouped[method]

        for k in KS:
            summary.append({
                "method": method,
                "k": k,
                "precision_at_k": mean(
                    precision_at_k(
                        r["retrieved_ids"], r["relevant_ids"], k
                    )
                    for r in rows
                ),
                "recall_at_k": mean(
                    recall_at_k(
                        r["retrieved_ids"], r["relevant_ids"], k
                    )
                    for r in rows
                ),
                "mrr_at_k": mrr_at_k(
                    [r["retrieved_ids"] for r in rows],
                    [r["relevant_ids"] for r in rows],
                    k,
                ),
                "ndcg_at_k": mean(
                    ndcg_at_k(
                        r["retrieved_ids"],
                        r["relevance_grades"],
                        k,
                    )
                    for r in rows
                ),
            })

    return summary


def save_csv(summary, output_dir):
    path = output_dir / "metrics_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    return path


def plot_comparison(summary, output_dir):
    """Grouped bar chart for K=5."""
    scores = {
        row["method"]: row
        for row in summary if row["k"] == 5
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    width = 0.18
    positions = range(len(METRICS))

    for index, method in enumerate(METHODS):
        values = [scores[method][key] for key, _ in METRICS]
        offsets = [
            x + (index - 1.5) * width for x in positions
        ]
        ax.bar(offsets, values, width, label=method)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Metrics Comparison (K=5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    path = output_dir / "metric_comparison_k5.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_by_k(summary, output_dir):
    """Metric trends for K=1,3,5 with visually separated lines."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, (metric_key, metric_label) in zip(
        axes.flat, METRICS
    ):
        for method in METHODS:
            rows = sorted(
                (r for r in summary if r["method"] == method),
                key=lambda r: r["k"],
            )

            style = LINE_STYLES[method]

            x_values = [r["k"] + style["x_offset"] for r in rows]
            y_values = [r[metric_key] for r in rows]

            ax.plot(
                x_values,
                y_values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=7,
                markeredgewidth=1.2,
                label=method,
                alpha=0.95,
                zorder=3,
            )

        ax.set_title(f"{metric_label}@K")
        ax.set_xlabel("K")
        ax.set_ylabel("Mean score")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, zorder=0)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Retrieval Metrics across K\n"
        "(small horizontal offsets are used only for visual separation)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / "metrics_by_k.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Plot EVAL-02 Retrieval Metrics"
    )
    parser.add_argument(
        "--input", required=True,
        help="Labeled retrieval results in JSONL format",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    grouped = load_data(args.input)
    summary = calculate_summary(grouped)

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        save_csv(summary, output_dir),
        plot_comparison(summary, output_dir),
        plot_by_k(summary, output_dir),
    ):
        print(f"Created: {path}")

    print("PLOTTING COMPLETED")


if __name__ == "__main__":
    main()
