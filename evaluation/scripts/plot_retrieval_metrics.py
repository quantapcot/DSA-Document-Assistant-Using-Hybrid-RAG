"""Plot ranking quality and latency directly from benchmark results."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from evaluation.retrieval_results import KS, METRIC_KEYS, load_results, report_summary

METRICS = tuple(zip(METRIC_KEYS, ("Precision", "Recall", "MRR", "nDCG")))
METHOD_LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid_rrf": "Hybrid RRF",
                 "hybrid_rerank": "Hybrid Rerank"}
MARKERS = {"bm25": "o", "dense": "s", "hybrid_rrf": "^", "hybrid_rerank": "D"}
LATENCIES = (("mean_ms", "Mean"), ("p50_ms", "p50"), ("p95_ms", "p95"))


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_comparison(rows, methods, k, output_dir):
    scores = {row["method"]: row for row in rows if row["k"] == k}
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        positions = [position + (index - (len(methods) - 1) / 2) * width for position in range(len(METRICS))]
        ax.bar(positions, [scores[method][key] for key, _ in METRICS], width, label=METHOD_LABELS[method])
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS])
    ax.set(ylabel="Mean score", ylim=(0, 1.05), title=f"Retrieval quality at K={k}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    return save_plot(fig, output_dir / f"metric_comparison_k{k}.png")


def plot_by_k(rows, methods, ks, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for method in methods:
            group = [row for row in rows if row["method"] == method]
            ax.plot([row["k"] for row in group], [row[metric] for row in group],
                    marker=MARKERS[method], label=METHOD_LABELS[method])
        ax.set(title=f"{label}@K", xlabel="K", ylabel="Mean score", ylim=(0, 1.05), xticks=sorted(ks))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Retrieval quality across cutoffs")
    return save_plot(fig, output_dir / "metrics_by_k.png")


def plot_latency(rows, output_dir, provenance):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.23
    for index, (key, label) in enumerate(LATENCIES):
        positions = [position + (index - 1) * width for position in range(len(rows))]
        ax.bar(positions, [row[key] for row in rows], width, label=label)
    ax.set_xticks(range(len(rows)), [METHOD_LABELS[row["method"]] for row in rows])
    profile = provenance.get("latency_profile")
    top_k = provenance.get("latency_top_k", provenance["retrieved_top_k"])
    pool = provenance["candidate_pool"]
    protocol = f"{profile}, warmed" if profile else "legacy, no warmup"
    ax.set(ylabel="Latency (ms)", title=f"Top-{top_k}/pool {pool} retrieval latency ({protocol})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    filename = "latency_comparison_production.png" if profile == "production" else "latency_comparison.png"
    return save_plot(fig, output_dir / filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Benchmark .json or .csv output")
    parser.add_argument("--output-dir", default="evaluation/results/plots")
    parser.add_argument("--comparison-k", type=int, choices=KS, help="K for the bar chart; defaults to the largest reported K")
    args = parser.parse_args(argv)
    try:
        data = load_results(args.input)
        provenance = data["provenance"]
        methods, ks = provenance["methods"], provenance["ks"]
        summary = report_summary(data)
        k = args.comparison_k if args.comparison_k is not None else max(ks, default=None)
        if k is not None and k not in ks:
            raise ValueError(f"K={k} is not present in this benchmark")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    production = provenance.get("latency_profile") == "production"
    latency_csv = "latency_summary_production.csv" if production else "latency_summary.csv"
    paths = [
        save_csv(summary["latency"], output_dir / latency_csv),
        plot_latency(summary["latency"], output_dir, provenance),
    ]
    if summary["metrics"]:
        paths += [
            save_csv(summary["metrics"], output_dir / "metrics_summary.csv"),
            plot_comparison(summary["metrics"], methods, k, output_dir),
            plot_by_k(summary["metrics"], methods, ks, output_dir),
        ]
    for path in paths:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
