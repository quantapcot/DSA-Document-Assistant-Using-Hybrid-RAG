"""CLI entrypoint for Supplementary Stage 04 AC-04 Completion (Live Groq & HTTP E2E)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from evaluation.ac04_completion import (
    PROTOCOL_HTTP_CONFIG_A,
    PROTOCOL_HTTP_CONFIG_B,
)
from evaluation.ac04_completion.groq_smoke import run_groq_smoke_test
from evaluation.ac04_completion.http_experiments import (
    run_edge_robustness_quiz_roadmap,
    run_http_chat_batch,
    run_live_groq_no_rag,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "http_smoke", "http_a", "http_b", "no_rag", "study_robust", "aggregate", "figures", "validate", "all_live"),
        required=True,
    )
    args = parser.parse_args(argv)

    if args.mode == "smoke":
        res = run_groq_smoke_test(ROOT)
        ok = bool(res["groq_smoke_test"]["connectivity_ok"])
        print("SMOKE_OK:", ok, "MODEL:", res["groq_smoke_test"]["returned_model"])
        return 0 if ok else 1

    if args.mode == "http_smoke":
        rows = run_http_chat_batch(
            ROOT,
            tag="config_a_smoke",
            protocol=PROTOCOL_HTTP_CONFIG_A,
            candidate_top_k=10,
            out_filename="http_smoke_3q.jsonl",
            query_ids=["Q001", "Q011", "Q026"],
        )
        print("HTTP_SMOKE_COUNT:", len(rows))
        return 0

    if args.mode == "http_a":
        rows = run_http_chat_batch(
            ROOT,
            tag="config_a_pool10",
            protocol=PROTOCOL_HTTP_CONFIG_A,
            candidate_top_k=10,
            out_filename="http_chat_config_a.jsonl",
        )
        print("HTTP_CONFIG_A_COUNT:", len(rows))
        return 0

    if args.mode == "http_b":
        rows = run_http_chat_batch(
            ROOT,
            tag="config_b_pool30",
            protocol=PROTOCOL_HTTP_CONFIG_B,
            candidate_top_k=30,
            out_filename="http_chat_config_b.jsonl",
        )
        print("HTTP_CONFIG_B_COUNT:", len(rows))
        return 0

    if args.mode == "no_rag":
        rows = run_live_groq_no_rag(ROOT)
        print("LIVE_GROQ_NO_RAG_COUNT:", len(rows))
        return 0

    if args.mode == "study_robust":
        res = run_edge_robustness_quiz_roadmap(ROOT)
        print("STUDY_ROBUST_COUNTS:", json.dumps(res))
        return 0

    if args.mode == "all_live":
        run_http_chat_batch(
            ROOT, tag="config_a_pool10", protocol=PROTOCOL_HTTP_CONFIG_A,
            candidate_top_k=10, out_filename="http_chat_config_a.jsonl",
        )
        run_http_chat_batch(
            ROOT, tag="config_b_pool30", protocol=PROTOCOL_HTTP_CONFIG_B,
            candidate_top_k=30, out_filename="http_chat_config_b.jsonl",
        )
        run_live_groq_no_rag(ROOT)
        run_edge_robustness_quiz_roadmap(ROOT)
        return 0

    if args.mode == "aggregate":
        from evaluation.ac04_completion.judging import aggregate_all_ac04_metrics
        summary = aggregate_all_ac04_metrics(ROOT)
        print("AGGREGATE_OK:", json.dumps(summary["gates"], indent=2))
        return 0

    if args.mode == "figures":
        from evaluation.ac04_completion.visualizations import generate_ac04_figures
        figs = generate_ac04_figures(ROOT)
        print("GENERATED_SUPPLEMENTARY_FIGURES:", len(figs))
        return 0

    if args.mode == "validate":
        from evaluation.ac04_completion.visualizations import validate_ac04_completion
        res = validate_ac04_completion(ROOT)
        print(json.dumps(res["summary"], indent=2))
        return 0 if res["overall_pass"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
