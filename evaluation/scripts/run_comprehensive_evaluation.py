"""Stage 04 – Unified Comprehensive Evaluation Orchestrator CLI."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.stage03.pooling import verify_frozen_corpora_and_index
from evaluation.stage04.analysis import run_comprehensive_analysis
from evaluation.stage04.chunking_eval import run_chunking_evaluation
from evaluation.stage04.generation_eval import run_study_features_contract_checks
from evaluation.stage04.latency_benchmark import run_all_performance_benchmarks
from evaluation.stage04.retrieval_benchmark import (
    run_candidate_pool_sensitivity,
    run_retrieval_and_evidence_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 04 Comprehensive System Evaluation Orchestrator")
    parser.add_argument(
        "--mode",
        choices=[
            "preflight",
            "chunking",
            "retrieval_evidence",
            "candidate_pool",
            "performance",
            "study_contract",
            "analysis",
            "validate",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--cold-trials", type=int, default=3)
    args = parser.parse_args(argv)

    stage04_dir = ROOT / "evaluation/artifacts/stage04"
    stage04_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stage04_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"run_id": f"stage04_{uuid.uuid4().hex[:10]}", "steps": {}}
    )
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    if args.mode in ("preflight", "all"):
        v = verify_frozen_corpora_and_index(ROOT)
        manifest["steps"]["preflight"] = {
            "status": "COMPLETED",
            "canonical_fingerprint_sha256": v["canonical_fingerprint_sha256"],
            "qdrant_total_points": v["qdrant_total_points"],
            "profiles": v["profiles"],
        }
        print("PREFLIGHT_OK:", v["qdrant_total_points"], "points")

    if args.mode in ("retrieval_evidence", "all"):
        re_res = run_retrieval_and_evidence_benchmark(root=ROOT, stage04_dir=stage04_dir, resume=args.resume)
        manifest["steps"]["retrieval_evidence"] = {"status": "COMPLETED", **re_res["ranking_drift"]}
        print("RETRIEVAL_EVIDENCE_OK:", json.dumps(re_res["ranking_drift"]))

    if args.mode in ("chunking", "all"):
        ret_path = stage04_dir / "retrieval/retrieval_rankings.jsonl"
        ch_res = run_chunking_evaluation(
            root=ROOT,
            output_dir=stage04_dir / "chunking",
            retrieval_rankings_path=ret_path if ret_path.exists() else None,
        )
        manifest["steps"]["chunking"] = {"status": "COMPLETED", "combined_qdrant_eval_bytes": ch_res["combined_qdrant_eval_bytes"]}
        print("CHUNKING_OK")

    if args.mode in ("candidate_pool", "all"):
        cp_res = run_candidate_pool_sensitivity(root=ROOT, stage04_dir=stage04_dir, resume=args.resume)
        manifest["steps"]["candidate_pool_sensitivity"] = {"status": "COMPLETED", "rows": len(cp_res)}
        print("CANDIDATE_POOL_OK:", len(cp_res), "rows")

    if args.mode in ("performance", "all"):
        pf_res = run_all_performance_benchmarks(
            root=ROOT,
            stage04_dir=stage04_dir,
            warm_repeats=args.warm_repeats,
            cold_trials=args.cold_trials,
            resume=args.resume,
        )
        manifest["steps"]["performance"] = {"status": "COMPLETED", "total_measured_samples": pf_res["warm_protocol"]["total_measured_samples"]}
        print("PERFORMANCE_OK:", pf_res["warm_protocol"]["total_measured_samples"], "samples")

    if args.mode in ("study_contract", "all"):
        st_res = run_study_features_contract_checks(root=ROOT, output_dir=stage04_dir / "study_features")
        manifest["steps"]["study_features_contract"] = {
            "status": "COMPLETED",
            "passed": st_res["contract_checks_passed"],
            "total": st_res["contract_checks_total"],
        }
        print("STUDY_CONTRACT_OK:", st_res["contract_checks_passed"], "/", st_res["contract_checks_total"])

    if args.mode in ("analysis", "all"):
        an_res = run_comprehensive_analysis(root=ROOT, stage04_dir=stage04_dir)
        manifest["steps"]["analysis"] = {"status": "COMPLETED", **an_res["recommendations"]}
        print("ANALYSIS_OK:", json.dumps(an_res["recommendations"]))

    if args.mode == "validate":
        required = [
            "chunking/chunking_metrics.csv",
            "chunking/chunking_diagnostics.json",
            "retrieval/retrieval_rankings.jsonl",
            "retrieval/ranking_drift.json",
            "retrieval/retrieval_metrics_per_query.csv",
            "retrieval/retrieval_metrics_summary.csv",
            "retrieval/strict_relevance_metrics.csv",
            "retrieval/judgment_coverage.csv",
            "evidence/evidence_metrics_per_query.csv",
            "evidence/evidence_metrics_summary.csv",
            "evidence/knowledge_unit_coverage.jsonl",
            "evidence/evidence_redundancy.csv",
            "performance/latency_raw.jsonl",
            "performance/latency_summary.csv",
            "performance/stage_latency.csv",
            "performance/cold_start_latency.csv",
            "performance/resource_usage.csv",
            "performance/production_profile_latency.csv",
            "performance/performance_run_manifest.json",
            "ablation/ablation_results.csv",
            "ablation/paired_method_comparisons.csv",
            "sensitivity/top_k_sensitivity.csv",
            "sensitivity/candidate_pool_sensitivity.csv",
            "statistics/statistical_summary.csv",
            "statistics/paired_bootstrap_results.csv",
            "statistics/confidence_intervals.csv",
            "statistics/win_tie_loss.csv",
            "statistics/subgroup_metrics.csv",
            "statistics/annotation_sensitivity.csv",
            "error_analysis/error_analysis.jsonl",
            "error_analysis/failure_taxonomy_summary.csv",
            "error_analysis/representative_case_studies.json",
            "tradeoff/quality_latency_tradeoff.csv",
            "tradeoff/pareto_frontier.csv",
            "tradeoff/configuration_recommendations.json",
            "study_features/functional_test_results.json",
        ]
        from evaluation.path_resolver import resolve_stage04_file

        missing = [r for r in required if not resolve_stage04_file(ROOT, r).exists()]
        print("VALIDATE_MISSING:", missing)
        return 1 if missing else 0

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
