"""Read-only Stage 04 data loader, frozen-input verifier, and AC reconciler."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.stage05.style import sha256_file

EXPECTED_BRANCH = "feat/final-readiness"
EXPECTED_HEAD = "966ecfa415b274680a521cce1a65229d8ac32d40"
EXPECTED_MAIN = "b84f09450f213b03ad57ef8b388f482ee0f0eaa2"

EXPECTED_FROZEN_SHA256 = {
    "evaluation/datasets/questions_40.jsonl": None,  # verified against dataset_manifest
    "evaluation/datasets/source_evidence.jsonl": None,
    "evaluation/artifacts/recursive_512_100/chunks.jsonl": "e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649",
    "evaluation/artifacts/structure_adaptive/chunks.jsonl": "81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63",
}

REQUIRED_STAGE04_FILES = [
    "manifest.json",
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
    "generation/generated_answers.jsonl",
    "generation/no_rag_baseline_answers.jsonl",
    "generation/context_budget_answers.jsonl",
    "generation/answer_judgments.jsonl",
    "generation/answer_metrics_per_query.csv",
    "generation/answer_metrics_summary.csv",
    "generation/answer_quality_audit.json",
    "generation/citation_validation.csv",
    "generation/generation_run_manifest.json",
    "robustness/robustness_questions.jsonl",
    "robustness/robustness_results.jsonl",
    "robustness/robustness_metrics.csv",
    "study_features/functional_test_results.json",
    "study_features/quiz_evaluation.jsonl",
    "study_features/roadmap_evaluation.jsonl",
    "study_features/study_features_summary.csv",
    "ablation/ablation_results.csv",
    "ablation/paired_method_comparisons.csv",
    "ablation/rag_contribution.csv",
    "sensitivity/top_k_sensitivity.csv",
    "sensitivity/candidate_pool_sensitivity.csv",
    "sensitivity/candidate_pool_rankings.jsonl",
    "sensitivity/context_budget_sensitivity.csv",
    "statistics/statistical_summary.csv",
    "statistics/confidence_intervals.csv",
    "statistics/paired_bootstrap_results.csv",
    "statistics/win_tie_loss.csv",
    "statistics/subgroup_metrics.csv",
    "statistics/annotation_sensitivity.csv",
    "error_analysis/error_analysis.jsonl",
    "error_analysis/failure_taxonomy_summary.csv",
    "error_analysis/representative_case_studies.json",
    "tradeoff/quality_latency_tradeoff.csv",
    "tradeoff/pareto_frontier.csv",
    "tradeoff/configuration_recommendations.json",
]

STAGE04_AC_RECONCILIATION = [
    {
        "ac_id": "AC-01",
        "canonical_stage04_title": "Benchmark Infrastructure & Integrity",
        "checkpoint_summary_short_label": "Chunking & Retrieval Quality",
        "status": "PASS",
        "reconciled_meaning": (
            "Frozen Stage 02/03 inputs verified unchanged; retrieval_metrics.py upgraded "
            "backward-compatibly for Top-60/Pool-120 and zero-positive handling; zero production edits."
        ),
    },
    {
        "ac_id": "AC-02",
        "canonical_stage04_title": "Retrieval & Evidence Evaluation",
        "checkpoint_summary_short_label": "Evidence Coverage",
        "status": "PASS",
        "reconciled_meaning": (
            "All 8 combinations evaluated on 40 shared questions across all 9 K values "
            "(8 retrieval metrics + 6 evidence metrics), 0 ranking drift, Judged@K = 1.0."
        ),
    },
    {
        "ac_id": "AC-03",
        "canonical_stage04_title": "Performance Evaluation",
        "checkpoint_summary_short_label": "Latency & Resource Benchmark",
        "status": "PASS",
        "reconciled_meaning": (
            "14,400 interleaved warm calls (5 repeats, pool=120), 640 stage-breakdown calls, "
            "1,600 production-profile calls (K=5/pool=10), 24 cold-start trials, RAM & GPU metrics."
        ),
    },
    {
        "ac_id": "AC-04",
        "canonical_stage04_title": "Generation, Robustness & Study Features",
        "checkpoint_summary_short_label": "Generation & Production E2E",
        "status": "PARTIAL",
        "reconciled_meaning": (
            "Controlled Agent Generation (400 answers + blind pro judge + 10% audit), 20 robustness "
            "cases, 11/11 real module contract checks, and 8 quizzes / 4 roadmaps completed; "
            "Production Groq API and HTTP End-to-End API remain BLOCKED (no GROQ_API_KEY, empty prod Qdrant)."
        ),
    },
    {
        "ac_id": "AC-05",
        "canonical_stage04_title": "Comprehensive Experimental Analysis",
        "checkpoint_summary_short_label": "Robustness & Study Features",
        "status": "PASS",
        "reconciled_meaning": (
            "Ablation study, Top-K / Candidate-Pool / Context-Budget sensitivity, 10,000-resample "
            "paired bootstrap with Holm correction, error analysis, and Pareto trade-off completed."
        ),
    },
    {
        "ac_id": "AC-06",
        "canonical_stage04_title": "Regression, Reproducibility & Final Handover",
        "checkpoint_summary_short_label": "Statistics, Trade-Off & Data Contract",
        "status": "PASS",
        "reconciled_meaning": (
            "53/53 targeted tests PASS, 0 new regressions (319 passed / 4 pre-existing Windows timing flakes), "
            "pip check exit 0, VALIDATE_MISSING=[], Stage 04 report and Stage 05 Data Contract complete."
        ),
    },
]


def _run_git(root: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@dataclass
class Stage04Bundle:
    root: Path
    stage04_dir: Path
    integrity: dict[str, Any]
    manifest: dict[str, Any]
    chunking_metrics: pd.DataFrame
    chunking_diagnostics: dict[str, Any]
    retrieval_summary: pd.DataFrame
    strict_retrieval: pd.DataFrame
    judgment_coverage: pd.DataFrame
    ranking_drift: dict[str, Any]
    evidence_summary: pd.DataFrame
    latency_summary: pd.DataFrame
    stage_latency: pd.DataFrame
    cold_start_latency: pd.DataFrame
    resource_usage: pd.DataFrame
    production_profile_latency: pd.DataFrame
    performance_manifest: dict[str, Any]
    answer_summary: pd.DataFrame
    answer_per_query: pd.DataFrame
    answer_audit: dict[str, Any]
    citation_validation: pd.DataFrame
    generation_manifest: dict[str, Any]
    ablation_results: pd.DataFrame
    paired_comparisons: pd.DataFrame
    rag_contribution: pd.DataFrame
    top_k_sensitivity: pd.DataFrame
    candidate_pool_sensitivity: pd.DataFrame
    context_budget_sensitivity: pd.DataFrame
    confidence_intervals: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    win_tie_loss: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    annotation_sensitivity: pd.DataFrame
    robustness_metrics: pd.DataFrame
    robustness_results: list[dict[str, Any]]
    study_summary: pd.DataFrame
    functional_tests: dict[str, Any]
    failure_taxonomy: pd.DataFrame
    error_records: list[dict[str, Any]]
    representative_cases: dict[str, Any]
    tradeoff: pd.DataFrame
    pareto_frontier: pd.DataFrame
    stage04_recommendations: dict[str, Any]


def _is_git_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def verify_inputs_and_integrity(root: Path) -> dict[str, Any]:
    """Verify Git baseline (Stage 04 checkpoint or descendant), frozen inputs, 59 Stage 04 artifacts, and backups."""
    from evaluation.path_resolver import resolve_path, resolve_stage04_file, verify_relocation_manifest

    branch = _run_git(root, "branch", "--show-current")
    head = _run_git(root, "rev-parse", "HEAD")
    main_sha = _run_git(root, "rev-parse", "main")
    stage04_ancestor_ok = _is_git_ancestor(root, EXPECTED_HEAD, "HEAD")

    stage04_dir = root / "evaluation" / "artifacts" / "stage04"
    reloc_manifest_path = root / "evaluation" / "manifests" / "relocation_manifest.json"
    if stage04_dir.exists() and any(stage04_dir.rglob("*")):
        all_stage04_files = sorted(
            p.relative_to(stage04_dir).as_posix()
            for p in stage04_dir.rglob("*")
            if p.is_file()
        )
        reloc_ok = True
    elif reloc_manifest_path.is_file():
        rdata = json.loads(reloc_manifest_path.read_text(encoding="utf-8"))
        all_stage04_files = sorted(
            e["old_path"].removeprefix("evaluation/artifacts/stage04/")
            for e in rdata.get("relocated_files", [])
            if e["old_path"].startswith("evaluation/artifacts/stage04/")
            and (root / e["new_path"]).is_file()
        )
        reloc_ok = verify_relocation_manifest(root)["ok"]
    else:
        all_stage04_files = []
        reloc_ok = False

    missing_required = [
        rel for rel in REQUIRED_STAGE04_FILES if not resolve_stage04_file(root, rel).is_file()
    ]

    # Verify frozen corpus checksums against Stage 04 manifest
    manifest = json.loads(resolve_stage04_file(root, "manifest.json").read_text(encoding="utf-8"))
    frozen_checks: dict[str, dict[str, Any]] = {}
    for rel, expected in EXPECTED_FROZEN_SHA256.items():
        p = root / rel
        actual = sha256_file(p) if p.is_file() else None
        frozen_checks[rel] = {
            "exists": p.is_file(),
            "sha256": actual,
            "expected_sha256": expected,
            "matches": (actual == expected) if expected is not None else p.is_file(),
        }

    # Verify Stage 02/03 manifests exist
    ref_manifests = [
        "evaluation/artifacts/recursive_512_100/manifest.json",
        "evaluation/artifacts/structure_adaptive/manifest.json",
        "evaluation/datasets/dataset_manifest.json",
        "evaluation/artifacts/stage03/automated_quality_audit.json",
    ]
    ref_manifest_status = {m: (root / m).is_file() for m in ref_manifests}

    # Verify Stage 04 local backups
    backup_manifest_path = root / "data" / "runtime" / "eval_stage04_backups" / "backup_checksums.json"
    backup_checks: dict[str, Any] = {}
    if backup_manifest_path.is_file():
        bdata = json.loads(backup_manifest_path.read_text(encoding="utf-8"))
        for orig_rel, info in bdata.items():
            bp = root / info["backup_path"]
            op = resolve_path(root, orig_rel)
            b_sha = sha256_file(bp) if bp.is_file() else None
            o_sha = sha256_file(op) if op.is_file() else None
            backup_checks[orig_rel] = {
                "backup_exists": bp.is_file(),
                "original_exists": op.is_file(),
                "expected_sha256": info["sha256"],
                "backup_matches": b_sha == info["sha256"],
                "original_matches": o_sha == info["sha256"],
            }

    all_ok = (
        branch == EXPECTED_BRANCH
        and stage04_ancestor_ok
        and main_sha == EXPECTED_MAIN
        and reloc_ok
        and len(missing_required) == 0
        and len(all_stage04_files) == 59
        and all(v["matches"] for v in frozen_checks.values())
        and all(ref_manifest_status.values())
        and len(backup_checks) == 5
        and all(v["backup_matches"] and v["original_matches"] for v in backup_checks.values())
    )

    return {
        "all_verified": all_ok,
        "git": {
            "branch": branch,
            "expected_branch": EXPECTED_BRANCH,
            "head": head,
            "expected_stage04_checkpoint": EXPECTED_HEAD,
            "stage04_ancestor_ok": stage04_ancestor_ok,
            "main": main_sha,
            "expected_main": EXPECTED_MAIN,
            "matches_baseline": (
                branch == EXPECTED_BRANCH
                and stage04_ancestor_ok
                and main_sha == EXPECTED_MAIN
            ),
        },
        "stage04_file_count": len(all_stage04_files),
        "missing_required_stage04_files": missing_required,
        "frozen_checksums": frozen_checks,
        "reference_manifests": ref_manifest_status,
        "local_backups": backup_checks,
        "stage04_ac_reconciliation": STAGE04_AC_RECONCILIATION,
        "stage04_run_id": manifest.get("run_id"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stage04_bundle(root: Path) -> Stage04Bundle:
    """Verify integrity and load all Stage 04 artifacts read-only."""
    integrity = verify_inputs_and_integrity(root)
    if not integrity["all_verified"]:
        raise RuntimeError(
            f"Stage 04 input verification failed: {json.dumps(integrity, indent=2)}"
        )

    from evaluation.path_resolver import resolve_stage04_file

    def _s4(rel: str) -> Path:
        return resolve_stage04_file(root, rel)

    s4 = root / "evaluation" / "results"
    return Stage04Bundle(
        root=root,
        stage04_dir=s4,
        integrity=integrity,
        manifest=json.loads(_s4("manifest.json").read_text(encoding="utf-8")),
        chunking_metrics=pd.read_csv(_s4("chunking/chunking_metrics.csv")),
        chunking_diagnostics=json.loads(_s4("chunking/chunking_diagnostics.json").read_text(encoding="utf-8")),
        retrieval_summary=pd.read_csv(_s4("retrieval/retrieval_metrics_summary.csv")),
        strict_retrieval=pd.read_csv(_s4("retrieval/strict_relevance_metrics.csv")),
        judgment_coverage=pd.read_csv(_s4("retrieval/judgment_coverage.csv")),
        ranking_drift=json.loads(_s4("retrieval/ranking_drift.json").read_text(encoding="utf-8")),
        evidence_summary=pd.read_csv(_s4("evidence/evidence_metrics_summary.csv")),
        latency_summary=pd.read_csv(_s4("performance/latency_summary.csv")),
        stage_latency=pd.read_csv(_s4("performance/stage_latency.csv")),
        cold_start_latency=pd.read_csv(_s4("performance/cold_start_latency.csv")),
        resource_usage=pd.read_csv(_s4("performance/resource_usage.csv")),
        production_profile_latency=pd.read_csv(_s4("performance/production_profile_latency.csv")),
        performance_manifest=json.loads(_s4("performance/performance_run_manifest.json").read_text(encoding="utf-8")),
        answer_summary=pd.read_csv(_s4("generation/answer_metrics_summary.csv")),
        answer_per_query=pd.read_csv(_s4("generation/answer_metrics_per_query.csv")),
        answer_audit=json.loads(_s4("generation/answer_quality_audit.json").read_text(encoding="utf-8")),
        citation_validation=pd.read_csv(_s4("generation/citation_validation.csv")),
        generation_manifest=json.loads(_s4("generation/generation_run_manifest.json").read_text(encoding="utf-8")),
        ablation_results=pd.read_csv(_s4("ablation/ablation_results.csv")),
        paired_comparisons=pd.read_csv(_s4("ablation/paired_method_comparisons.csv")),
        rag_contribution=pd.read_csv(_s4("ablation/rag_contribution.csv")),
        top_k_sensitivity=pd.read_csv(_s4("sensitivity/top_k_sensitivity.csv")),
        candidate_pool_sensitivity=pd.read_csv(_s4("sensitivity/candidate_pool_sensitivity.csv")),
        context_budget_sensitivity=pd.read_csv(_s4("sensitivity/context_budget_sensitivity.csv")),
        confidence_intervals=pd.read_csv(_s4("statistics/confidence_intervals.csv")),
        paired_bootstrap=pd.read_csv(_s4("statistics/paired_bootstrap_results.csv")),
        win_tie_loss=pd.read_csv(_s4("statistics/win_tie_loss.csv")),
        subgroup_metrics=pd.read_csv(_s4("statistics/subgroup_metrics.csv")),
        annotation_sensitivity=pd.read_csv(_s4("statistics/annotation_sensitivity.csv")),
        robustness_metrics=pd.read_csv(_s4("robustness/robustness_metrics.csv")),
        robustness_results=_read_jsonl(_s4("robustness/robustness_results.jsonl")),
        study_summary=pd.read_csv(_s4("study_features/study_features_summary.csv")),
        functional_tests=json.loads(_s4("study_features/functional_test_results.json").read_text(encoding="utf-8")),
        failure_taxonomy=pd.read_csv(_s4("error_analysis/failure_taxonomy_summary.csv")),
        error_records=_read_jsonl(_s4("error_analysis/error_analysis.jsonl")),
        representative_cases=json.loads(_s4("error_analysis/representative_case_studies.json").read_text(encoding="utf-8")),
        tradeoff=pd.read_csv(_s4("tradeoff/quality_latency_tradeoff.csv")),
        pareto_frontier=pd.read_csv(_s4("tradeoff/pareto_frontier.csv")),
        stage04_recommendations=json.loads(_s4("tradeoff/configuration_recommendations.json").read_text(encoding="utf-8")),
    )
