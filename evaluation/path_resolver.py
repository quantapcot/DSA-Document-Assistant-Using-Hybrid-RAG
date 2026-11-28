"""Backward-compatible path resolver for consolidated Evaluation deliverables.

Resolves canonical post-consolidation paths under:
- evaluation/results/
- evaluation/figures/{consolidated,original}/
- evaluation/tables/
- evaluation/manifests/
- evaluation/reports/

while preserving full backward compatibility with legacy evaluation/artifacts/
and root-level STAGE_04/05_*.md paths without mutating historical checksums.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Explicit deterministic mapping of legacy artifact paths -> canonical consolidated paths.
LEGACY_TO_CANONICAL_PREFIXES: tuple[tuple[str, str], ...] = (
    # Stage 04 protocol-separated results
    ("evaluation/artifacts/stage04/generation/", "evaluation/results/generation/controlled_agent/"),
    ("evaluation/artifacts/stage04/robustness/", "evaluation/results/robustness/controlled/"),
    ("evaluation/artifacts/stage04/study_features/", "evaluation/results/study_features/controlled/"),
    ("evaluation/artifacts/stage04/chunking/", "evaluation/results/chunking/"),
    ("evaluation/artifacts/stage04/retrieval/", "evaluation/results/retrieval/"),
    ("evaluation/artifacts/stage04/evidence/", "evaluation/results/evidence/"),
    ("evaluation/artifacts/stage04/performance/", "evaluation/results/performance/"),
    ("evaluation/artifacts/stage04/ablation/", "evaluation/results/ablation/"),
    ("evaluation/artifacts/stage04/sensitivity/", "evaluation/results/sensitivity/"),
    ("evaluation/artifacts/stage04/statistics/", "evaluation/results/statistics/"),
    ("evaluation/artifacts/stage04/error_analysis/", "evaluation/results/error_analysis/"),
    ("evaluation/artifacts/stage04/tradeoff/", "evaluation/results/tradeoff/"),
    # Stage 05 figures & tables
    ("evaluation/artifacts/stage05/figures/png/", "evaluation/figures/original/png/"),
    ("evaluation/artifacts/stage05/figures/svg/", "evaluation/figures/original/svg/"),
    ("evaluation/artifacts/stage05/tables/", "evaluation/tables/"),
    # Supplementary AC-04 figures
    ("evaluation/artifacts/ac04_completion/figures/png/", "evaluation/figures/original/png/"),
    ("evaluation/artifacts/ac04_completion/figures/svg/", "evaluation/figures/original/svg/"),
)

LEGACY_TO_CANONICAL_FILES: dict[str, str] = {
    # Historical manifests (relocated intact without editing historical SHA-256 hashes)
    "evaluation/artifacts/stage04/manifest.json": "evaluation/manifests/stage04_manifest.json",
    "evaluation/artifacts/stage05/manifest.json": "evaluation/manifests/stage05_manifest.json",
    "evaluation/artifacts/stage05/figure_manifest.json": "evaluation/manifests/stage05_figure_manifest.json",
    "evaluation/artifacts/stage05/chart_validation.json": "evaluation/manifests/stage05_chart_validation.json",
    "evaluation/artifacts/stage05/configuration_recommendations.json": "evaluation/manifests/stage05_configuration_recommendations.json",
    "evaluation/artifacts/ac04_completion/manifest.json": "evaluation/manifests/ac04_manifest.json",
    "evaluation/artifacts/ac04_completion/figure_manifest.json": "evaluation/manifests/ac04_figure_manifest.json",
    "evaluation/artifacts/ac04_completion/ac04_gates_assessment.json": "evaluation/manifests/ac04_gates_assessment.json",
    "evaluation/artifacts/ac04_completion/smoke_test_report.json": "evaluation/manifests/ac04_smoke_test_report.json",
    # Supplementary AC-04 live Groq generation artifacts
    "evaluation/artifacts/ac04_completion/answer_judgments_live.jsonl": "evaluation/results/generation/live_groq/answer_judgments_live.jsonl",
    "evaluation/artifacts/ac04_completion/answer_metrics_per_query_live.csv": "evaluation/results/generation/live_groq/answer_metrics_per_query_live.csv",
    "evaluation/artifacts/ac04_completion/answer_quality_audit_live.json": "evaluation/results/generation/live_groq/answer_quality_audit_live.json",
    "evaluation/artifacts/ac04_completion/answer_quality_summary.csv": "evaluation/results/generation/live_groq/answer_quality_summary.csv",
    "evaluation/artifacts/ac04_completion/judge_inputs_chat.jsonl": "evaluation/results/generation/live_groq/judge_inputs_chat.jsonl",
    "evaluation/artifacts/ac04_completion/live_groq_no_rag.jsonl": "evaluation/results/generation/live_groq/live_groq_no_rag.jsonl",
    "evaluation/artifacts/ac04_completion/secondary_audit_live.jsonl": "evaluation/results/generation/live_groq/secondary_audit_live.jsonl",
    # Supplementary AC-04 HTTP E2E artifacts
    "evaluation/artifacts/ac04_completion/edge_cases_summary.csv": "evaluation/results/http_e2e/edge_cases_summary.csv",
    "evaluation/artifacts/ac04_completion/http_chat_config_a.jsonl": "evaluation/results/http_e2e/http_chat_config_a.jsonl",
    "evaluation/artifacts/ac04_completion/http_chat_config_b.jsonl": "evaluation/results/http_e2e/http_chat_config_b.jsonl",
    "evaluation/artifacts/ac04_completion/http_edge_cases.jsonl": "evaluation/results/http_e2e/http_edge_cases.jsonl",
    "evaluation/artifacts/ac04_completion/http_latency_summary.csv": "evaluation/results/http_e2e/http_latency_summary.csv",
    "evaluation/artifacts/ac04_completion/http_smoke_3q.jsonl": "evaluation/results/http_e2e/http_smoke_3q.jsonl",
    "evaluation/artifacts/ac04_completion/token_usage_summary.csv": "evaluation/results/http_e2e/token_usage_summary.csv",
    # Supplementary AC-04 live HTTP robustness artifacts
    "evaluation/artifacts/ac04_completion/http_robustness.jsonl": "evaluation/results/robustness/live_http/http_robustness.jsonl",
    "evaluation/artifacts/ac04_completion/judge_inputs_robustness.jsonl": "evaluation/results/robustness/live_http/judge_inputs_robustness.jsonl",
    "evaluation/artifacts/ac04_completion/robustness_judgments_live.jsonl": "evaluation/results/robustness/live_http/robustness_judgments_live.jsonl",
    "evaluation/artifacts/ac04_completion/robustness_summary_live.csv": "evaluation/results/robustness/live_http/robustness_summary_live.csv",
    # Supplementary AC-04 live HTTP study features artifacts
    "evaluation/artifacts/ac04_completion/http_quiz.jsonl": "evaluation/results/study_features/live_http/http_quiz.jsonl",
    "evaluation/artifacts/ac04_completion/http_roadmap.jsonl": "evaluation/results/study_features/live_http/http_roadmap.jsonl",
    "evaluation/artifacts/ac04_completion/judge_inputs_study.jsonl": "evaluation/results/study_features/live_http/judge_inputs_study.jsonl",
    "evaluation/artifacts/ac04_completion/study_features_summary_live.csv": "evaluation/results/study_features/live_http/study_features_summary_live.csv",
    "evaluation/artifacts/ac04_completion/study_judgments_live.jsonl": "evaluation/results/study_features/live_http/study_judgments_live.jsonl",
    # Official reports
    "STAGE_02_DUAL_CHUNKING_INDEXING_REPORT.md": "evaluation/reports/02_chunking_indexing.md",
    "STAGE_03_GROUND_TRUTH_REPORT.md": "evaluation/reports/03_dataset_ground_truth.md",
    "STAGE_04_COMPREHENSIVE_EVALUATION_REPORT.md": "evaluation/reports/04_system_evaluation.md",
    "STAGE_04_AC04_COMPLETION_REPORT.md": "evaluation/reports/04_system_evaluation.md",
    "STAGE_05_FINAL_EVALUATION_REPORT.md": "evaluation/reports/05_final_evaluation.md",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_relpath(rel_path: str) -> str:
    """Map a legacy relative path to its canonical post-consolidation relative path."""
    norm = rel_path.replace("\\", "/").lstrip("./")
    if norm in LEGACY_TO_CANONICAL_FILES:
        return LEGACY_TO_CANONICAL_FILES[norm]
    for old_prefix, new_prefix in LEGACY_TO_CANONICAL_PREFIXES:
        if norm.startswith(old_prefix):
            return new_prefix + norm[len(old_prefix):]
    return norm


def resolve_path(root: Path, rel_path: str) -> Path:
    """Resolve canonical consolidated path first, falling back to legacy path if present."""
    canon = root / canonical_relpath(rel_path)
    if canon.exists():
        return canon
    legacy = root / rel_path.replace("\\", "/")
    if legacy.exists():
        return legacy
    return canon


def resolve_stage04_file(root: Path, sub_rel: str) -> Path:
    """Resolve a Stage 04 file relative to the Stage 04 root (e.g. 'retrieval/retrieval_metrics_summary.csv')."""
    return resolve_path(root, f"evaluation/artifacts/stage04/{sub_rel.replace(chr(92), '/')}")


def resolve_stage05_file(root: Path, sub_rel: str) -> Path:
    """Resolve a Stage 05 file relative to the Stage 05 root (e.g. 'figures/png/fig01_...png')."""
    return resolve_path(root, f"evaluation/artifacts/stage05/{sub_rel.replace(chr(92), '/')}")


def resolve_ac04_file(root: Path, sub_rel: str) -> Path:
    """Resolve a supplementary AC-04 file relative to the AC-04 root."""
    return resolve_path(root, f"evaluation/artifacts/ac04_completion/{sub_rel.replace(chr(92), '/')}")


def verify_relocation_manifest(root: Path) -> dict[str, Any]:
    """Read-only verification that every relocated file matches its pre-move SHA-256 hash."""
    manifest_path = root / "evaluation" / "manifests" / "relocation_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": "Missing evaluation/manifests/relocation_manifest.json"}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    checked = 0
    for entry in data.get("relocated_files", []):
        new_p = root / entry["new_path"]
        if not new_p.exists():
            missing.append(entry["new_path"])
            continue
        actual = sha256_file(new_p)
        checked += 1
        if actual != entry["sha256"]:
            mismatches.append(
                {
                    "path": entry["new_path"],
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": actual,
                }
            )
    return {
        "ok": not missing and not mismatches and checked == len(data.get("relocated_files", [])),
        "checked_count": checked,
        "missing": missing,
        "mismatches": mismatches,
    }
