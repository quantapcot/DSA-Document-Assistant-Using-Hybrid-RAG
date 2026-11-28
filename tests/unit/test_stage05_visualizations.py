"""Unit tests validating Stage 05 visualizations, manifests, data integrity, and report links."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.stage05.data_loader import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    EXPECTED_MAIN,
    STAGE04_AC_RECONCILIATION,
    verify_inputs_and_integrity,
)
from evaluation.stage05.validator import EXPECTED_DATA_CONTRACT_IDS, run_stage05_validation

ROOT = Path(__file__).resolve().parents[2]
STAGE05_DIR = ROOT / "evaluation" / "artifacts" / "stage05"


def test_stage04_inputs_and_git_baseline_intact() -> None:
    integrity = verify_inputs_and_integrity(ROOT)
    assert integrity["all_verified"] is True
    assert integrity["git"]["branch"] == EXPECTED_BRANCH
    assert integrity["git"]["expected_stage04_checkpoint"] == EXPECTED_HEAD
    assert integrity["git"]["stage04_ancestor_ok"] is True
    assert integrity["git"]["main"] == EXPECTED_MAIN
    assert integrity["stage04_file_count"] == 59
    assert integrity["missing_required_stage04_files"] == []
    assert len(integrity["local_backups"]) == 5


def test_stage04_ac_reconciliation_preserves_5_pass_1_partial() -> None:
    statuses = [row["status"] for row in STAGE04_AC_RECONCILIATION]
    assert len(STAGE04_AC_RECONCILIATION) == 6
    assert statuses.count("PASS") == 5
    assert statuses.count("PARTIAL") == 1
    ac04 = next(r for r in STAGE04_AC_RECONCILIATION if r["ac_id"] == "AC-04")
    assert ac04["status"] == "PARTIAL"
    assert ac04["canonical_stage04_title"] == "Generation, Robustness & Study Features"


def test_all_28_logical_figures_png_svg_and_manifests_valid() -> None:
    fman = json.loads((STAGE05_DIR / "figure_manifest.json").read_text(encoding="utf-8"))
    assert fman["total_logical_figures"] == 28
    assert fman["total_png_files"] == 28
    assert fman["total_svg_files"] == 28
    dc_ids = sorted(int(f["data_contract_id"]) for f in fman["figures"])
    assert dc_ids == EXPECTED_DATA_CONTRACT_IDS

    validation = run_stage05_validation(ROOT, require_report=True)
    assert validation["overall_pass"] is True
    summary = validation["summary"]
    assert summary["png_passed"] == 28
    assert summary["svg_passed"] == 28
    assert summary["data_contract_1_to_28_complete"] is True
    assert summary["groups_A_to_K_complete"] is True
    assert summary["missing_source_artifacts"] == []
    assert summary["no_rag_na_preserved"] is True
    assert summary["report_validation_ok"] is True


def test_configuration_recommendations_exploratory_and_honest() -> None:
    rec = json.loads((STAGE05_DIR / "configuration_recommendations.json").read_text(encoding="utf-8"))
    gov = rec["governance"]
    assert gov["human_validated"] is False
    assert gov["human_approval_status"] == "DRAFT"
    assert gov["production_groq_e2e_status"] == "BLOCKED"
    assert "EXPLORATORY" in gov["status"]
    scenarios = rec["operating_scenarios"]
    assert set(scenarios.keys()) == {
        "quality_oriented",
        "low_latency_lightweight",
        "alternative_chunking",
        "practical_balanced_exploratory",
    }
    assert scenarios["practical_balanced_exploratory"]["candidate_pool"] == 30
    assert scenarios["practical_balanced_exploratory"]["context_budget_tokens"] == 2048


def test_final_report_has_all_25_sections_and_no_broken_links() -> None:
    report = (ROOT / "STAGE_05_FINAL_EVALUATION_REPORT.md").read_text(encoding="utf-8")
    for idx in range(1, 26):
        assert f"\n## {idx}. " in report, f"Missing section ## {idx}."
    assert "NO_RAG" in report
    assert "AC-04" in report and "PARTIAL" in report
    assert "human_validated = false" in report
    assert "BLOCKED" in report
