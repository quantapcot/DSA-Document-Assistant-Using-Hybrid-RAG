"""Unit tests for Supplementary Stage 04 AC-04 Completion (Live Groq & HTTP E2E)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.ac04_completion.visualizations import validate_ac04_completion

ROOT = Path(__file__).resolve().parents[2]
AC04_DIR = ROOT / "evaluation" / "artifacts" / "ac04_completion"


def test_ac04_validation_6_figures_zero_secrets_and_preserved_stage04_05() -> None:
    res = validate_ac04_completion(ROOT, write_manifest=False)
    assert res["overall_pass"] is True
    s = res["summary"]
    assert s["supplementary_figures_found"] == 6
    assert s["png_passed"] == 6
    assert s["svg_passed"] == 6
    assert s["secret_hits"] == []
    assert s["stage04_stage05_integrity_ok"] is True
    assert s["stage05_28_figures_preserved"] is True
    assert s["report_ok"] is True


def test_ac04_gates_and_token_accounting_complete() -> None:
    gates_doc = json.loads((AC04_DIR / "ac04_gates_assessment.json").read_text(encoding="utf-8"))
    assert gates_doc["original_qdrant_eval_untouched"] is True
    assert gates_doc["historical_stage04_ac04_status"] == "PARTIAL"
    assert gates_doc["audit_summary"]["human_validated"] is False
    for g in ("GATE-01", "GATE-02", "GATE-03", "GATE-04", "GATE-05"):
        assert gates_doc["gates"][g]["status"] == "PASS"

    tok_df = pd.read_csv(AC04_DIR / "token_usage_summary.csv")
    tot = tok_df[tok_df["experiment_group"] == "TOTAL_ALL_LIVE_GROQ"].iloc[0]
    assert int(tot["actual_provider_calls"]) >= 80
    assert int(tot["total_tokens"]) > 50000


def test_ac04_no_rag_na_and_raw_qz03_422_preserved() -> None:
    ans_df = pd.read_csv(AC04_DIR / "answer_quality_summary.csv", keep_default_na=False)
    norag = ans_df[(ans_df["protocol"] == "direct_live_groq_no_rag") & (ans_df["subset"] == "all_40_queries")].iloc[0]
    assert norag["faithfulness_norm"] == "N/A"
    assert norag["citation_accuracy_norm"] == "N/A"
    assert norag["citation_validity"] == "N/A"
    assert norag["unsupported_claim_rate"] == "N/A"
    assert float(norag["answer_completeness_norm"]) > 0.5

    quiz_rows = [
        json.loads(line)
        for line in (AC04_DIR / "http_quiz.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_probe = next(r for r in quiz_rows if r.get("is_raw_validation_probe"))
    assert raw_probe["original_topic"] == "Tree, BST & AVL"
    assert raw_probe["http_status"] == 422


def test_ac04_edge_cases_all_9_passed() -> None:
    edge_df = pd.read_csv(AC04_DIR / "edge_cases_summary.csv")
    assert len(edge_df) == 9
    assert bool(edge_df["passed"].all()) is True
    assert int(edge_df[edge_df["case_id"] == "EDGE_SERVER_504_TIMEOUT"]["http_status"].iloc[0]) == 504
