"""Focused regression tests for Final Evaluation Consolidation & Repository Cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.consolidation import CONSOLIDATED_FIGURES, consolidated_png_dir, consolidated_svg_dir, original_png_dir, original_svg_dir
from evaluation.consolidation.inventory_and_parity import validate_consolidated_deliverables
from evaluation.path_resolver import resolve_ac04_file, resolve_path, resolve_stage04_file, verify_relocation_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_exactly_10_consolidated_figures_and_34_originals_preserved() -> None:
    c_pngs = sorted(p.name for p in consolidated_png_dir(ROOT).glob("*.png"))
    c_svgs = sorted(p.name for p in consolidated_svg_dir(ROOT).glob("*.svg"))
    expected_pngs = sorted(f"{fid}.png" for _, fid, _ in CONSOLIDATED_FIGURES)
    expected_svgs = sorted(f"{fid}.svg" for _, fid, _ in CONSOLIDATED_FIGURES)
    assert c_pngs == expected_pngs
    assert c_svgs == expected_svgs

    o_pngs = list(original_png_dir(ROOT).glob("*.png"))
    o_svgs = list(original_svg_dir(ROOT).glob("*.svg"))
    assert len(o_pngs) == 34
    assert len(o_svgs) == 34


def test_complete_panel_mapping_and_numerical_data_parity() -> None:
    fmap = json.loads((ROOT / "evaluation/figures/figure_mapping.json").read_text(encoding="utf-8"))
    assert fmap["total_original_figures"] == 34
    assert fmap["total_consolidated_figures"] == 10
    assert fmap["all_34_original_accounted"] is True

    parity = pd.read_csv(ROOT / "evaluation/tables/figure_data_parity.csv")
    assert len(parity) == 34
    assert (parity["parity_status"] == "PASS").all()
    assert int(parity["numerical_discrepancy_count"].sum()) == 0
    assert (parity["na_preserved"] == "YES").all()


def test_vietnamese_svg_rendering_and_na_and_protocol_preservation() -> None:
    # Verify Vietnamese diacritics appear intact in scalable SVG text nodes
    svg1 = (consolidated_svg_dir(ROOT) / "fig01_danh_gia_chunking.svg").read_text(encoding="utf-8")
    assert "ĐÁNH GIÁ CHIẾN LƯỢC CHUNKING" in svg1
    svg5 = (consolidated_svg_dir(ROOT) / "fig05_chat_luong_cau_tra_loi.svg").read_text(encoding="utf-8")
    assert "N/A" in svg5
    assert "NO_RAG" in svg5
    svg4 = (consolidated_svg_dir(ROOT) / "fig04_hieu_nang_va_tai_nguyen.svg").read_text(encoding="utf-8")
    assert "41,2%" in svg4
    svg10 = (consolidated_svg_dir(ROOT) / "fig10_chuc_nang_va_do_on_dinh_api.svg").read_text(encoding="utf-8")
    assert "NOT PRODUCTION-READY" in svg10
    assert "166" in svg10 and "90" in svg10


def test_relocation_manifest_and_path_resolution() -> None:
    reloc = verify_relocation_manifest(ROOT)
    assert reloc["ok"] is True
    assert reloc["checked_count"] == 168
    assert reloc["missing"] == []
    assert reloc["mismatches"] == []

    assert resolve_stage04_file(ROOT, "retrieval/retrieval_metrics_summary.csv").is_file()
    assert resolve_ac04_file(ROOT, "token_usage_summary.csv").is_file()
    assert resolve_path(ROOT, "STAGE_05_FINAL_EVALUATION_REPORT.md").name == "05_final_evaluation.md"


def test_exact_5_official_reports_and_readonly_validation() -> None:
    res = validate_consolidated_deliverables(ROOT, write_report_json=False)
    assert res["overall_pass"] is True
    s = res["summary"]
    assert s["consolidated_png_passed"] == 10
    assert s["consolidated_svg_passed"] == 10
    assert s["original_34_figures_verified"] is True
    assert s["data_parity_34_of_34_pass"] is True
    assert s["exact_5_official_reports"] is True
    assert s["report_broken_links"] == {}
    assert s["readme_broken_links"] == []
