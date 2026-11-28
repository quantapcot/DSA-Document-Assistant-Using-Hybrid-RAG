"""Reproducible CLI entrypoint for Stage 05 visualizations, tables, manifests, and QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from evaluation.stage05.charts_analysis import generate_analysis_figures
from evaluation.stage05.charts_chunking import generate_chunking_figures
from evaluation.stage05.charts_generation_ablation import (
    generate_generation_ablation_sensitivity_figures,
)
from evaluation.stage05.charts_performance import generate_performance_figures
from evaluation.stage05.charts_retrieval_evidence import (
    generate_retrieval_and_evidence_figures,
)
from evaluation.stage05.data_loader import load_stage04_bundle
from evaluation.stage05.style import configure_academic_style
from evaluation.stage05.tables_and_manifests import (
    generate_stage05_tables,
    write_stage05_manifests,
)
from evaluation.stage05.validator import run_stage05_validation


def run_pipeline(mode: str = "all", require_report: bool = False) -> int:
    stage05_dir = ROOT / "evaluation" / "artifacts" / "stage05"
    png_dir = stage05_dir / "figures" / "png"
    svg_dir = stage05_dir / "figures" / "svg"
    tables_dir = stage05_dir / "tables"

    if mode == "validate":
        res = run_stage05_validation(ROOT, require_report=require_report, write_outputs=False)
        print(json.dumps(res["summary"], indent=2))
        if not res["overall_pass"]:
            if res["report_check"].get("broken_links"):
                print("BROKEN_LINKS:", res["report_check"]["broken_links"])
            if res["report_check"].get("missing_figures_in_report"):
                print("MISSING_FIGS_IN_REPORT:", res["report_check"]["missing_figures_in_report"])
            return 1
        return 0

    style_meta = configure_academic_style()
    bundle = load_stage04_bundle(ROOT)

    figure_records = []
    if mode in ("all", "figures"):
        figure_records.extend(generate_chunking_figures(bundle, png_dir, svg_dir))
        figure_records.extend(generate_retrieval_and_evidence_figures(bundle, png_dir, svg_dir))
        figure_records.extend(generate_performance_figures(bundle, png_dir, svg_dir))
        figure_records.extend(generate_generation_ablation_sensitivity_figures(bundle, png_dir, svg_dir))
        figure_records.extend(generate_analysis_figures(bundle, png_dir, svg_dir))
        figure_records.sort(key=lambda r: int(r["figure_number"]))

    table_records = []
    if mode in ("all", "tables"):
        table_records = generate_stage05_tables(bundle, tables_dir)

    if mode == "all":
        write_stage05_manifests(bundle, stage05_dir, figure_records, table_records, style_meta)
        val = run_stage05_validation(ROOT, require_report=require_report, write_outputs=True)
        write_stage05_manifests(bundle, stage05_dir, figure_records, table_records, style_meta, validation_report=val)
        print(f"GENERATED_LOGICAL_FIGURES: {len(figure_records)}")
        print(f"GENERATED_TABLES: {len(table_records)}")
        print("VALIDATION_SUMMARY:", json.dumps(val["summary"], indent=2))
        return 0 if val["overall_pass"] else 1

    print(f"MODE={mode} completed (figures={len(figure_records)}, tables={len(table_records)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "figures", "tables", "validate"),
        default="all",
        help="Execution mode (default: all)",
    )
    parser.add_argument(
        "--require-report",
        action="store_true",
        help="Fail validation if STAGE_05_FINAL_EVALUATION_REPORT.md is missing or has broken/missing figure links.",
    )
    args = parser.parse_args(argv)
    return run_pipeline(mode=args.mode, require_report=args.require_report)


if __name__ == "__main__":
    raise SystemExit(main())
