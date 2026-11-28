"""CLI entrypoint to generate or read-only validate the 10 consolidated Vietnamese Evaluation figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.consolidation import CONSOLIDATED_FIGURES
from evaluation.consolidation.charts_part1 import (
    generate_fig01_danh_gia_chunking,
    generate_fig02_chat_luong_retrieval,
    generate_fig03_chat_luong_evidence,
    generate_fig04_hieu_nang_va_tai_nguyen,
    generate_fig05_chat_luong_cau_tra_loi,
)
from evaluation.consolidation.charts_part2 import (
    generate_fig06_ablation_va_do_nhay,
    generate_fig07_thong_ke_va_tradeoff,
    generate_fig08_do_tin_cay_va_phan_tich_loi,
    generate_fig09_live_groq_va_http_e2e,
    generate_fig10_chuc_nang_va_do_on_dinh_api,
)
from evaluation.consolidation.inventory_and_parity import (
    validate_consolidated_deliverables,
    write_mapping_and_parity_artifacts,
)
from evaluation.stage05.data_loader import load_stage04_bundle


def generate_all(root: Path) -> list[dict]:
    bundle = load_stage04_bundle(root)
    generators = [
        generate_fig01_danh_gia_chunking,
        generate_fig02_chat_luong_retrieval,
        generate_fig03_chat_luong_evidence,
        generate_fig04_hieu_nang_va_tai_nguyen,
        generate_fig05_chat_luong_cau_tra_loi,
        generate_fig06_ablation_va_do_nhay,
        generate_fig07_thong_ke_va_tradeoff,
        generate_fig08_do_tin_cay_va_phan_tich_loi,
        generate_fig09_live_groq_va_http_e2e,
        generate_fig10_chuc_nang_va_do_on_dinh_api,
    ]
    records: list[dict] = []
    for (num, fid, title), fn in zip(CONSOLIDATED_FIGURES, generators):
        meta = fn(bundle, root)
        records.append({"figure_number": num, "figure_id": fid, "title_vi": title, **meta})
        print(f"GENERATED [{num:02d}/10]: {fid} ({meta['png_bytes']:,} B PNG, {meta['svg_bytes']:,} B SVG)")
    write_mapping_and_parity_artifacts(root, records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or read-only validate 10 consolidated Vietnamese figures.")
    parser.add_argument("--mode", choices=["generate", "validate"], default="validate")
    parser.add_argument("--write-validation", action="store_true", help="Write consolidated_chart_validation.json")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_all(ROOT)
        res = validate_consolidated_deliverables(ROOT, write_report_json=True)
        print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
        return 0
    else:
        res = validate_consolidated_deliverables(ROOT, write_report_json=args.write_validation)
        print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
        return 0 if res["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
