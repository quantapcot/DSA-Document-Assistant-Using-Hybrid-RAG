"""Automated Figure, Manifest, Data-Contract & Report Link Quality Assurance."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.stage05.data_loader import verify_inputs_and_integrity

EXPECTED_DATA_CONTRACT_IDS = list(range(1, 29))
EXPECTED_GROUPS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}


def _check_png(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "error": "missing_file"}
    with Image.open(path) as img:
        width, height = img.size
        dpi = img.info.get("dpi", (0, 0))
        dpi_x = float(dpi[0]) if isinstance(dpi, (tuple, list)) and len(dpi) >= 1 else 0.0
        dpi_y = float(dpi[1]) if isinstance(dpi, (tuple, list)) and len(dpi) >= 2 else 0.0
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        std_val = float(arr.std())
    ok = width >= 1200 and height >= 600 and dpi_x >= 299.0 and dpi_y >= 299.0 and std_val > 5.0
    return {
        "ok": ok,
        "width_px": width,
        "height_px": height,
        "dpi": [round(dpi_x, 1), round(dpi_y, 1)],
        "pixel_std": round(std_val, 2),
        "bytes": path.stat().st_size,
    }


def _check_svg(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "error": "missing_file"}
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        is_svg = root.tag.endswith("svg")
        viewbox = root.attrib.get("viewBox", "")
        text_nodes = [el for el in root.iter() if el.tag.endswith("text")]
        ok = is_svg and bool(viewbox) and len(text_nodes) >= 5
        return {
            "ok": ok,
            "root_tag": root.tag,
            "viewBox": viewbox,
            "text_elements": len(text_nodes),
            "bytes": path.stat().st_size,
        }
    except ET.ParseError as exc:
        return {"ok": False, "error": f"xml_parse_error: {exc}"}


def validate_report_links(root: Path, report_path: Path, expected_fig_ids: list[str]) -> dict[str, Any]:
    """Verify all relative links in STAGE_05_FINAL_EVALUATION_REPORT.md and figure coverage."""
    if not report_path.is_file():
        return {"report_exists": False, "ok": False, "broken_links": [], "missing_figures": expected_fig_ids}

    text = report_path.read_text(encoding="utf-8")
    # Extract markdown links [label](target) excluding http/https/mailto/#anchors
    raw_links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
    broken: list[str] = []
    checked = 0
    for target in raw_links:
        t = target.strip()
        if t.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = t.split("#", 1)[0]
        if clean.startswith("file:///"):
            # Relative links required by Section 11; flag absolute file:// links inside Stage 05 report
            broken.append(f"non_relative_link:{t}")
            continue
        resolved = (report_path.parent / clean).resolve()
        checked += 1
        if not resolved.exists():
            broken.append(t)

    missing_figs = [fid for fid in expected_fig_ids if fid not in text]
    return {
        "report_exists": True,
        "relative_links_checked": checked,
        "broken_links": broken,
        "missing_figures_in_report": missing_figs,
        "ok": len(broken) == 0 and len(missing_figs) == 0,
    }


def run_stage05_validation(
    root: Path,
    require_report: bool = False,
    write_outputs: bool = False,
) -> dict[str, Any]:
    """Validate all 28 Stage 05 figures, manifests, data contract, N/A preservation & report links.

    By default ``write_outputs=False`` so validation is strictly read-only and never dirties
    committed Stage 05 manifest files after checkpointing.
    """
    from evaluation.path_resolver import resolve_path, resolve_stage05_file

    fig_manifest_path = resolve_stage05_file(root, "figure_manifest.json")
    rec_path = resolve_stage05_file(root, "configuration_recommendations.json")
    report_path = resolve_path(root, "STAGE_05_FINAL_EVALUATION_REPORT.md")

    integrity = verify_inputs_and_integrity(root)
    if not fig_manifest_path.is_file():
        raise FileNotFoundError(f"Missing figure manifest: {fig_manifest_path}")

    fman = json.loads(fig_manifest_path.read_text(encoding="utf-8"))
    figures = fman.get("figures", [])

    per_figure: list[dict[str, Any]] = []
    seen_dc_ids: list[int] = []
    seen_groups: set[str] = set()
    missing_sources: list[str] = []
    png_ok_count = 0
    svg_ok_count = 0

    for fig in figures:
        dc_id = int(fig["data_contract_id"])
        seen_dc_ids.append(dc_id)
        for g in str(fig["group"]).split(","):
            seen_groups.add(g.strip())

        png_res = _check_png(resolve_path(root, fig["png_path"]))
        svg_res = _check_svg(resolve_path(root, fig["svg_path"]))
        if png_res["ok"]:
            png_ok_count += 1
        if svg_res["ok"]:
            svg_ok_count += 1

        src_ok = True
        for src in fig.get("source_artifacts", []):
            if not resolve_path(root, src).is_file():
                src_ok = False
                missing_sources.append(src)

        per_figure.append({
            "figure_number": fig["figure_number"],
            "figure_id": fig["figure_id"],
            "data_contract_id": dc_id,
            "group": fig["group"],
            "png": png_res,
            "svg": svg_res,
            "sources_exist": src_ok,
            "has_caption_vi": bool(fig.get("caption_vi", "").strip()),
            "has_protocol": bool(fig.get("protocol", "").strip()),
            "all_ok": png_res["ok"] and svg_res["ok"] and src_ok and bool(fig.get("caption_vi")),
        })

    # Verify NO_RAG N/A preservation in table_05_generation_summary.csv
    t5_path = resolve_stage05_file(root, "tables/table_05_generation_summary.csv")
    na_preserved = False
    if t5_path.is_file():
        t5_txt = t5_path.read_text(encoding="utf-8")
        na_preserved = "NO_RAG,40,0.9688,1.0,N/A,0.9208,N/A,N/A,N/A,0.0175" in t5_txt

    report_check = validate_report_links(
        root,
        report_path,
        [f["figure_id"] for f in figures],
    )

    dc_complete = sorted(seen_dc_ids) == EXPECTED_DATA_CONTRACT_IDS
    groups_complete = EXPECTED_GROUPS.issubset(seen_groups)
    all_figs_ok = len(figures) == 28 and all(f["all_ok"] for f in per_figure)

    overall_pass = (
        integrity["all_verified"]
        and dc_complete
        and groups_complete
        and all_figs_ok
        and rec_path.is_file()
        and na_preserved
        and (report_check["ok"] if require_report else True)
    )

    result = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "summary": {
            "logical_figures_expected": 28,
            "logical_figures_found": len(figures),
            "png_passed": png_ok_count,
            "svg_passed": svg_ok_count,
            "data_contract_1_to_28_complete": dc_complete,
            "groups_A_to_K_complete": groups_complete,
            "missing_source_artifacts": sorted(set(missing_sources)),
            "no_rag_na_preserved": na_preserved,
            "input_integrity_ok": integrity["all_verified"],
            "report_validation_ok": report_check["ok"],
            "report_exists": report_check["report_exists"],
        },
        "report_check": report_check,
        "figures": per_figure,
    }

    if write_outputs:
        out_path = stage05_dir / "chart_validation.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest_path = stage05_dir / "manifest.json"
        if manifest_path.is_file():
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            mdata["validation_summary"] = result["summary"]
            manifest_path.write_text(json.dumps(mdata, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
