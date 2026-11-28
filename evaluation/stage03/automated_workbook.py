"""Stage 03 – Automated Annotation Workbook Generator (ground_truth_automated.xlsx, 6 sheets)."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.stage03.workbook import format_evidence_summary, format_kus_text, normalize_excel_text

AUTO_WORKBOOK_SHEETS = (
    "Instructions",
    "Questions",
    "Profile_A_Annotation",
    "Profile_B_Annotation",
    "Quality_Audit",
    "Annotation_Progress",
)

AUTO_ANNOTATION_HEADERS = (
    "Annotation ID",
    "Query ID",
    "Chunk ID",
    "Chunk Text",
    "Reference Answer",
    "Required Knowledge Units",
    "AI Grade",
    "Secondary Grade",
    "Final Grade",
    "Supporting Text",
    "Relevance Reason",
    "Disagreement Status",
    "Adjudication Reason",
    "Annotation Source",
    "Judge Model",
    "Review Status",
)


def _str_cell(ws, r: int, c: int, val: Any) -> Any:
    if val is None:
        return ws.cell(r, c, None)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return ws.cell(r, c, val)
    cleaned, _ = normalize_excel_text(str(val))
    cell = ws.cell(r, c, cleaned)
    cell.data_type = "s"
    return cell


def generate_automated_workbook(
    *,
    output_path: Path,
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    pool_a: list[dict[str, Any]],
    pool_b: list[dict[str, Any]],
    final_annotations_by_id: dict[str, dict[str, Any]],
    quality_audit: dict[str, Any],
) -> dict[str, Any]:
    """Generate 6-sheet audit workbook ground_truth_automated.xlsx without fabricating human labels."""
    ev_by_id = {ev["evidence_id"]: ev for ev in evidences}
    q_by_id = {q["query_id"]: q for q in questions}
    ku_text_by_qid = {qid: format_kus_text(q["required_knowledge_units"]) for qid, q in q_by_id.items()}

    wb = Workbook()
    hdr_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1E3A5F")
    body_font = Font(name="Arial", size=10, color="111827")
    bold_font = Font(name="Arial", size=10, bold=True, color="111827")
    wrap_top = Alignment(vertical="top", wrap_text=True)
    center_top = Alignment(horizontal="center", vertical="top", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    # 1. Instructions
    ws_i = wb.active
    ws_i.title = "Instructions"
    ws_i.column_dimensions["A"].width = 30
    ws_i.column_dimensions["B"].width = 112
    ws_i.merge_cells("A1:B1")
    ws_i["A1"] = "STAGE 03 – AUTOMATED LLM-AS-A-JUDGE GROUND TRUTH WORKBOOK (AUDIT & TRACEABILITY)"
    ws_i["A1"].font = Font(name="Arial", size=13, bold=True, color="FFFFFF")
    ws_i["A1"].fill = hdr_fill
    ws_i.row_dimensions[1].height = 28

    notes = [
        ("Phương pháp", "Fully automated, source-grounded LLM-as-a-Judge với Pilot Calibration, 10% Stratified Random AI Audit, Targeted Cross-Check và Evidence-Grounded Adjudication."),
        ("Trạng thái Human Validation", "human_validated = FALSE. Không có nhãn do con người gán thủ công trong workbook này; workbook ban đầu ground_truth_annotation.xlsx vẫn được bảo toàn nguyên vẹn."),
        ("Nguồn nhãn (Annotation Source)", "- AI-generated: Chấm bởi Primary Judge.\n- AI-cross-checked: Chấm độc lập bởi Primary & Secondary Judge và thống nhất.\n- AI-adjudicated: Primary & Secondary khác biệt, được phân xử dựa trên văn bản gốc."),
        ("Thang điểm Relevance", "0 = Irrelevant, 1 = Partially Relevant, 2 = Highly Relevant (hỗ trợ trực tiếp >= 1 required KU). Mọi hàng đều có Final Grade hợp lệ (0/1/2), không dùng Retrieval rank/score làm nhãn."),
        ("Mục đích Workbook", "Phục vụ truy vết, đối chiếu văn bản hỗ trợ (Supporting Text) và kiểm toán chất lượng tự động (Quality_Audit). Người dùng không cần điền thêm dữ liệu."),
    ]
    for idx, (k, v) in enumerate(notes, start=3):
        c1 = ws_i.cell(idx, 1, k)
        c1.font = bold_font
        c1.alignment = wrap_top
        c2 = ws_i.cell(idx, 2, v)
        c2.font = body_font
        c2.alignment = wrap_top
        ws_i.row_dimensions[idx].height = 42

    # 2. Questions
    ws_q = wb.create_sheet("Questions")
    ws_q.freeze_panes = "A2"
    q_hdrs = (
        "Query ID", "Question", "Topic", "Language", "Difficulty", "Question Type",
        "Reference Answer", "Required Knowledge Units", "Source References", "Source Evidence",
        "Human Approval Status", "Automated Review Status", "Automated Review Model", "Automated Review Notes",
    )
    ws_q.append(q_hdrs)
    for cell in ws_q[1]:
        cell.font = hdr_font
        cell.fill = hdr_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws_q.row_dimensions[1].height = 26
    for r_idx, q in enumerate(questions, start=2):
        vals = (
            q["query_id"], q["query"], q["topic"], q["language"], q["difficulty"], q["question_type"],
            q["reference_answer"], ku_text_by_qid[q["query_id"]], "\n".join(q["source_document_references"]),
            format_evidence_summary(q["source_evidence_ids"], ev_by_id),
            q.get("human_approval_status", "DRAFT"),
            q.get("automated_review_status", "AI_VALIDATED"),
            q.get("automated_review_model", ""),
            q.get("automated_review_reason", ""),
        )
        for c_idx, val in enumerate(vals, start=1):
            cell = _str_cell(ws_q, r_idx, c_idx, val)
            cell.font = body_font
            cell.alignment = wrap_top
    ws_q.auto_filter.ref = ws_q.dimensions
    for idx, w in enumerate((10, 44, 22, 9, 10, 22, 50, 44, 30, 50, 18, 20, 22, 42), start=1):
        ws_q.column_dimensions[get_column_letter(idx)].width = w

    # 3 & 4. Profile_A_Annotation & Profile_B_Annotation
    for sname, pool in (("Profile_A_Annotation", pool_a), ("Profile_B_Annotation", pool_b)):
        ws = wb.create_sheet(sname)
        ws.freeze_panes = "A2"
        ws.append(AUTO_ANNOTATION_HEADERS)
        for cell in ws[1]:
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 26
        for r_idx, cand in enumerate(pool, start=2):
            ann = final_annotations_by_id[cand["annotation_id"]]
            q = q_by_id[cand["query_id"]]
            vals = (
                cand["annotation_id"],
                cand["query_id"],
                cand["chunk_id"],
                cand["chunk_text"],
                q["reference_answer"],
                ku_text_by_qid[cand["query_id"]],
                ann["ai_grade"],
                ann.get("secondary_grade"),
                ann["final_grade"],
                ann.get("supporting_text_from_chunk", ""),
                ann.get("relevance_reason", ""),
                ann.get("disagreement_status", "NOT_CROSS_CHECKED"),
                ann.get("adjudication_reason", ""),
                ann["annotation_source"],
                ann["judge_model"],
                ann["annotation_status"],
            )
            for c_idx, val in enumerate(vals, start=1):
                cell = _str_cell(ws, r_idx, c_idx, val)
                cell.font = body_font
                cell.alignment = center_top if c_idx in (1, 2, 7, 8, 9, 12, 14, 16) else wrap_top
        ws.auto_filter.ref = ws.dimensions
        for idx, w in enumerate((15, 10, 25, 64, 44, 40, 10, 14, 11, 44, 44, 20, 40, 18, 22, 14), start=1):
            ws.column_dimensions[get_column_letter(idx)].width = w

    # 5. Quality_Audit
    ws_qa = wb.create_sheet("Quality_Audit")
    ws_qa.column_dimensions["A"].width = 42
    ws_qa.column_dimensions["B"].width = 24
    ws_qa.column_dimensions["C"].width = 58
    ws_qa.append(("Audit Metric", "Value", "Notes"))
    for cell in ws_qa[1]:
        cell.font = hdr_font
        cell.fill = hdr_fill

    def add_audit_row(label: str, val: Any, note: str = "") -> None:
        r = ws_qa.max_row + 1
        ws_qa.cell(r, 1, label).font = bold_font if label.startswith("[") else body_font
        _str_cell(ws_qa, r, 2, val).font = body_font
        _str_cell(ws_qa, r, 3, note).font = body_font

    add_audit_row("[Methodology]", "LLM-as-a-Judge (Automated)", "human_validated = false; AI quality audit only")
    add_audit_row("Primary Judge", quality_audit.get("primary_judge_model", ""), "")
    add_audit_row("Secondary Judge", quality_audit.get("secondary_judge_model", ""), "Independent higher-tier model (blind to primary grade)")
    add_audit_row("Total Final Annotations", quality_audit.get("total_annotations", 0), "100% completeness verified")
    add_audit_row("Supporting Text Validation Rate (%)", quality_audit.get("supporting_text_validation_rate_pct", 0.0), "Verified verbatim in chunk text for all positive grades")

    pilot = quality_audit.get("pilot_summary", {})
    add_audit_row("[Pilot Calibration]", f"{pilot.get('count', 0)} candidates", f"Rounds: {pilot.get('rounds_completed', 1)}")
    add_audit_row("Pilot Exact Agreement (%)", pilot.get("exact_agreement_pct", 0.0), "Target >= 85%")
    add_audit_row("Pilot Quadratic Weighted Kappa", pilot.get("quadratic_weighted_kappa"), "Target >= 0.75")
    add_audit_row("Pilot Severe 0-vs-2 Conflicts", pilot.get("severe_0_vs_2_conflicts", 0), "Target = 0 unexplained")

    rand_a = quality_audit.get("random_audit_10pct", {})
    add_audit_row("[10% Stratified Random AI Audit]", f"{rand_a.get('count', 0)} candidates", "Strictly separated from targeted cross-check")
    add_audit_row("Random Audit Exact Agreement (%)", rand_a.get("exact_agreement_pct", 0.0), "")
    add_audit_row("Random Audit Quadratic Weighted Kappa", rand_a.get("quadratic_weighted_kappa"), "")
    add_audit_row("Random Audit Linear Weighted Kappa", rand_a.get("linear_weighted_kappa"), "")
    add_audit_row("Random Audit Disagreement Count", rand_a.get("disagreement_count", 0), f"Rate: {rand_a.get('disagreement_rate_pct', 0.0)}%")
    add_audit_row("Random Audit Severe 0-vs-2 Conflicts", rand_a.get("severe_0_vs_2_conflicts", 0), "")
    add_audit_row("Random Audit Confusion Matrix (0/1/2)", str(rand_a.get("confusion_matrix", [])), "Rows=Primary, Cols=Secondary")

    tgt = quality_audit.get("targeted_cross_check", {})
    add_audit_row("[Targeted Cross-Check]", f"{tgt.get('count', 0)} candidates", "Low-confidence, ambiguous code/formula/short/composite, cross-profile")
    add_audit_row("Targeted Exact Agreement (%)", tgt.get("exact_agreement_pct", 0.0), "")
    add_audit_row("Targeted Disagreement Count", tgt.get("disagreement_count", 0), "")

    adj = quality_audit.get("adjudication_summary", {})
    add_audit_row("[Conflict Adjudication]", adj.get("total_adjudicated", 0), "Evidence-grounded resolution of all disagreements")
    add_audit_row("Grade Corrections vs Primary", adj.get("changed_from_primary_count", 0), f"Correction rate: {adj.get('grade_correction_rate_pct', 0.0)}%")
    add_audit_row("Cross-Profile Identical-Text Inconsistencies", quality_audit.get("cross_profile_inconsistency_count", 0), "Post-adjudication check on identical chunk text within same query")

    # 6. Annotation_Progress
    ws_p = wb.create_sheet("Annotation_Progress")
    ws_p.freeze_panes = "A9"
    ws_p.append(("Scope", "Total", "Completed", "Remaining", "Grade 0", "Grade 1", "Grade 2", "Completion %"))
    for cell in ws_p[1]:
        cell.font = hdr_font
        cell.fill = hdr_fill

    end_a = len(pool_a) + 1
    end_b = len(pool_b) + 1
    for r_idx, (label, sname, end_r) in enumerate(
        (
            ("Profile A (recursive_512_100)", "Profile_A_Annotation", end_a),
            ("Profile B (structure_adaptive)", "Profile_B_Annotation", end_b),
        ),
        start=2,
    ):
        row_vals = (
            label,
            f"=COUNTA({sname}!A2:A{end_r})",
            f"=E{r_idx}+F{r_idx}+G{r_idx}",
            f"=B{r_idx}-C{r_idx}",
            f"=COUNTIF({sname}!I2:I{end_r},0)",
            f"=COUNTIF({sname}!I2:I{end_r},1)",
            f"=COUNTIF({sname}!I2:I{end_r},2)",
            f"=IF(B{r_idx}=0,0,C{r_idx}/B{r_idx})",
        )
        for col, v in enumerate(row_vals, start=1):
            c = ws_p.cell(r_idx, col, v)
            c.font = body_font
            c.border = thin_border
            if col == 8:
                c.number_format = "0.0%"
    tot_vals = ("TOTAL", "=B2+B3", "=C2+C3", "=D2+D3", "=E2+E3", "=F2+F3", "=G2+G3", "=IF(B4=0,0,C4/B4)")
    for col, v in enumerate(tot_vals, start=1):
        c = ws_p.cell(4, col, v)
        c.font = bold_font
        c.border = thin_border
        if col == 8:
            c.number_format = "0.0%"

    pq_hdr = ("Query ID", "A Total", "A G0", "A G1", "A G2", "B Total", "B G0", "B G1", "B G2")
    for col, h in enumerate(pq_hdr, start=1):
        c = ws_p.cell(8, col, h)
        c.font = hdr_font
        c.fill = hdr_fill
    for idx, q in enumerate(questions):
        r = 9 + idx
        qid = q["query_id"]
        vals = (
            qid,
            f"=COUNTIF(Profile_A_Annotation!$B$2:$B${end_a},A{r})",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$I$2:$I${end_a},0)",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$I$2:$I${end_a},1)",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$I$2:$I${end_a},2)",
            f"=COUNTIF(Profile_B_Annotation!$B$2:$B${end_b},A{r})",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$I$2:$I${end_b},0)",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$I$2:$I${end_b},1)",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$I$2:$I${end_b},2)",
        )
        for col, v in enumerate(vals, start=1):
            c = ws_p.cell(r, col, v)
            c.font = body_font
            c.border = thin_border

    for idx, w in enumerate((30, 12, 12, 12, 10, 10, 10, 14, 10), start=1):
        ws_p.column_dimensions[get_column_letter(idx)].width = w

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent, prefix=".stage03-auto-xlsx-") as tmp:
        staged = Path(tmp) / output_path.name
        wb.save(staged)
        wb.close()
        if not is_zipfile(staged):
            raise ValueError("Generated automated workbook is not a valid ZIP")
        with ZipFile(staged) as zf:
            if bad := zf.testzip():
                raise ValueError(f"Corrupt ZIP entry in automated workbook: {bad}")
        check_wb = load_workbook(staged, read_only=True)
        try:
            assert tuple(check_wb.sheetnames) == AUTO_WORKBOOK_SHEETS
            assert check_wb["Profile_A_Annotation"].max_row - 1 == len(pool_a)
            assert check_wb["Profile_B_Annotation"].max_row - 1 == len(pool_b)
        finally:
            check_wb.close()
        staged.replace(output_path)

    return {
        "output_path": str(output_path),
        "file_bytes": output_path.stat().st_size,
        "sheets": list(AUTO_WORKBOOK_SHEETS),
        "profile_a_rows": len(pool_a),
        "profile_b_rows": len(pool_b),
        "zip_integrity": "PASSED",
    }
