"""Stage 03 – Five-Sheet Blind-Method Human Annotation XLSX Workbook Generator & Verifier."""
from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

REQUIRED_SHEETS = (
    "Instructions",
    "Questions",
    "Profile_A_Annotation",
    "Profile_B_Annotation",
    "Annotation_Progress",
)

QUESTIONS_HEADERS = (
    "Query ID",
    "Question",
    "Topic",
    "Language",
    "Difficulty",
    "Question Type",
    "Reference Answer",
    "Required Knowledge Units",
    "Source References",
    "Source Evidence",
    "Human Approval Status",
    "Reviewer Notes",
)

ANNOTATION_HEADERS = (
    "Annotation ID",
    "Query ID",
    "Question",
    "Reference Answer",
    "Required Knowledge Units",
    "Chunk ID",
    "Chunk Text",
    "Source Document",
    "Source Page/Location",
    "Relevance Grade",
    "Reviewer ID",
    "Reviewer Notes",
)

# Forbidden headers/metadata that must NEVER appear in annotation sheets (blind-method guarantee)
FORBIDDEN_ANNOTATION_COLUMNS = (
    "retrieval method",
    "method",
    "rank",
    "retrieval rank",
    "score",
    "retrieval score",
    "candidate frequency",
    "candidate_origin",
    "origin",
)

# XML 1.0 illegal control chars that openpyxl rejects
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_CELL_LIMIT = 32767


def normalize_excel_text(value: str) -> tuple[str, int]:
    """Ensure *value* is safe for Excel XML without truncating or altering meaningful characters."""
    if len(value) > EXCEL_CELL_LIMIT:
        raise ValueError(f"Cell text length {len(value)} exceeds Excel 32,767-char limit.")
    cleaned, n_sub = _ILLEGAL_XML_RE.subn("", value)
    return cleaned, n_sub


def format_kus_text(kus: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{ku['ku_id']} ({ku.get('evidence_mode', 'single')})] {ku['description']}" for ku in kus)


def format_evidence_summary(ev_ids: list[str], ev_by_id: dict[str, dict[str, Any]]) -> str:
    parts = []
    for eid in ev_ids:
        ev = ev_by_id[eid]
        sp0 = ev["source_spans"][0]
        sp_end = ev["source_spans"][-1]
        snippet = ev["evidence_text"].replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        parts.append(
            f"{eid} ({ev['knowledge_unit_id']}): {ev['source_path']} p.{ev['canonical_page']} "
            f"[{sp0['char_start']}:{sp_end['char_end']}] — \"{snippet}\""
        )
    return "\n".join(parts)


def _set_str_cell(sheet, row: int, col: int, value: str | None) -> Any:
    if value is None:
        return sheet.cell(row, col, None)
    cleaned, _ = normalize_excel_text(value)
    cell = sheet.cell(row, col, cleaned)
    # Keep source strings literal even if starting with '=', '+', '-', '@'
    cell.data_type = "s"
    return cell


def _build_instructions_sheet(ws) -> None:
    ws.title = "Instructions"
    ws.views.sheetView[0].showGridLines = True
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 115

    title_font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
    title_fill = PatternFill("solid", fgColor="1E3A5F")
    sec_font = Font(name="Arial", size=11, bold=True, color="1E3A5F")
    sec_fill = PatternFill("solid", fgColor="E2E8F0")
    body_font = Font(name="Arial", size=10, color="111827")
    wrap_top = Alignment(vertical="top", wrap_text=True)

    ws.merge_cells("A1:B1")
    ws["A1"] = "HƯỚNG DẪN GÁN NHÃN GROUND TRUTH – STAGE 03 (DSA DOCUMENT ASSISTANT USING HYBRID RAG)"
    ws["A1"].font = title_font
    ws["A1"].fill = title_fill
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[1].height = 30

    sections = [
        ("1. Mục tiêu Evaluation", (
            "Đánh giá so sánh công bằng 8 tổ hợp thực nghiệm gồm 2 chiến lược Chunking (Profile A: Recursive 512/100 và "
            "Profile B: Structure-aware Adaptive) kết hợp 4 phương pháp Truy xuất (BM25, Dense, Hybrid RRF, Hybrid Rerank) "
            "trên DUY NHẤT 1 bộ 40 câu hỏi chung (Q001–Q040) tại các ngưỡng K = 1, 3, 5, 10, 15, 20, 30, 45, 60."
        )),
        ("2. Hai Chunking Profiles", (
            "- Profile_A_Annotation (recursive_512_100): Các đoạn cắt cố định tối đa 512 ký tự, overlap 100 ký tự.\n"
            "- Profile_B_Annotation (structure_adaptive): Các đoạn cắt theo cấu trúc tài liệu DSA (định nghĩa, bước thuật toán, code, công thức, bảng), giới hạn cứng <= 128 MiniLM tokens.\n"
            "- Lưu ý: Workbook sử dụng cơ chế BLIND-METHOD ANNOTATION (ẩn hoàn toàn Retrieval Method, Rank, Score và nguồn gốc candidate; thứ tự các dòng trong từng câu hỏi đã được xáo trộn ngẫu nhiên cố định). Người gán nhãn vẫn phân biệt được Profile A/B do tách thành 2 sheet riêng."
        )),
        ("3. Bước 1 – Review 40 câu hỏi (Sheet Questions)", (
            "1) Mở sheet 'Questions'. Kiểm tra từng câu hỏi Q001–Q040, Reference Answer, Required Knowledge Units và Source Evidence.\n"
            "2) Tại cột 'Human Approval Status' (ban đầu là DRAFT), chọn:\n"
            "   - APPROVED: Câu hỏi, đáp án chuẩn và nguồn chứng minh chính xác, sẵn sàng đưa vào Ground Truth chính thức.\n"
            "   - NEEDS_REVISION: Cần sửa câu hỏi/đáp án/evidence (ghi rõ yêu cầu vào cột 'Reviewer Notes').\n"
            "3) KHÔNG sửa trực tiếp nội dung câu hỏi trong Excel để tránh lệch với pool đã sinh; nếu cần sửa, đánh dấu NEEDS_REVISION để hệ thống cập nhật và tái sinh riêng pool của câu đó."
        )),
        ("4. Bước 2 – Chấm điểm Relevance Grade (0 / 1 / 2)", (
            "Đánh giá mức độ hữu ích thực sự của Chunk Text trong việc cung cấp bằng chứng trả lời câu hỏi dựa trên Required Knowledge Units:\n"
            "- Grade 0 (Irrelevant): Không cung cấp bằng chứng giúp trả lời câu hỏi; chỉ trùng từ khóa nhưng sai chủ đề hoặc quá chung chung.\n"
            "- Grade 1 (Partially Relevant): Cung cấp một phần ngữ cảnh hoặc hỗ trợ gián tiếp/chưa đầy đủ cho một Knowledge Unit.\n"
            "- Grade 2 (Highly Relevant): Cung cấp bằng chứng trực tiếp, chính xác, rõ ràng cho ít nhất một Required Knowledge Unit (không bắt buộc 1 chunk chứa toàn bộ đáp án đối với câu hỏi nhiều KUs).\n"
            "LƯU Ý CÔNG BẰNG: Không thiên vị chunk dài chỉ vì nhiều chữ, không thiên vị chunk ngắn chỉ vì gọn; chỉ đánh giá theo bằng chứng thực tế."
        )),
        ("5. Phân biệt Grade 0 vs Ô trống (UNJUDGED)", (
            "- Ô TRỐNG = UNJUDGED (chưa chấm). KHÔNG ĐƯỢC hiểu ô trống là Grade 0.\n"
            "- Grade 0 = ĐÃ CHẤM là không liên quan (Irrelevant). Trên sheet 'Annotation_Progress', ô điền số 0 được tính vào 'Judged', còn ô trống được tính vào 'Unjudged'.\n"
            "- Hệ thống importer yêu cầu 100% các dòng phải được điền nhãn 0, 1 hoặc 2 trước khi xuất Ground Truth chính thức."
        )),
        ("6. Sử dụng Bộ lọc (Filters) & Reviewer Notes", (
            "- Cả 3 sheet dữ liệu đã bật Auto-Filter và Freeze Panes dòng tiêu đề.\n"
            "- Nên lọc theo từng 'Query ID' (Q001 → Q040) để đọc Reference Answer và Required Knowledge Units một lần rồi chấm liên tiếp toàn bộ candidates của câu đó.\n"
            "- Có thể lọc cột 'Relevance Grade' = (Blanks) để tìm nhanh các dòng chưa chấm.\n"
            "- Cột 'Reviewer ID' (tùy chọn nếu nhiều người cùng chấm) và 'Reviewer Notes' dùng để ghi chú trường hợp biên hoặc nghi ngờ lỗi trích xuất.\n"
            "- KHÔNG chỉnh sửa các cột cố định (Annotation ID, Query ID, Question, Reference Answer, Chunk ID, Chunk Text, Source Document, Source Page/Location) — importer sẽ tự động đối chiếu và từ chối nếu các trường này bị thay đổi."
        )),
        ("7. Kiểm tra Tiến độ & Hoàn thành", (
            "- Theo dõi sheet 'Annotation_Progress' để xem tổng số đã chấm, chưa chấm, phân bố Grade 0/1/2 và % hoàn thành của từng Profile và từng câu Q001–Q040.\n"
            "- Khi 40/40 câu ở sheet 'Questions' đều là APPROVED và cả hai sheet Annotation đạt 100.0% Completion (0 Unjudged), lưu file và chạy lệnh importer để tạo ground_truth_recursive.jsonl và ground_truth_adaptive.jsonl."
        )),
    ]

    row = 3
    for heading, body in sections:
        c_h = ws.cell(row, 1, heading)
        c_h.font = sec_font
        c_h.fill = sec_fill
        c_h.alignment = wrap_top
        c_b = ws.cell(row, 2, body)
        c_b.font = body_font
        c_b.alignment = wrap_top
        lines = sum(max(1, len(ln) // 95 + 1) for ln in body.split("\n"))
        ws.row_dimensions[row].height = min(409, max(36, 15 * lines + 10))
        row += 1

    ws.freeze_panes = "A2"


def _build_questions_sheet(ws, questions: list[dict[str, Any]], ev_by_id: dict[str, dict[str, Any]]) -> None:
    ws.title = "Questions"
    ws.freeze_panes = "A2"
    ws.append(QUESTIONS_HEADERS)

    widths = (11, 46, 24, 10, 11, 22, 56, 48, 34, 58, 22, 30)
    for idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w

    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1E3A5F")
    body_font = Font(name="Arial", size=10, color="111827")
    edit_fill = PatternFill("solid", fgColor="FFFBEB")
    wrap_top = Alignment(vertical="top", wrap_text=True)

    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 26

    for r_idx, q in enumerate(questions, start=2):
        ku_str = format_kus_text(q["required_knowledge_units"])
        src_refs = "\n".join(q["source_document_references"])
        ev_str = format_evidence_summary(q["source_evidence_ids"], ev_by_id)
        vals = (
            q["query_id"],
            q["query"],
            q["topic"],
            q["language"],
            q["difficulty"],
            q["question_type"],
            q["reference_answer"],
            ku_str,
            src_refs,
            ev_str,
            q["human_approval_status"],
            None,
        )
        for c_idx, val in enumerate(vals, start=1):
            cell = _set_str_cell(ws, r_idx, c_idx, val)
            cell.font = body_font
            cell.alignment = wrap_top
            if c_idx in (11, 12):
                cell.fill = edit_fill

        max_lines = max(
            sum(max(1, len(ln) // 45 + 1) for ln in q["reference_answer"].split("\n")),
            sum(max(1, len(ln) // 42 + 1) for ln in ku_str.split("\n")),
            sum(max(1, len(ln) // 52 + 1) for ln in ev_str.split("\n")),
        )
        ws.row_dimensions[r_idx].height = min(409, max(45, 14 * max_lines + 8))

    ws.auto_filter.ref = ws.dimensions
    dv = DataValidation(
        type="list",
        formula1='"DRAFT,APPROVED,NEEDS_REVISION"',
        allow_blank=False,
        showErrorMessage=True,
        errorStyle="stop",
        errorTitle="Invalid status",
        error="Choose DRAFT, APPROVED, or NEEDS_REVISION.",
    )
    ws.add_data_validation(dv)
    dv.add(f"K2:K{ws.max_row}")


def _build_annotation_sheet(
    ws,
    sheet_title: str,
    pool_rows: list[dict[str, Any]],
    q_by_id: dict[str, dict[str, Any]],
) -> int:
    """Populate Profile_A_Annotation or Profile_B_Annotation (blind-method: zero method/rank/score)."""
    ws.title = sheet_title
    ws.freeze_panes = "A2"
    ws.append(ANNOTATION_HEADERS)

    widths = (16, 10, 40, 48, 44, 26, 78, 32, 26, 16, 16, 30)
    for idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w

    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="203653")
    body_font = Font(name="Arial", size=10, color="111827")
    grade_fill = PatternFill("solid", fgColor="FFFBEB")
    wrap_top = Alignment(vertical="top", wrap_text=True)
    center_top = Alignment(horizontal="center", vertical="top", wrap_text=True)

    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 26

    # Pre-format KUs per query once for speed
    ku_text_by_qid = {qid: format_kus_text(q["required_knowledge_units"]) for qid, q in q_by_id.items()}
    sanitized_control_chars = 0

    for r_idx, item in enumerate(pool_rows, start=2):
        q = q_by_id[item["query_id"]]
        cleaned_chunk, n_sub = normalize_excel_text(item["chunk_text"])
        sanitized_control_chars += n_sub

        vals = (
            item["annotation_id"],
            item["query_id"],
            q["query"],
            q["reference_answer"],
            ku_text_by_qid[item["query_id"]],
            item["chunk_id"],
            cleaned_chunk,
            item["source_path"],
            item["source_location"],
            None,  # Relevance Grade ALWAYS blank initially (UNJUDGED)
            None,  # Reviewer ID
            None,  # Reviewer Notes
        )
        for c_idx, val in enumerate(vals, start=1):
            cell = _set_str_cell(ws, r_idx, c_idx, val)
            cell.font = body_font
            cell.alignment = center_top if c_idx in (1, 2, 10, 11) else wrap_top
            if c_idx in (10, 11, 12):
                cell.fill = grade_fill

        chunk_lines = sum(max(1, len(ln) // 72 + 1) for ln in cleaned_chunk.split("\n"))
        ws.row_dimensions[r_idx].height = min(409, max(32, 14 * min(chunk_lines, 24) + 6))

    ws.auto_filter.ref = ws.dimensions
    grade_range = f"J2:J{ws.max_row}"
    dv = DataValidation(
        type="list",
        formula1='"0,1,2"',
        allow_blank=True,
        showErrorMessage=True,
        errorStyle="stop",
        errorTitle="Invalid relevance grade",
        error="Chỉ được nhập 0 (Irrelevant), 1 (Partially Relevant), 2 (Highly Relevant), hoặc để trống (Unjudged).",
    )
    ws.add_data_validation(dv)
    dv.add(grade_range)

    # Conditional formatting: blank (Unjudged) vs 0 / 1 / 2 (Judged)
    ws.conditional_formatting.add(
        grade_range,
        FormulaRule(formula=['ISBLANK(J2)'], fill=PatternFill("solid", fgColor="FFFBEB")),
    )
    ws.conditional_formatting.add(
        grade_range,
        CellIsRule(operator="equal", formula=["0"], fill=PatternFill("solid", fgColor="E5E7EB")),
    )
    ws.conditional_formatting.add(
        grade_range,
        CellIsRule(operator="equal", formula=["1"], fill=PatternFill("solid", fgColor="DBEAFE")),
    )
    ws.conditional_formatting.add(
        grade_range,
        CellIsRule(operator="equal", formula=["2"], fill=PatternFill("solid", fgColor="DCFCE7")),
    )
    return sanitized_control_chars


def _build_progress_sheet(ws, questions: list[dict[str, Any]], count_a: int, count_b: int) -> None:
    ws.title = "Annotation_Progress"
    ws.freeze_panes = "A10"

    end_a = count_a + 1
    end_b = count_b + 1
    end_q = len(questions) + 1

    title_font = Font(name="Arial", size=12, bold=True, color="FFFFFF")
    title_fill = PatternFill("solid", fgColor="1E3A5F")
    hdr_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="203653")
    bold_font = Font(name="Arial", size=10, bold=True, color="111827")
    body_font = Font(name="Arial", size=10, color="111827")
    total_fill = PatternFill("solid", fgColor="E2E8F0")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    ws.merge_cells("A1:I1")
    ws["A1"] = "STAGE 03 – ANNOTATION PROGRESS TRACKER (0 = JUDGED, BLANK = UNJUDGED)"
    ws["A1"].font = title_font
    ws["A1"].fill = title_fill
    ws.row_dimensions[1].height = 26

    # Overall summary table (Rows 3-7)
    sum_headers = (
        "Scope",
        "Total Candidates",
        "Judged (0/1/2)",
        "Unjudged (Blank)",
        "Grade 0",
        "Grade 1",
        "Grade 2",
        "Completion %",
        "Questions Approved",
    )
    for col, h in enumerate(sum_headers, start=1):
        c = ws.cell(3, col, h)
        c.font = hdr_font
        c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[3].height = 24

    summary_rows = [
        (
            "Profile A (recursive_512_100)",
            f"=COUNTA(Profile_A_Annotation!A2:A{end_a})",
            "=E4+F4+G4",
            "=B4-C4",
            f"=COUNTIF(Profile_A_Annotation!J2:J{end_a},0)",
            f"=COUNTIF(Profile_A_Annotation!J2:J{end_a},1)",
            f"=COUNTIF(Profile_A_Annotation!J2:J{end_a},2)",
            "=IF(B4=0,0,C4/B4)",
            f'=COUNTIF(Questions!K2:K{end_q},"APPROVED")',
        ),
        (
            "Profile B (structure_adaptive)",
            f"=COUNTA(Profile_B_Annotation!A2:A{end_b})",
            "=E5+F5+G5",
            "=B5-C5",
            f"=COUNTIF(Profile_B_Annotation!J2:J{end_b},0)",
            f"=COUNTIF(Profile_B_Annotation!J2:J{end_b},1)",
            f"=COUNTIF(Profile_B_Annotation!J2:J{end_b},2)",
            "=IF(B5=0,0,C5/B5)",
            f'=COUNTIF(Questions!K2:K{end_q},"APPROVED")',
        ),
        (
            "TOTAL WORKBOOK",
            "=B4+B5",
            "=C4+C5",
            "=D4+D5",
            "=E4+E5",
            "=F4+F5",
            "=G4+G5",
            "=IF(B6=0,0,C6/B6)",
            f'=COUNTIF(Questions!K2:K{end_q},"APPROVED")&"/"&COUNTA(Questions!A2:A{end_q})',
        ),
    ]
    for r_off, row_vals in enumerate(summary_rows, start=4):
        is_tot = r_off == 6
        for col, val in enumerate(row_vals, start=1):
            c = ws.cell(r_off, col, val)
            c.font = bold_font if (is_tot or col == 1) else body_font
            if is_tot:
                c.fill = total_fill
            c.border = thin_border
            if col == 8:
                c.number_format = "0.0%"
            c.alignment = Alignment(horizontal="left" if col == 1 else "center", vertical="center")

    # Per-question progress table (Rows 9..(9+len(questions)))
    pq_headers = (
        "Query ID",
        "Approval Status",
        "A Total",
        "A Judged",
        "A Unjudged",
        "A G0",
        "A G1",
        "A G2",
        "A %",
        "B Total",
        "B Judged",
        "B Unjudged",
        "B G0",
        "B G1",
        "B G2",
        "B %",
        "Total %",
    )
    for col, h in enumerate(pq_headers, start=1):
        c = ws.cell(9, col, h)
        c.font = hdr_font
        c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[9].height = 24

    for idx, q in enumerate(questions):
        r = 10 + idx
        q_row = 2 + idx
        qid = q["query_id"]
        vals = (
            qid,
            f"=Questions!K{q_row}",
            f"=COUNTIF(Profile_A_Annotation!$B$2:$B${end_a},A{r})",
            f"=F{r}+G{r}+H{r}",
            f"=C{r}-D{r}",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$J$2:$J${end_a},0)",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$J$2:$J${end_a},1)",
            f"=COUNTIFS(Profile_A_Annotation!$B$2:$B${end_a},A{r},Profile_A_Annotation!$J$2:$J${end_a},2)",
            f"=IF(C{r}=0,0,D{r}/C{r})",
            f"=COUNTIF(Profile_B_Annotation!$B$2:$B${end_b},A{r})",
            f"=M{r}+N{r}+O{r}",
            f"=J{r}-K{r}",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$J$2:$J${end_b},0)",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$J$2:$J${end_b},1)",
            f"=COUNTIFS(Profile_B_Annotation!$B$2:$B${end_b},A{r},Profile_B_Annotation!$J$2:$J${end_b},2)",
            f"=IF(J{r}=0,0,K{r}/J{r})",
            f"=IF((C{r}+J{r})=0,0,(D{r}+K{r})/(C{r}+J{r}))",
        )
        for col, val in enumerate(vals, start=1):
            c = ws.cell(r, col, val)
            c.font = bold_font if col == 1 else body_font
            c.border = thin_border
            c.alignment = Alignment(horizontal="center", vertical="center")
            if col in (9, 16, 17):
                c.number_format = "0.0%"

    col_widths = (28, 16, 14, 14, 14, 10, 10, 10, 13, 12, 12, 12, 10, 10, 10, 11, 11)
    for col, w in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(col)].width = w


def verify_annotation_workbook(
    path: Path,
    *,
    expected_question_count: int,
    expected_pool_a_count: int,
    expected_pool_b_count: int,
) -> dict[str, Any]:
    """Verify ZIP integrity, 5 required sheets, row counts, blank grades, blind headers, and formulas."""
    if not is_zipfile(path):
        raise ValueError(f"Workbook is not a valid XLSX ZIP file: {path}")
    try:
        with ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP entry in workbook: {bad}")
    except BadZipFile as exc:
        raise ValueError("Corrupt ZIP directory in workbook") from exc

    wb = load_workbook(path, data_only=False)
    try:
        if tuple(wb.sheetnames) != REQUIRED_SHEETS:
            raise ValueError(f"Sheet names mismatch: {wb.sheetnames} != {list(REQUIRED_SHEETS)}")

        ws_q = wb["Questions"]
        q_header = tuple(ws_q.cell(1, c).value for c in range(1, len(QUESTIONS_HEADERS) + 1))
        if q_header != QUESTIONS_HEADERS:
            raise ValueError("Questions sheet header mismatch")
        q_rows = ws_q.max_row - 1
        if q_rows != expected_question_count:
            raise ValueError(f"Questions row count mismatch: {q_rows} != {expected_question_count}")
        statuses = [ws_q.cell(r, 11).value for r in range(2, ws_q.max_row + 1)]
        if any(s != "DRAFT" for s in statuses):
            raise ValueError("All initial question statuses must be DRAFT")

        sheet_stats = {}
        for sname, exp_cnt in (
            ("Profile_A_Annotation", expected_pool_a_count),
            ("Profile_B_Annotation", expected_pool_b_count),
        ):
            ws = wb[sname]
            hdr = tuple(ws.cell(1, c).value for c in range(1, len(ANNOTATION_HEADERS) + 1))
            if hdr != ANNOTATION_HEADERS:
                raise ValueError(f"{sname}: header mismatch")
            lower_hdrs = {str(h).strip().lower() for h in hdr}
            for forbidden in FORBIDDEN_ANNOTATION_COLUMNS:
                if forbidden in lower_hdrs:
                    raise ValueError(f"{sname}: blind-method violation, forbidden column {forbidden!r}")
            n_rows = ws.max_row - 1
            if n_rows != exp_cnt:
                raise ValueError(f"{sname}: row count {n_rows} != expected {exp_cnt}")
            non_blank_grades = sum(
                1 for r in range(2, ws.max_row + 1) if ws.cell(r, 10).value is not None
            )
            if non_blank_grades != 0:
                raise ValueError(f"{sname}: found {non_blank_grades} non-blank initial relevance grades!")
            if not ws.data_validations.dataValidation:
                raise ValueError(f"{sname}: missing relevance grade data validation")
            sheet_stats[sname] = {
                "rows": n_rows,
                "blank_grades": n_rows - non_blank_grades,
                "non_blank_grades": non_blank_grades,
                "data_validations": len(ws.data_validations.dataValidation),
            }

        ws_p = wb["Annotation_Progress"]
        if not str(ws_p["C4"].value).startswith("=") or not str(ws_p["E4"].value).startswith("=COUNTIF"):
            raise ValueError("Annotation_Progress summary formulas missing")
        if not str(ws_p["F10"].value).startswith("=COUNTIFS"):
            raise ValueError("Annotation_Progress per-question COUNTIFS formulas missing")
    finally:
        wb.close()

    return {
        "zip_integrity": "PASSED",
        "sheets": list(REQUIRED_SHEETS),
        "question_rows": q_rows,
        "all_initial_statuses_draft": True,
        "all_initial_grades_blank": True,
        "blind_method_verified": True,
        **sheet_stats,
    }


def generate_annotation_workbook(
    *,
    output_path: Path,
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    pool_recursive: list[dict[str, Any]],
    pool_adaptive: list[dict[str, Any]],
    overwrite: bool = True,
) -> dict[str, Any]:
    """Generate and atomically verify the 5-sheet Stage 03 human annotation XLSX workbook."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Workbook already exists: {output_path}")

    ev_by_id = {ev["evidence_id"]: ev for ev in evidences}
    q_by_id = {q["query_id"]: q for q in questions}

    wb = Workbook()
    _build_instructions_sheet(wb.active)
    _build_questions_sheet(wb.create_sheet("Questions"), questions, ev_by_id)
    san_a = _build_annotation_sheet(wb.create_sheet("Profile_A_Annotation"), "Profile_A_Annotation", pool_recursive, q_by_id)
    san_b = _build_annotation_sheet(wb.create_sheet("Profile_B_Annotation"), "Profile_B_Annotation", pool_adaptive, q_by_id)
    _build_progress_sheet(wb.create_sheet("Annotation_Progress"), questions, len(pool_recursive), len(pool_adaptive))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent, prefix=".stage03-xlsx-") as tmp_dir:
        staged = Path(tmp_dir) / output_path.name
        wb.save(staged)
        wb.close()
        verification = verify_annotation_workbook(
            staged,
            expected_question_count=len(questions),
            expected_pool_a_count=len(pool_recursive),
            expected_pool_b_count=len(pool_adaptive),
        )
        staged.replace(output_path)

    verification["output_path"] = str(output_path)
    verification["file_bytes"] = output_path.stat().st_size
    verification["sanitized_xml_control_chars"] = san_a + san_b
    return verification
