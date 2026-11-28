"""Unit & integration tests for Stage 03 dataset schema, pooling, deduplication, XLSX workbook, and importer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.provenance import CanonicalBlock, CanonicalDocument, CanonicalPage, sha256_bytes
from evaluation.stage03.dataset_schema import (
    STAGE03_DATASET_VERSION,
    check_ku_retention_in_corpora,
    chunk_matches_evidence,
    locate_canonical_spans,
    validate_dataset_and_evidence,
)
from evaluation.stage03.automated_workbook import AUTO_WORKBOOK_SHEETS, generate_automated_workbook
from evaluation.stage03.importer import import_automated_ground_truth, import_ground_truth_from_workbook
from evaluation.stage03.judge import (
    JUDGE_PROMPT_VERSION,
    append_jsonl_records,
    build_judge_batch_prompt,
    build_multi_query_judge_prompt,
    compute_inter_judge_metrics,
    load_jsonl_checkpoint,
    select_mandatory_targeted_crosscheck,
    select_pilot_sample,
    select_stratified_random_audit,
    validate_judge_batch_output,
    verify_supporting_text,
)
from evaluation.stage03.pooling import build_deduplicated_pools
from evaluation.stage03.workbook import (
    FORBIDDEN_ANNOTATION_COLUMNS,
    REQUIRED_SHEETS,
    generate_annotation_workbook,
    verify_annotation_workbook,
)


def _synthetic_stage03_fixture(tmp_path: Path):
    """Build an isolated synthetic 2-query Stage 03 fixture (never touches official datasets)."""
    page_text = (
        "Binary Search requires a sorted array.\n\n"
        "Hash Table resolves collisions via chaining or open addressing.\n\n"
        "=A1+B1 formula-like literal text in source chunk.\n\n"
        "Extra heading context for repeated header verification."
    )
    blk0 = CanonicalBlock("doc1:p1:b0", "paragraph", "Binary Search requires a sorted array.", 0, 38)
    s1 = page_text.index("Hash Table")
    blk1 = CanonicalBlock("doc1:p1:b1", "paragraph", "Hash Table resolves collisions via chaining or open addressing.", s1, s1 + 63)
    s2 = page_text.index("=A1+B1")
    blk2 = CanonicalBlock("doc1:p1:b2", "paragraph", "=A1+B1 formula-like literal text in source chunk.", s2, s2 + 49)
    s3 = page_text.index("Extra heading")
    blk3 = CanonicalBlock("doc1:p1:b3", "heading", "Extra heading context for repeated header verification.", s3, s3 + 55)

    doc = CanonicalDocument(
        document_id="doc1",
        filename="sample.pdf",
        source_path="pdf_Files/eng_pdf/sample.pdf",
        source_revision="a" * 64,
        file_type="pdf",
        file_bytes=512,
        lang_hint="en",
        pages=[CanonicalPage("doc1", 1, "pdf", page_text, "en", "en", [blk0, blk1, blk2, blk3])],
    )

    sp_q1, txt_q1 = locate_canonical_spans(doc.pages[0], "Binary Search requires a sorted array.")
    sp_q2, txt_q2 = locate_canonical_spans(doc.pages[0], "Hash Table resolves collisions via chaining or open addressing.")
    sp_hdg, txt_hdg = locate_canonical_spans(doc.pages[0], "Extra heading context for repeated header verification.")

    evidences = [
        {
            "evidence_id": "Q001_E01",
            "query_id": "Q001",
            "knowledge_unit_id": "KU1",
            "evidence_mode": "single",
            "alternative_group_id": None,
            "document_id": "doc1",
            "source_revision": "a" * 64,
            "source_path": "pdf_Files/eng_pdf/sample.pdf",
            "canonical_page": 1,
            "source_spans": sp_q1,
            "evidence_text": txt_q1,
        },
        {
            "evidence_id": "Q002_E01",
            "query_id": "Q002",
            "knowledge_unit_id": "KU1",
            "evidence_mode": "single",
            "alternative_group_id": None,
            "document_id": "doc1",
            "source_revision": "a" * 64,
            "source_path": "pdf_Files/eng_pdf/sample.pdf",
            "canonical_page": 1,
            "source_spans": sp_q2,
            "evidence_text": txt_q2,
        },
        {
            "evidence_id": "Q002_E02",
            "query_id": "Q002",
            "knowledge_unit_id": "KU2",
            "evidence_mode": "single",
            "alternative_group_id": None,
            "document_id": "doc1",
            "source_revision": "a" * 64,
            "source_path": "pdf_Files/eng_pdf/sample.pdf",
            "canonical_page": 1,
            "source_spans": sp_hdg,
            "evidence_text": txt_hdg,
        },
    ]

    questions = [
        {
            "query_id": "Q001",
            "query": "What precondition does Binary Search require?",
            "topic": "Searching & Sorting",
            "language": "en",
            "difficulty": "easy",
            "question_type": "definition",
            "reference_answer": "Binary Search requires a sorted array.",
            "required_knowledge_units": [{"ku_id": "KU1", "description": "Sorted array required.", "evidence_mode": "single"}],
            "source_evidence_ids": ["Q001_E01"],
            "source_document_references": ["pdf_Files/eng_pdf/sample.pdf (p.1)"],
            "human_approval_status": "DRAFT",
        },
        {
            "query_id": "Q002",
            "query": "How does a Hash Table resolve collisions?",
            "topic": "Hashing",
            "language": "en",
            "difficulty": "easy",
            "question_type": "concept_explanation",
            "reference_answer": "Via chaining or open addressing.",
            "required_knowledge_units": [
                {"ku_id": "KU1", "description": "Chaining or open addressing.", "evidence_mode": "single"},
                {"ku_id": "KU2", "description": "Heading context verified.", "evidence_mode": "single"},
            ],
            "source_evidence_ids": ["Q002_E01", "Q002_E02"],
            "source_document_references": ["pdf_Files/eng_pdf/sample.pdf (p.1)"],
            "human_approval_status": "DRAFT",
        },
    ]

    # Profile A chunks (including overlap-backed evidence & leading '=' literal text)
    chunks_a = [
        {
            "chunk_id": "A_001", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec1",
            "text": "Binary Search requires a sorted array.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_A_ID, "content_type": "recursive_segment",
                         "source_spans": sp_q1, "overlap_segments": [], "synthetic_segments": []},
        },
        {
            "chunk_id": "A_002", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec2",
            "text": "=A1+B1 formula-like literal text in source chunk.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_A_ID, "content_type": "recursive_segment",
                         "source_spans": [{"page": 1, "block_id": "doc1:p1:b2", "char_start": s2, "char_end": s2 + 49}],
                         "overlap_segments": [{"page": 1, "char_start": s1, "char_end": s1 + 35, "text_len": 35}],
                         "synthetic_segments": []},
        },
        {
            "chunk_id": "A_003", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec3",
            "text": "Extra heading context for repeated header verification.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_A_ID, "content_type": "recursive_segment",
                         "source_spans": sp_hdg, "overlap_segments": [], "synthetic_segments": []},
        },
    ]

    # Profile B chunks (including source_backed synthetic heading_prefix + non-source-backed synthetic segment)
    chunks_b = [
        {
            "chunk_id": "B_001", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec1",
            "text": "Binary Search requires a sorted array.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_B_ID, "content_type": "definition",
                         "source_spans": sp_q1, "overlap_segments": [], "synthetic_segments": []},
        },
        {
            "chunk_id": "B_002", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec2",
            "text": "Extra heading context for repeated header verification.\nHash Table resolves collisions via chaining or open addressing.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_B_ID, "content_type": "narrative",
                         "source_spans": sp_q2, "overlap_segments": [],
                         "synthetic_segments": [{"kind": "heading_prefix", "source_backed": True, "page": 1,
                                                 "char_start": sp_hdg[0]["char_start"], "char_end": sp_hdg[0]["char_end"], "text_len": 55}]},
        },
        {
            "chunk_id": "B_003", "document_id": "doc1", "filename": "sample.pdf", "page": 1, "section": "Sec3",
            "text": "Unrelated chunk with non-source-backed synthetic segment.",
            "metadata": {"source_path": "pdf_Files/eng_pdf/sample.pdf", "profile_id": PROFILE_B_ID, "content_type": "narrative",
                         "source_spans": [{"page": 1, "block_id": "doc1:p1:b2", "char_start": s2, "char_end": s2 + 10}],
                         "overlap_segments": [],
                         "synthetic_segments": [{"kind": "heading_prefix", "source_backed": False, "page": 1,
                                                 "char_start": sp_hdg[0]["char_start"], "char_end": sp_hdg[0]["char_end"], "text_len": 55}]},
        },
    ]

    # Raw retrieval results with duplicates across methods; A_002 & A_003 omitted from Q002 raw to test evidence augmentation!
    raw_results = [
        {"query_id": "Q001", "profile_id": PROFILE_A_ID, "method": "bm25", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "A_001", "score": 5.0}, {"rank": 2, "chunk_id": "A_002", "score": 1.0}]},
        {"query_id": "Q001", "profile_id": PROFILE_A_ID, "method": "dense", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "A_001", "score": 0.9}]},
        {"query_id": "Q002", "profile_id": PROFILE_A_ID, "method": "bm25", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "A_001", "score": 2.0}]},
        {"query_id": "Q001", "profile_id": PROFILE_B_ID, "method": "bm25", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "B_001", "score": 5.0}, {"rank": 2, "chunk_id": "B_003", "score": 0.5}]},
        {"query_id": "Q001", "profile_id": PROFILE_B_ID, "method": "dense", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "B_001", "score": 0.92}]},
        {"query_id": "Q002", "profile_id": PROFILE_B_ID, "method": "bm25", "requested_top_k": 60,
         "results": [{"rank": 1, "chunk_id": "B_003", "score": 1.2}]},
    ]

    corpus_rows = {PROFILE_A_ID: chunks_a, PROFILE_B_ID: chunks_b}
    corpus_by_id = {pid: {c["chunk_id"]: c for c in rows} for pid, rows in corpus_rows.items()}
    pooled = build_deduplicated_pools(
        questions=questions,
        evidences=evidences,
        raw_results=raw_results,
        corpus_rows_by_profile=corpus_rows,
        corpus_by_id=corpus_by_id,
    )

    q_path = tmp_path / "questions.jsonl"
    q_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    ca_path = tmp_path / "corpus_a.jsonl"
    cb_path = tmp_path / "corpus_b.jsonl"
    ca_path.write_text("\n".join(json.dumps(c) for c in chunks_a) + "\n", encoding="utf-8")
    cb_path.write_text("\n".join(json.dumps(c) for c in chunks_b) + "\n", encoding="utf-8")
    pa_path = tmp_path / "pool_a.jsonl"
    pb_path = tmp_path / "pool_b.jsonl"
    pa_path.write_text("\n".join(json.dumps(r) for r in pooled["pool_recursive"]) + "\n", encoding="utf-8")
    pb_path.write_text("\n".join(json.dumps(r) for r in pooled["pool_adaptive"]) + "\n", encoding="utf-8")

    manifest = {
        "dataset_version": STAGE03_DATASET_VERSION,
        "files": {"questions_sha256": sha256_bytes(q_path.read_bytes())},
        "frozen_corpora": {
            PROFILE_A_ID: {"corpus_sha256": sha256_bytes(ca_path.read_bytes())},
            PROFILE_B_ID: {"corpus_sha256": sha256_bytes(cb_path.read_bytes())},
        },
    }
    m_path = tmp_path / "manifest.json"
    m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    wb_path = tmp_path / "annotation.xlsx"
    generate_annotation_workbook(
        output_path=wb_path,
        questions=questions,
        evidences=evidences,
        pool_recursive=pooled["pool_recursive"],
        pool_adaptive=pooled["pool_adaptive"],
    )

    return {
        "doc": doc,
        "questions": questions,
        "evidences": evidences,
        "corpus_rows": corpus_rows,
        "pooled": pooled,
        "q_path": q_path,
        "m_path": m_path,
        "pa_path": pa_path,
        "pb_path": pb_path,
        "ca_path": ca_path,
        "cb_path": cb_path,
        "wb_path": wb_path,
    }


def _complete_workbook(wb_path: Path, *, q2_all_zero_profile_a: bool = False) -> None:
    """Fill APPROVED statuses and valid 0/1/2 grades on a synthetic workbook for importer tests."""
    wb = load_workbook(wb_path)
    ws_q = wb["Questions"]
    for r in range(2, ws_q.max_row + 1):
        ws_q.cell(r, 11).value = "APPROVED"
    for sname in ("Profile_A_Annotation", "Profile_B_Annotation"):
        ws = wb[sname]
        for r in range(2, ws.max_row + 1):
            qid = ws.cell(r, 2).value
            cid = ws.cell(r, 6).value
            if q2_all_zero_profile_a and sname == "Profile_A_Annotation" and qid == "Q002":
                ws.cell(r, 10).value = 0
            elif cid in ("A_001", "B_001", "B_002"):
                ws.cell(r, 10).value = 2
            elif cid == "A_002":
                ws.cell(r, 10).value = 1
            else:
                ws.cell(r, 10).value = 0
    wb.save(wb_path)
    wb.close()


def test_dataset_validation_evidence_matching_and_dedup_augmentation(tmp_path):
    fx = _synthetic_stage03_fixture(tmp_path)
    stats = validate_dataset_and_evidence(
        fx["questions"], fx["evidences"], [fx["doc"]], expected_question_count=2, require_draft_status=True
    )
    assert stats["question_count"] == 2
    assert stats["evidence_count"] == 3

    # Verify source-backed synthetic heading_prefix matches Q002_E02, while non-source-backed B_003 does NOT
    ev_hdg = fx["evidences"][2]
    assert chunk_matches_evidence(fx["corpus_rows"][PROFILE_B_ID][1], ev_hdg) is True
    assert chunk_matches_evidence(fx["corpus_rows"][PROFILE_B_ID][2], ev_hdg) is False
    # Overlap segment in A_002 matches Q002_E01
    assert chunk_matches_evidence(fx["corpus_rows"][PROFILE_A_ID][1], fx["evidences"][1]) is True

    # Verify KU retention check
    ret = check_ku_retention_in_corpora(fx["questions"], fx["evidences"], fx["corpus_rows"])
    assert ret[PROFILE_A_ID]["retained_kus"] == 3 and ret[PROFILE_B_ID]["retained_kus"] == 3

    # Verify deduplication removed duplicate A_001 (Q001) & B_001 (Q001) and evidence augmentation added missing chunks
    diag = fx["pooled"]["diagnostics"]
    assert diag["totals_by_profile"][PROFILE_A_ID]["duplicates_removed"] == 1
    assert diag["totals_by_profile"][PROFILE_A_ID]["evidence_added_candidates"] == 2  # A_002 (overlap) & A_003 (span)
    assert diag["totals_by_profile"][PROFILE_B_ID]["evidence_added_candidates"] == 1  # B_002 only (not B_003!)

    # Same chunk_id across different queries must NOT be deduplicated away
    a_q_pairs = {(r["query_id"], r["chunk_id"]) for r in fx["pooled"]["pool_recursive"]}
    assert ("Q001", "A_001") in a_q_pairs and ("Q002", "A_001") in a_q_pairs


def test_workbook_structure_blindness_literal_equals_and_progress_formulas(tmp_path):
    fx = _synthetic_stage03_fixture(tmp_path)
    v = verify_annotation_workbook(
        fx["wb_path"],
        expected_question_count=2,
        expected_pool_a_count=len(fx["pooled"]["pool_recursive"]),
        expected_pool_b_count=len(fx["pooled"]["pool_adaptive"]),
    )
    assert v["zip_integrity"] == "PASSED"
    assert tuple(v["sheets"]) == REQUIRED_SHEETS
    assert v["all_initial_statuses_draft"] is True
    assert v["all_initial_grades_blank"] is True
    assert v["blind_method_verified"] is True

    wb = load_workbook(fx["wb_path"], data_only=False)
    try:
        # Check literal '=' string preservation (must stay data_type='s', never Excel formula)
        ws_a = wb["Profile_A_Annotation"]
        eq_cells = [ws_a.cell(r, 7) for r in range(2, ws_a.max_row + 1) if str(ws_a.cell(r, 7).value).startswith("=A1+B1")]
        assert eq_cells and all(c.data_type == "s" for c in eq_cells)

        # Verify zero forbidden columns
        hdrs = {str(ws_a.cell(1, c).value).strip().lower() for c in range(1, ws_a.max_column + 1)}
        assert not (hdrs & set(FORBIDDEN_ANNOTATION_COLUMNS))

        # Verify Annotation_Progress formulas count 0 as judged (Judged = G0 + G1 + G2, Unjudged = Total - Judged)
        ws_p = wb["Annotation_Progress"]
        assert ws_p["C4"].value == "=E4+F4+G4"
        assert ws_p["D4"].value == "=B4-C4"
        assert "COUNTIF(Profile_A_Annotation!J2:J" in ws_p["E4"].value and ",0)" in ws_p["E4"].value
        assert ws_p["D10"].value == "=F10+G10+H10"
        assert ws_p["E10"].value == "=C10-D10"
        assert "COUNTIFS(Profile_A_Annotation!$B$2:$B$" in ws_p["F10"].value and ",0)" in ws_p["F10"].value
    finally:
        wb.close()


def test_importer_valid_complete_and_zero_positive_query(tmp_path):
    fx = _synthetic_stage03_fixture(tmp_path)
    _complete_workbook(fx["wb_path"], q2_all_zero_profile_a=True)

    out_a = tmp_path / "gt_recursive.jsonl"
    out_b = tmp_path / "gt_adaptive.jsonl"
    res = import_ground_truth_from_workbook(
        fx["wb_path"],
        questions_path=fx["q_path"],
        manifest_path=fx["m_path"],
        pool_a_path=fx["pa_path"],
        pool_b_path=fx["pb_path"],
        corpus_a_path=fx["ca_path"],
        corpus_b_path=fx["cb_path"],
        output_a_path=out_a,
        output_b_path=out_b,
        expected_question_count=2,
        expected_dataset_version=STAGE03_DATASET_VERSION,
    )
    assert out_a.exists() and out_b.exists()
    # Q002 in Profile A was judged all-zero: must stay preserved with zero_positive=True & all 0s kept!
    assert res["zero_positive_queries"][PROFILE_A_ID] == ["Q002"]
    assert res["zero_positive_queries"][PROFILE_B_ID] == []
    q2_a = next(r for r in res["ground_truth_recursive"] if r["query_id"] == "Q002")
    assert q2_a["relevant_ids"] == []
    assert q2_a["zero_positive"] is True
    assert q2_a["labels_complete"] is True
    assert q2_a["relevance_grades"] and all(v == 0 for v in q2_a["relevance_grades"].values())
    q1_a = next(r for r in res["ground_truth_recursive"] if r["query_id"] == "Q001")
    assert set(q1_a["relevant_ids"]) == {"A_001", "A_002"}


@pytest.mark.parametrize("case", [
    "missing_grade",
    "invalid_grade",
    "duplicate_ann_id",
    "unknown_chunk_id",
    "wrong_profile_chunk",
    "missing_candidate_row",
    "modified_query_text",
    "modified_chunk_text",
    "corpus_sha_mismatch",
    "dataset_version_mismatch",
    "incomplete_approval",
])
def test_importer_rejects_all_invalid_annotation_cases(tmp_path, case):
    fx = _synthetic_stage03_fixture(tmp_path)
    _complete_workbook(fx["wb_path"])

    kwargs = dict(
        questions_path=fx["q_path"],
        manifest_path=fx["m_path"],
        pool_a_path=fx["pa_path"],
        pool_b_path=fx["pb_path"],
        corpus_a_path=fx["ca_path"],
        corpus_b_path=fx["cb_path"],
        expected_question_count=2,
        expected_dataset_version=STAGE03_DATASET_VERSION,
    )

    wb = load_workbook(fx["wb_path"])
    if case == "missing_grade":
        wb["Profile_A_Annotation"].cell(2, 10).value = None
    elif case == "invalid_grade":
        wb["Profile_A_Annotation"].cell(2, 10).value = 3
    elif case == "duplicate_ann_id":
        ws = wb["Profile_A_Annotation"]
        ws.cell(3, 1).value = ws.cell(2, 1).value
    elif case == "unknown_chunk_id":
        wb["Profile_A_Annotation"].cell(2, 6).value = "UNKNOWN_CHUNK_999"
    elif case == "wrong_profile_chunk":
        wb["Profile_A_Annotation"].cell(2, 6).value = "B_001"
    elif case == "missing_candidate_row":
        wb["Profile_A_Annotation"].delete_rows(2, 1)
    elif case == "modified_query_text":
        wb["Questions"].cell(2, 2).value = "Tampered question text?"
    elif case == "modified_chunk_text":
        wb["Profile_A_Annotation"].cell(2, 7).value = "Tampered chunk text"
    elif case == "incomplete_approval":
        wb["Questions"].cell(2, 11).value = "DRAFT"
    wb.save(fx["wb_path"])
    wb.close()

    if case == "corpus_sha_mismatch":
        fx["ca_path"].write_text(fx["ca_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif case == "dataset_version_mismatch":
        kwargs["expected_dataset_version"] = "wrong_version_v999"

    with pytest.raises(ValueError):
        import_ground_truth_from_workbook(fx["wb_path"], **kwargs)


def test_judge_supporting_text_verification():
    chunk = "Binary Search requires a sorted array.\nIt runs in O(log n) worst-case time."
    assert verify_supporting_text("Binary Search requires a sorted array.", chunk) is True
    assert verify_supporting_text("binary   search requires\n a sorted array.", chunk) is True
    assert verify_supporting_text("Binary Search requires ... O(log n) worst-case time.", chunk) is True
    assert verify_supporting_text("Binary Search requires ... completely fabricated claim", chunk) is False
    assert verify_supporting_text("Hash table uses chaining", chunk) is False
    assert verify_supporting_text("", chunk) is False
    pdf_chunk = "What is the \ufb01rst element of LIS? max{x(i) | 0 \uf8ff i < |A|}"
    assert verify_supporting_text("first element of LIS? max{x(i) | 0 <= i < |A|}", pdf_chunk) is True
    assert verify_supporting_text("first element of LIS? max{x(i) | 0  i < |A|}", pdf_chunk) is True


def test_judge_batch_validation_and_blind_prompts(tmp_path):
    fx = _synthetic_stage03_fixture(tmp_path)
    q1 = fx["questions"][0]
    ev1 = [fx["evidences"][0]]
    cands = [r for r in fx["pooled"]["pool_recursive"] if r["query_id"] == "Q001"]
    assert len(cands) == 2
    c_bin = next(c for c in cands if c["chunk_id"] == "A_001")
    c_other = next(c for c in cands if c["chunk_id"] == "A_002")

    prompt = build_judge_batch_prompt(q1, ev1, [{"annotation_id": c["annotation_id"], "chunk_text": c["chunk_text"]} for c in cands])
    assert "PROMPT-INJECTION" in prompt
    assert "bm25" not in prompt.lower()
    assert "recursive_512_100" not in prompt

    multi_prompt = build_multi_query_judge_prompt(
        [(q1, ev1, [{"annotation_id": c["annotation_id"], "chunk_text": c["chunk_text"]} for c in cands])]
    )
    assert "QUERY GROUP Q001" in multi_prompt
    assert "PROMPT-INJECTION" in multi_prompt

    # Valid output for both candidates
    raw_ok = {
        "annotations": [
            {
                "annotation_id": c_bin["annotation_id"],
                "ai_grade": 2,
                "supported_knowledge_unit_ids": ["KU1"],
                "supporting_text_from_chunk": "Binary Search requires a sorted array.",
                "relevance_reason": "Directly states the sorted-array precondition.",
                "confidence_indicator": "high",
            },
            {
                "annotation_id": c_other["annotation_id"],
                "ai_grade": 0,
                "supported_knowledge_unit_ids": [],
                "supporting_text_from_chunk": "",
                "relevance_reason": "Unrelated formula text.",
                "confidence_indicator": "high",
            },
        ]
    }
    valid, errs = validate_judge_batch_output(
        raw_ok, cands, {"KU1"},
        judge_model="test_flash", prompt_version=JUDGE_PROMPT_VERSION, judge_config={"tier": "flash"},
    )
    assert not errs and len(valid) == 2
    assert valid[0]["supporting_text_valid"] is True and valid[0]["ai_grade"] == 2
    assert valid[1]["ai_grade"] == 0 and valid[1]["supported_knowledge_unit_ids"] == []

    # Invalid cases: fabricated supporting text, missing ID, grade 2 without KU, duplicate/unknown/invalid grade
    raw_bad = {
        "annotations": [
            {
                "annotation_id": c_bin["annotation_id"],
                "ai_grade": 2,
                "supported_knowledge_unit_ids": ["KU1"],
                "supporting_text_from_chunk": "Fabricated text not in chunk",
                "relevance_reason": "Claimed support",
                "confidence_indicator": "high",
            },
            {
                "annotation_id": c_bin["annotation_id"],
                "ai_grade": 1,
                "supported_knowledge_unit_ids": ["KU1"],
                "supporting_text_from_chunk": "Binary Search",
                "relevance_reason": "Duplicate ID",
                "confidence_indicator": "medium",
            },
            {
                "annotation_id": "UNKNOWN_ID",
                "ai_grade": 0,
                "supported_knowledge_unit_ids": [],
                "supporting_text_from_chunk": "",
                "relevance_reason": "Unknown",
                "confidence_indicator": "high",
            },
        ]
    }
    valid_b, errs_b = validate_judge_batch_output(
        raw_bad, cands, {"KU1"},
        judge_model="test_flash", prompt_version=JUDGE_PROMPT_VERSION, judge_config={"tier": "flash"},
    )
    assert valid_b == []
    err_codes = {e["error"] for e in errs_b}
    assert "unverified_supporting_text" in err_codes
    assert "duplicate_annotation_id" in err_codes
    assert "unknown_annotation_id" in err_codes
    assert "missing_annotation_id" in err_codes

    # Grade 2 without KU ID and boolean grade rejected
    raw_bad2 = {
        "annotations": [
            {
                "annotation_id": c_bin["annotation_id"],
                "ai_grade": 2,
                "supported_knowledge_unit_ids": [],
                "supporting_text_from_chunk": "Binary Search requires a sorted array.",
                "relevance_reason": "Missing KU",
                "confidence_indicator": "high",
            },
            {
                "annotation_id": c_other["annotation_id"],
                "ai_grade": True,
                "supported_knowledge_unit_ids": [],
                "supporting_text_from_chunk": "",
                "relevance_reason": "Bool grade",
                "confidence_indicator": "high",
            },
        ]
    }
    valid_b2, errs_b2 = validate_judge_batch_output(
        raw_bad2, cands, {"KU1"},
        judge_model="test_flash", prompt_version=JUDGE_PROMPT_VERSION, judge_config={"tier": "flash"},
    )
    assert valid_b2 == []
    assert any(e["error"] == "grade2_missing_ku_id" for e in errs_b2)
    assert any(e["error"].startswith("invalid_grade:") for e in errs_b2)


def test_inter_judge_metrics_sampling_and_checkpoint_roundtrip(tmp_path):
    empty = compute_inter_judge_metrics([])
    assert empty["count"] == 0 and empty["quadratic_weighted_kappa"] is None

    perfect = compute_inter_judge_metrics([(0, 0), (1, 1), (2, 2), (0, 0)])
    assert perfect["exact_agreement_pct"] == 100.0
    assert perfect["quadratic_weighted_kappa"] == 1.0
    assert perfect["linear_weighted_kappa"] == 1.0
    assert perfect["severe_0_vs_2_conflicts"] == 0

    mixed = compute_inter_judge_metrics([(0, 0), (0, 0), (1, 1), (2, 2), (0, 2), (2, 1)])
    assert mixed["count"] == 6
    assert mixed["severe_0_vs_2_conflicts"] == 1
    assert mixed["confusion_matrix"] == [[2, 0, 1], [0, 1, 0], [0, 1, 1]]
    assert -1.0 <= mixed["quadratic_weighted_kappa"] <= 1.0

    with pytest.raises(ValueError):
        compute_inter_judge_metrics([(0, 3)])

    fx = _synthetic_stage03_fixture(tmp_path)
    q_by_id = {q["query_id"]: q for q in fx["questions"]}
    pilot = select_pilot_sample(fx["pooled"]["pool_recursive"], fx["pooled"]["pool_adaptive"], q_by_id, per_profile_target=2)
    assert len(pilot) == 4

    all_pool = fx["pooled"]["pool_recursive"] + fx["pooled"]["pool_adaptive"]
    pool_by_id = {c["annotation_id"]: c for c in all_pool}
    prim = []
    for idx, c in enumerate(all_pool):
        # Introduce cross-profile inconsistency on identical text "Binary Search requires a sorted array."
        g = 2 if c["chunk_id"] == "A_001" else (1 if c["chunk_id"] == "B_001" else 0)
        conf = "low" if idx == 0 else "high"
        prim.append({
            "annotation_id": c["annotation_id"],
            "query_id": c["query_id"],
            "profile_id": c["profile_id"],
            "chunk_id": c["chunk_id"],
            "ai_grade": g,
            "confidence_indicator": conf,
            "supporting_text_valid": True,
            "supported_knowledge_unit_ids": ["KU1"] if g > 0 else [],
            "supporting_text_from_chunk": "Binary Search" if g > 0 else "",
            "relevance_reason": "Clear test explanation for synthetic chunk.",
        })
    rand_ids = select_stratified_random_audit(prim, fraction=0.25, seed=42)
    assert len(rand_ids) == round(len(prim) * 0.25)

    triggers = select_mandatory_targeted_crosscheck(prim, pool_by_id, q_by_id)
    assert any("cross_profile_inconsistency" in v for v in triggers.values())
    assert "low_confidence" in triggers[prim[0]["annotation_id"]]

    ckpt_path = tmp_path / "ckpt.jsonl"
    append_jsonl_records(ckpt_path, prim[:2])
    append_jsonl_records(ckpt_path, prim[2:])
    loaded = load_jsonl_checkpoint(ckpt_path)
    assert set(loaded.keys()) == {r["annotation_id"] for r in prim}


def _build_automated_fixture(tmp_path: Path, *, q2_all_zero_profile_a: bool = False):
    fx = _synthetic_stage03_fixture(tmp_path)
    # Mark questions AI_VALIDATED (human_approval_status stays DRAFT)
    ai_questions = []
    for q in fx["questions"]:
        q2 = dict(q)
        q2["automated_review_status"] = "AI_VALIDATED"
        q2["automated_review_model"] = "antigravity_agent_pro"
        q2["automated_review_reason"] = "Verified against canonical source."
        ai_questions.append(q2)
    fx["q_path"].write_text("\n".join(json.dumps(q) for q in ai_questions) + "\n", encoding="utf-8")
    manifest = json.loads(fx["m_path"].read_text(encoding="utf-8"))
    manifest["files"]["questions_sha256"] = sha256_bytes(fx["q_path"].read_bytes())
    fx["m_path"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_pool = fx["pooled"]["pool_recursive"] + fx["pooled"]["pool_adaptive"]
    final_rows = []
    for cand in all_pool:
        qid = cand["query_id"]
        cid = cand["chunk_id"]
        if q2_all_zero_profile_a and cand["profile_id"] == PROFILE_A_ID and qid == "Q002":
            fg = 0
        elif cid in ("A_001", "B_001", "B_002"):
            fg = 2
        elif cid == "A_002":
            fg = 1
        else:
            fg = 0
        final_rows.append({
            "annotation_id": cand["annotation_id"],
            "query_id": qid,
            "profile_id": cand["profile_id"],
            "chunk_id": cid,
            "ai_grade": fg,
            "secondary_grade": fg if fg == 2 else None,
            "final_grade": fg,
            "supporting_text_from_chunk": cand["chunk_text"][:20] if fg > 0 else "",
            "supporting_text_valid": True,
            "supported_knowledge_unit_ids": ["KU1"] if fg > 0 else [],
            "relevance_reason": "Synthetic automated judgment.",
            "disagreement_status": "AGREED" if fg == 2 else "NOT_CROSS_CHECKED",
            "adjudication_reason": "",
            "annotation_source": "AI-cross-checked" if fg == 2 else "AI-generated",
            "judge_model": "antigravity_agent_flash",
            "annotation_status": "VALID",
        })
    final_path = tmp_path / "ai_annotations_final.jsonl"
    final_path.write_text("\n".join(json.dumps(r) for r in final_rows) + "\n", encoding="utf-8")
    audit = {
        "primary_judge_model": "antigravity_agent_flash",
        "secondary_judge_model": "antigravity_agent_pro",
        "adjudicator_model": "antigravity_agent_pro",
        "final_prompt_version": JUDGE_PROMPT_VERSION,
        "total_annotations": len(final_rows),
        "supporting_text_validation_rate_pct": 100.0,
        "pilot_summary": {"count": 4, "rounds_completed": 1, "exact_agreement_pct": 100.0, "quadratic_weighted_kappa": 1.0, "severe_0_vs_2_conflicts": 0},
        "random_audit_10pct": {"count": 2, "exact_agreement_pct": 100.0, "quadratic_weighted_kappa": 1.0, "linear_weighted_kappa": 1.0, "disagreement_count": 0, "disagreement_rate_pct": 0.0, "severe_0_vs_2_conflicts": 0, "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 1]]},
        "targeted_cross_check": {"count": 1, "exact_agreement_pct": 100.0, "disagreement_count": 0},
        "adjudication_summary": {"total_adjudicated": 0, "changed_from_primary_count": 0, "grade_correction_rate_pct": 0.0},
        "cross_profile_inconsistency_count": 0,
    }
    audit_path = tmp_path / "automated_quality_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fx["ai_questions"] = ai_questions
    fx["final_rows"] = final_rows
    fx["final_path"] = final_path
    fx["audit"] = audit
    fx["audit_path"] = audit_path
    return fx


def test_automated_workbook_and_importer_valid_complete(tmp_path):
    fx = _build_automated_fixture(tmp_path, q2_all_zero_profile_a=True)
    auto_wb = tmp_path / "ground_truth_automated.xlsx"
    wb_res = generate_automated_workbook(
        output_path=auto_wb,
        questions=fx["ai_questions"],
        evidences=fx["evidences"],
        pool_a=fx["pooled"]["pool_recursive"],
        pool_b=fx["pooled"]["pool_adaptive"],
        final_annotations_by_id={r["annotation_id"]: r for r in fx["final_rows"]},
        quality_audit=fx["audit"],
    )
    assert wb_res["zip_integrity"] == "PASSED"
    assert tuple(wb_res["sheets"]) == AUTO_WORKBOOK_SHEETS

    wb = load_workbook(auto_wb, data_only=False)
    try:
        ws_a = wb["Profile_A_Annotation"]
        eq_cells = [ws_a.cell(r, 4) for r in range(2, ws_a.max_row + 1) if str(ws_a.cell(r, 4).value).startswith("=A1+B1")]
        assert eq_cells and all(c.data_type == "s" for c in eq_cells)
        ws_q = wb["Questions"]
        assert ws_q.cell(2, 11).value == "DRAFT"
        assert ws_q.cell(2, 12).value == "AI_VALIDATED"
    finally:
        wb.close()

    out_a = tmp_path / "gt_auto_recursive.jsonl"
    out_b = tmp_path / "gt_auto_adaptive.jsonl"
    res = import_automated_ground_truth(
        questions_path=fx["q_path"],
        manifest_path=fx["m_path"],
        pool_a_path=fx["pa_path"],
        pool_b_path=fx["pb_path"],
        corpus_a_path=fx["ca_path"],
        corpus_b_path=fx["cb_path"],
        final_annotations_path=fx["final_path"],
        quality_audit_path=fx["audit_path"],
        output_a_path=out_a,
        output_b_path=out_b,
        expected_question_count=2,
        expected_dataset_version=STAGE03_DATASET_VERSION,
    )
    assert res["human_validated"] is False
    assert res["annotation_method"] == "llm_generated_pooled_relevance_judgments"
    assert res["zero_positive_queries"][PROFILE_A_ID] == ["Q002"]
    assert res["zero_positive_queries"][PROFILE_B_ID] == []
    assert out_a.exists() and out_b.exists()
    q2_a = next(r for r in res["ground_truth_recursive"] if r["query_id"] == "Q002")
    assert q2_a["zero_positive"] is True and q2_a["human_validated"] is False


@pytest.mark.parametrize("case", [
    "missing_ai_validated",
    "fabricated_human_approved",
    "missing_annotation_id",
    "duplicate_annotation_id",
    "extra_annotation_id",
    "invalid_final_grade",
    "metadata_mismatch",
])
def test_automated_importer_rejects_invalid_cases(tmp_path, case):
    fx = _build_automated_fixture(tmp_path)
    if case == "missing_ai_validated":
        fx["ai_questions"][0]["automated_review_status"] = "PENDING"
        fx["q_path"].write_text("\n".join(json.dumps(q) for q in fx["ai_questions"]) + "\n", encoding="utf-8")
        m = json.loads(fx["m_path"].read_text(encoding="utf-8"))
        m["files"]["questions_sha256"] = sha256_bytes(fx["q_path"].read_bytes())
        fx["m_path"].write_text(json.dumps(m), encoding="utf-8")
    elif case == "fabricated_human_approved":
        fx["ai_questions"][0]["human_approval_status"] = "APPROVED"
        fx["q_path"].write_text("\n".join(json.dumps(q) for q in fx["ai_questions"]) + "\n", encoding="utf-8")
        m = json.loads(fx["m_path"].read_text(encoding="utf-8"))
        m["files"]["questions_sha256"] = sha256_bytes(fx["q_path"].read_bytes())
        fx["m_path"].write_text(json.dumps(m), encoding="utf-8")
    elif case == "missing_annotation_id":
        fx["final_path"].write_text("\n".join(json.dumps(r) for r in fx["final_rows"][1:]) + "\n", encoding="utf-8")
    elif case == "duplicate_annotation_id":
        rows = fx["final_rows"] + [fx["final_rows"][0]]
        fx["final_path"].write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    elif case == "extra_annotation_id":
        extra = dict(fx["final_rows"][0], annotation_id="EXTRA_999")
        rows = fx["final_rows"] + [extra]
        fx["final_path"].write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    elif case == "invalid_final_grade":
        fx["final_rows"][0]["final_grade"] = 5
        fx["final_path"].write_text("\n".join(json.dumps(r) for r in fx["final_rows"]) + "\n", encoding="utf-8")
    elif case == "metadata_mismatch":
        fx["final_rows"][0]["chunk_id"] = "WRONG_CHUNK"
        fx["final_path"].write_text("\n".join(json.dumps(r) for r in fx["final_rows"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        import_automated_ground_truth(
            questions_path=fx["q_path"],
            manifest_path=fx["m_path"],
            pool_a_path=fx["pa_path"],
            pool_b_path=fx["pb_path"],
            corpus_a_path=fx["ca_path"],
            corpus_b_path=fx["cb_path"],
            final_annotations_path=fx["final_path"],
            quality_audit_path=fx["audit_path"],
            expected_question_count=2,
            expected_dataset_version=STAGE03_DATASET_VERSION,
        )
