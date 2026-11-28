"""Stage 03 – Strict Ground Truth Importer (Human-Reviewed XLSX -> Profile A & Profile B JSONL)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.provenance import sha256_bytes
from evaluation.stage03.workbook import ANNOTATION_HEADERS, QUESTIONS_HEADERS, REQUIRED_SHEETS, normalize_excel_text


def _parse_grade(raw_value: Any, *, sheet_name: str, row_num: int, ann_id: str) -> int:
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        raise ValueError(f"{sheet_name} row {row_num} ({ann_id}): missing relevance grade (UNJUDGED)")
    if isinstance(raw_value, bool):
        raise ValueError(f"{sheet_name} row {row_num} ({ann_id}): invalid boolean grade {raw_value!r}")
    if isinstance(raw_value, int):
        grade = raw_value
    elif isinstance(raw_value, float) and raw_value.is_integer():
        grade = int(raw_value)
    elif isinstance(raw_value, str) and raw_value.strip() in ("0", "1", "2"):
        grade = int(raw_value.strip())
    else:
        raise ValueError(f"{sheet_name} row {row_num} ({ann_id}): invalid relevance grade {raw_value!r}; must be 0, 1, or 2")
    if grade not in (0, 1, 2):
        raise ValueError(f"{sheet_name} row {row_num} ({ann_id}): grade {grade} outside {{0, 1, 2}}")
    return grade


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def import_ground_truth_from_workbook(
    workbook_path: Path,
    *,
    questions_path: Path,
    manifest_path: Path,
    pool_a_path: Path,
    pool_b_path: Path,
    corpus_a_path: Path,
    corpus_b_path: Path,
    output_a_path: Path | None = None,
    output_b_path: Path | None = None,
    expected_question_count: int = 40,
    expected_dataset_version: str | None = None,
) -> dict[str, Any]:
    """Validate a human-completed annotation workbook against immutable pools & manifests and build Ground Truth."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_version = manifest["dataset_version"]
    if expected_dataset_version is not None and dataset_version != expected_dataset_version:
        raise ValueError(
            f"Question dataset version mismatch: manifest={dataset_version!r} != expected={expected_dataset_version!r}"
        )

    # 1. Verify immutable question dataset & frozen corpora checksums against manifest
    actual_q_sha = sha256_bytes(questions_path.read_bytes())
    if actual_q_sha != manifest["files"]["questions_sha256"]:
        raise ValueError("questions_40.jsonl SHA-256 does not match dataset_manifest.json")

    corpus_paths = {PROFILE_A_ID: corpus_a_path, PROFILE_B_ID: corpus_b_path}
    corpus_ids_by_profile: dict[str, set[str]] = {}
    corpus_sha_by_profile: dict[str, str] = {}
    for pid, cpath in corpus_paths.items():
        actual_csha = sha256_bytes(cpath.read_bytes())
        expected_csha = manifest["frozen_corpora"][pid]["corpus_sha256"]
        if actual_csha != expected_csha:
            raise ValueError(f"Frozen corpus fingerprint mismatch for {pid}: {actual_csha} != {expected_csha}")
        corpus_sha_by_profile[pid] = actual_csha
        corpus_ids_by_profile[pid] = {row["chunk_id"] for row in _load_jsonl(cpath)}

    questions = _load_jsonl(questions_path)
    if len(questions) != expected_question_count:
        raise ValueError(f"Expected {expected_question_count} questions, found {len(questions)}")
    q_by_id = {q["query_id"]: q for q in questions}

    pool_by_profile = {
        PROFILE_A_ID: _load_jsonl(pool_a_path),
        PROFILE_B_ID: _load_jsonl(pool_b_path),
    }

    # 2. Load & validate workbook sheets
    wb = load_workbook(workbook_path, data_only=False)
    try:
        if tuple(wb.sheetnames) != REQUIRED_SHEETS:
            raise ValueError(f"Workbook sheets mismatch: {wb.sheetnames} != {list(REQUIRED_SHEETS)}")

        # Validate Questions sheet & require 100% APPROVED human status + unmodified immutable text
        ws_q = wb["Questions"]
        q_hdr = tuple(ws_q.cell(1, c).value for c in range(1, len(QUESTIONS_HEADERS) + 1))
        if q_hdr != QUESTIONS_HEADERS:
            raise ValueError("Questions sheet headers modified or invalid")

        seen_qids: set[str] = set()
        for r in range(2, ws_q.max_row + 1):
            qid = ws_q.cell(r, 1).value
            if qid is None:
                raise ValueError(f"Questions row {r}: empty Query ID")
            if qid in seen_qids:
                raise ValueError(f"Questions row {r}: duplicate Query ID {qid!r}")
            if qid not in q_by_id:
                raise ValueError(f"Questions row {r}: unexpected Query ID {qid!r}")
            seen_qids.add(qid)

            orig_q = q_by_id[qid]
            wb_query = ws_q.cell(r, 2).value
            wb_ref = ws_q.cell(r, 7).value
            wb_status = ws_q.cell(r, 11).value
            if wb_query != normalize_excel_text(orig_q["query"])[0]:
                raise ValueError(f"Questions row {r} ({qid}): immutable query text was modified; revise dataset version & regenerate pools")
            if wb_ref != normalize_excel_text(orig_q["reference_answer"])[0]:
                raise ValueError(f"Questions row {r} ({qid}): immutable reference_answer was modified; revise dataset version & regenerate pools")
            if wb_status != "APPROVED":
                raise ValueError(
                    f"Questions row {r} ({qid}): human_approval_status is {wb_status!r}, must be 'APPROVED' before official import"
                )

        if seen_qids != set(q_by_id.keys()):
            missing = sorted(set(q_by_id.keys()) - seen_qids)
            raise ValueError(f"Questions sheet missing approved queries: {missing}")

        # Validate Profile_A_Annotation & Profile_B_Annotation sheets
        gt_by_profile: dict[str, list[dict[str, Any]]] = {}
        zero_positive_queries: dict[str, list[str]] = {PROFILE_A_ID: [], PROFILE_B_ID: []}
        grade_counts: dict[str, dict[int, int]] = {
            PROFILE_A_ID: {0: 0, 1: 0, 2: 0},
            PROFILE_B_ID: {0: 0, 1: 0, 2: 0},
        }

        for pid, sheet_name in (
            (PROFILE_A_ID, "Profile_A_Annotation"),
            (PROFILE_B_ID, "Profile_B_Annotation"),
        ):
            other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
            ws = wb[sheet_name]
            hdr = tuple(ws.cell(1, c).value for c in range(1, len(ANNOTATION_HEADERS) + 1))
            if hdr != ANNOTATION_HEADERS:
                raise ValueError(f"{sheet_name}: headers modified or invalid")

            expected_pool = pool_by_profile[pid]
            expected_by_ann_id = {row["annotation_id"]: row for row in expected_pool}

            seen_ann_ids: set[str] = set()
            seen_q_chunk: set[tuple[str, str]] = set()
            grades_by_qid: dict[str, dict[str, int]] = defaultdict(dict)
            notes_by_qid: dict[str, dict[str, str]] = defaultdict(dict)

            for r in range(2, ws.max_row + 1):
                ann_id = ws.cell(r, 1).value
                qid = ws.cell(r, 2).value
                wb_qtext = ws.cell(r, 3).value
                wb_ref = ws.cell(r, 4).value
                cid = ws.cell(r, 6).value
                wb_chunk_text = ws.cell(r, 7).value
                wb_src_doc = ws.cell(r, 8).value
                wb_src_loc = ws.cell(r, 9).value
                raw_grade = ws.cell(r, 10).value
                reviewer_note = ws.cell(r, 12).value

                if not isinstance(ann_id, str) or not ann_id.strip():
                    raise ValueError(f"{sheet_name} row {r}: missing Annotation ID")
                if ann_id in seen_ann_ids:
                    raise ValueError(f"{sheet_name} row {r}: duplicate Annotation ID {ann_id!r}")
                seen_ann_ids.add(ann_id)

                if ann_id not in expected_by_ann_id:
                    raise ValueError(f"{sheet_name} row {r}: unexpected Annotation ID {ann_id!r}")
                orig = expected_by_ann_id[ann_id]

                if cid not in corpus_ids_by_profile[pid]:
                    if cid in corpus_ids_by_profile[other_pid]:
                        raise ValueError(f"{sheet_name} row {r} ({ann_id}): wrong-profile chunk_id {cid!r}")
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): unknown chunk_id {cid!r}")
                if cid != orig["chunk_id"] or qid != orig["query_id"]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable query_id/chunk_id modified")
                if (qid, cid) in seen_q_chunk:
                    raise ValueError(f"{sheet_name} row {r}: duplicate (query_id, chunk_id) = ({qid}, {cid})")
                seen_q_chunk.add((qid, cid))

                orig_q = q_by_id[qid]
                if wb_qtext != normalize_excel_text(orig_q["query"])[0]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable Question text modified")
                if wb_ref != normalize_excel_text(orig_q["reference_answer"])[0]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable Reference Answer modified")
                if wb_chunk_text != normalize_excel_text(orig["chunk_text"])[0]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable Chunk Text modified")
                if wb_src_doc != normalize_excel_text(orig["source_path"])[0]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable Source Document modified")
                if wb_src_loc != normalize_excel_text(orig["source_location"])[0]:
                    raise ValueError(f"{sheet_name} row {r} ({ann_id}): immutable Source Page/Location modified")

                grade = _parse_grade(raw_grade, sheet_name=sheet_name, row_num=r, ann_id=ann_id)
                grades_by_qid[qid][cid] = grade
                grade_counts[pid][grade] += 1
                if reviewer_note is not None and str(reviewer_note).strip():
                    notes_by_qid[qid][cid] = str(reviewer_note).strip()

            missing_ann_ids = set(expected_by_ann_id.keys()) - seen_ann_ids
            if missing_ann_ids:
                raise ValueError(
                    f"{sheet_name}: missing {len(missing_ann_ids)} required candidate rows (e.g. {sorted(missing_ann_ids)[:3]})"
                )

            records: list[dict[str, Any]] = []
            for q in questions:
                qid = q["query_id"]
                q_grades = grades_by_qid.get(qid, {})
                relevant_ids = [cid for cid, g in q_grades.items() if g > 0]
                is_zero_pos = len(relevant_ids) == 0
                if is_zero_pos:
                    zero_positive_queries[pid].append(qid)
                records.append({
                    "query_id": qid,
                    "query": q["query"],
                    "profile_id": pid,
                    "topic": q["topic"],
                    "language": q["language"],
                    "difficulty": q["difficulty"],
                    "question_type": q["question_type"],
                    "reference_answer": q["reference_answer"],
                    "required_knowledge_units": q["required_knowledge_units"],
                    "source_evidence_ids": q["source_evidence_ids"],
                    "relevant_ids": relevant_ids,
                    "relevance_grades": q_grades,
                    "labels_complete": True,
                    "judgment_scope": "judged_pool_complete_not_corpus_exhaustive",
                    "judged_count": len(q_grades),
                    "positive_count": len(relevant_ids),
                    "zero_positive": is_zero_pos,
                    "corpus_sha256": corpus_sha_by_profile[pid],
                    "question_dataset_version": dataset_version,
                    "question_dataset_sha256": actual_q_sha,
                })
            gt_by_profile[pid] = records
    finally:
        wb.close()

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
        path.write_bytes(payload)
        return sha256_bytes(payload)

    written: dict[str, str] = {}
    if output_a_path is not None:
        written[PROFILE_A_ID] = write_jsonl(output_a_path, gt_by_profile[PROFILE_A_ID])
    if output_b_path is not None:
        written[PROFILE_B_ID] = write_jsonl(output_b_path, gt_by_profile[PROFILE_B_ID])

    return {
        "dataset_version": dataset_version,
        "question_count": len(questions),
        "ground_truth_recursive": gt_by_profile[PROFILE_A_ID],
        "ground_truth_adaptive": gt_by_profile[PROFILE_B_ID],
        "grade_counts": grade_counts,
        "zero_positive_queries": zero_positive_queries,
        "written_sha256": written,
    }


def import_automated_ground_truth(
    *,
    questions_path: Path,
    manifest_path: Path,
    pool_a_path: Path,
    pool_b_path: Path,
    corpus_a_path: Path,
    corpus_b_path: Path,
    final_annotations_path: Path,
    quality_audit_path: Path,
    output_a_path: Path | None = None,
    output_b_path: Path | None = None,
    expected_question_count: int = 40,
    expected_dataset_version: str | None = None,
) -> dict[str, Any]:
    """Explicit automated LLM-as-a-Judge import mode (separate from strict human-approval mode).

    Preserves human_validated=False and never fabricates Human Approval Status=APPROVED.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_version = manifest["dataset_version"]
    if expected_dataset_version is not None and dataset_version != expected_dataset_version:
        raise ValueError(
            f"Question dataset version mismatch: manifest={dataset_version!r} != expected={expected_dataset_version!r}"
        )

    actual_q_sha = sha256_bytes(questions_path.read_bytes())
    if actual_q_sha != manifest["files"]["questions_sha256"]:
        raise ValueError("questions_40.jsonl SHA-256 does not match dataset_manifest.json")

    corpus_paths = {PROFILE_A_ID: corpus_a_path, PROFILE_B_ID: corpus_b_path}
    corpus_ids_by_profile: dict[str, set[str]] = {}
    corpus_sha_by_profile: dict[str, str] = {}
    for pid, cpath in corpus_paths.items():
        actual_csha = sha256_bytes(cpath.read_bytes())
        expected_csha = manifest["frozen_corpora"][pid]["corpus_sha256"]
        if actual_csha != expected_csha:
            raise ValueError(f"Frozen corpus fingerprint mismatch for {pid}: {actual_csha} != {expected_csha}")
        corpus_sha_by_profile[pid] = actual_csha
        corpus_ids_by_profile[pid] = {row["chunk_id"] for row in _load_jsonl(cpath)}

    questions = _load_jsonl(questions_path)
    if len(questions) != expected_question_count:
        raise ValueError(f"Expected {expected_question_count} questions, found {len(questions)}")
    for q in questions:
        if q.get("automated_review_status") != "AI_VALIDATED":
            raise ValueError(
                f"{q['query_id']}: automated_review_status must be 'AI_VALIDATED' in automated mode, "
                f"got {q.get('automated_review_status')!r}"
            )
        if q.get("human_approval_status") == "APPROVED":
            raise ValueError(
                f"{q['query_id']}: automated mode refuses fabricated human_approval_status='APPROVED'"
            )

    pool_by_profile = {
        PROFILE_A_ID: _load_jsonl(pool_a_path),
        PROFILE_B_ID: _load_jsonl(pool_b_path),
    }
    quality_audit = json.loads(quality_audit_path.read_text(encoding="utf-8"))
    final_rows = _load_jsonl(final_annotations_path)
    final_by_ann_id: dict[str, dict[str, Any]] = {}
    for row in final_rows:
        aid = row.get("annotation_id")
        if not isinstance(aid, str) or not aid.strip():
            raise ValueError("Automated annotation record missing annotation_id")
        if aid in final_by_ann_id:
            raise ValueError(f"Duplicate automated annotation_id: {aid}")
        final_by_ann_id[aid] = row

    gt_by_profile: dict[str, list[dict[str, Any]]] = {}
    zero_positive_queries: dict[str, list[str]] = {PROFILE_A_ID: [], PROFILE_B_ID: []}
    grade_counts: dict[str, dict[int, int]] = {
        PROFILE_A_ID: {0: 0, 1: 0, 2: 0},
        PROFILE_B_ID: {0: 0, 1: 0, 2: 0},
    }
    source_counts: dict[str, dict[str, int]] = {PROFILE_A_ID: {}, PROFILE_B_ID: {}}

    expected_all_ids: set[str] = set()
    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        other_pid = PROFILE_B_ID if pid == PROFILE_A_ID else PROFILE_A_ID
        pool = pool_by_profile[pid]
        grades_by_qid: dict[str, dict[str, int]] = defaultdict(dict)

        for cand in pool:
            aid = cand["annotation_id"]
            expected_all_ids.add(aid)
            if aid not in final_by_ann_id:
                raise ValueError(f"Missing automated annotation for required candidate {aid}")
            ann = final_by_ann_id[aid]

            if ann.get("query_id") != cand["query_id"] or ann.get("profile_id") != pid or ann.get("chunk_id") != cand["chunk_id"]:
                raise ValueError(f"{aid}: metadata mismatch between automated annotation and candidate pool")
            cid = cand["chunk_id"]
            if cid not in corpus_ids_by_profile[pid]:
                if cid in corpus_ids_by_profile[other_pid]:
                    raise ValueError(f"{aid}: wrong-profile chunk_id {cid}")
                raise ValueError(f"{aid}: unknown chunk_id {cid}")

            grade = ann.get("final_grade")
            if isinstance(grade, bool) or type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError(f"{aid}: invalid final_grade {grade!r}")
            if cid in grades_by_qid[cand["query_id"]]:
                raise ValueError(f"{aid}: duplicate (query_id, chunk_id) in {pid}")

            grades_by_qid[cand["query_id"]][cid] = grade
            grade_counts[pid][grade] += 1
            src = ann.get("annotation_source", "AI-generated")
            source_counts[pid][src] = source_counts[pid].get(src, 0) + 1

        records: list[dict[str, Any]] = []
        judge_cfg = {
            "primary_judge_model": quality_audit.get("primary_judge_model"),
            "secondary_judge_model": quality_audit.get("secondary_judge_model"),
            "adjudicator_model": quality_audit.get("adjudicator_model"),
            "prompt_version": quality_audit.get("final_prompt_version"),
        }
        provenance = {
            "pipeline": "fully_automated_source_grounded_llm_as_a_judge",
            "human_validated": False,
            "final_annotations_sha256": sha256_bytes(final_annotations_path.read_bytes()),
            "quality_audit_sha256": sha256_bytes(quality_audit_path.read_bytes()),
            "random_audit_exact_agreement_pct": (quality_audit.get("random_audit_10pct") or {}).get("exact_agreement_pct"),
            "random_audit_quadratic_kappa": (quality_audit.get("random_audit_10pct") or {}).get("quadratic_weighted_kappa"),
        }

        for q in questions:
            qid = q["query_id"]
            q_grades = grades_by_qid.get(qid, {})
            relevant_ids = [cid for cid, g in q_grades.items() if g > 0]
            is_zero = len(relevant_ids) == 0
            if is_zero:
                zero_positive_queries[pid].append(qid)
            records.append({
                "query_id": qid,
                "query": q["query"],
                "profile_id": pid,
                "topic": q["topic"],
                "language": q["language"],
                "difficulty": q["difficulty"],
                "question_type": q["question_type"],
                "reference_answer": q["reference_answer"],
                "required_knowledge_units": q["required_knowledge_units"],
                "source_evidence_ids": q["source_evidence_ids"],
                "relevant_ids": relevant_ids,
                "relevance_grades": q_grades,
                "labels_complete": True,
                "judgment_scope": "judged_pool_complete_not_corpus_exhaustive",
                "judged_count": len(q_grades),
                "positive_count": len(relevant_ids),
                "zero_positive": is_zero,
                "corpus_sha256": corpus_sha_by_profile[pid],
                "question_dataset_version": dataset_version,
                "question_dataset_sha256": actual_q_sha,
                "annotation_method": "llm_generated_pooled_relevance_judgments",
                "human_validated": False,
                "annotation_provenance": provenance,
                "judge_configuration": judge_cfg,
            })
        gt_by_profile[pid] = records

    extra_ids = set(final_by_ann_id.keys()) - expected_all_ids
    if extra_ids:
        raise ValueError(f"Unexpected extra automated annotation IDs: {sorted(extra_ids)[:5]}")

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
        path.write_bytes(payload)
        return sha256_bytes(payload)

    written: dict[str, str] = {}
    if output_a_path is not None:
        written[PROFILE_A_ID] = write_jsonl(output_a_path, gt_by_profile[PROFILE_A_ID])
    if output_b_path is not None:
        written[PROFILE_B_ID] = write_jsonl(output_b_path, gt_by_profile[PROFILE_B_ID])

    return {
        "dataset_version": dataset_version,
        "question_count": len(questions),
        "annotation_method": "llm_generated_pooled_relevance_judgments",
        "human_validated": False,
        "ground_truth_recursive": gt_by_profile[PROFILE_A_ID],
        "ground_truth_adaptive": gt_by_profile[PROFILE_B_ID],
        "grade_counts": grade_counts,
        "source_counts": source_counts,
        "zero_positive_queries": zero_positive_queries,
        "written_sha256": written,
    }
