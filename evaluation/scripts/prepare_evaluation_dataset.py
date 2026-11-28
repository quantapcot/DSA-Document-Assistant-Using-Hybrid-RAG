"""CLI: Build and verify Stage 03 40 shared questions, canonical source evidence, and dataset manifest."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import torch
from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.benchmark_builder import build_eval_retrieval_config
from evaluation.corpus.shared_preprocessor import preprocess_raw_dataset
from evaluation.stage03.dataset_schema import (
    CANONICAL_FINGERPRINT_SHA256,
    FROZEN_CORPORA,
    STAGE03_CANDIDATE_POOL,
    build_dataset_manifest,
    check_ku_retention_in_corpora,
    compute_canonical_fingerprint,
    locate_canonical_spans,
    validate_dataset_and_evidence,
)
from evaluation.stage03.question_bank_40 import QUESTION_SPECS_40


def build_stage03_dataset(
    *,
    raw_dir: Path | None = None,
    datasets_dir: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    raw_root = raw_dir or (root / "data" / "corpus" / "dataset_raw")
    out_dir = datasets_dir or (root / "evaluation" / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical documents in memory and verify immutable canonical fingerprint
    docs = preprocess_raw_dataset(raw_root)
    can_fp = compute_canonical_fingerprint(docs)
    if can_fp != CANONICAL_FINGERPRINT_SHA256:
        raise RuntimeError(f"Canonical fingerprint drift: {can_fp} != {CANONICAL_FINGERPRINT_SHA256}")

    doc_by_path = {d.source_path: d for d in docs}
    page_by_path_num = {(d.source_path, p.page): p for d in docs for p in d.pages}

    questions: list[dict[str, Any]] = []
    evidences: list[dict[str, Any]] = []

    for spec in QUESTION_SPECS_40:
        qid = spec["query_id"]
        ev_ids: list[str] = []
        src_refs_ordered: list[str] = []
        src_refs_seen: set[str] = set()

        for e_idx, ev_spec in enumerate(spec["evidences"], start=1):
            sp = ev_spec["source_path"]
            pg = ev_spec["page"]
            doc = doc_by_path.get(sp)
            if doc is None:
                raise ValueError(f"{qid}: unknown source_path {sp!r}")
            page_obj = page_by_path_num.get((sp, pg))
            if page_obj is None:
                raise ValueError(f"{qid}: unknown page {pg} in {sp!r}")

            spans, exact_text = locate_canonical_spans(
                page_obj,
                ev_spec["snippet"],
                occurrence=ev_spec.get("occurrence", 0),
            )
            eid = f"{qid}_E{e_idx:02d}"
            ev_ids.append(eid)
            ref_label = f"{sp} (p.{pg})"
            if ref_label not in src_refs_seen:
                src_refs_seen.add(ref_label)
                src_refs_ordered.append(ref_label)

            evidences.append({
                "evidence_id": eid,
                "query_id": qid,
                "knowledge_unit_id": ev_spec["ku_id"],
                "evidence_mode": ev_spec.get("mode", "single"),
                "alternative_group_id": ev_spec.get("alt_group"),
                "document_id": doc.document_id,
                "source_revision": doc.source_revision,
                "source_path": doc.source_path,
                "canonical_page": pg,
                "source_spans": spans,
                "evidence_text": exact_text,
            })

        questions.append({
            "query_id": qid,
            "query": spec["query"],
            "topic": spec["topic"],
            "language": spec["language"],
            "difficulty": spec["difficulty"],
            "question_type": spec["question_type"],
            "reference_answer": spec["reference_answer"],
            "required_knowledge_units": spec["required_knowledge_units"],
            "source_evidence_ids": ev_ids,
            "source_document_references": src_refs_ordered,
            "human_approval_status": "DRAFT",  # Never pre-approved
        })

    stats = validate_dataset_and_evidence(
        questions,
        evidences,
        docs,
        expected_question_count=40,
        require_draft_status=True,
    )

    # Check KU retention in Frozen Corpus A & Frozen Corpus B (transparent reporting, zero KU dropping)
    corpus_rows_by_profile = {}
    for pid, fspec in FROZEN_CORPORA.items():
        cpath = root / fspec["rel_chunks_path"]
        corpus_rows_by_profile[pid] = [
            json.loads(ln) for ln in cpath.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    ku_retention = check_ku_retention_in_corpora(questions, evidences, corpus_rows_by_profile)

    q_path = out_dir / "questions_40.jsonl"
    ev_path = out_dir / "source_evidence.jsonl"
    q_path.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")
    ev_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in evidences) + "\n", encoding="utf-8")

    cfg = build_eval_retrieval_config(candidate_top_k=STAGE03_CANDIDATE_POOL)
    cfg_dict = {k: v for k, v in asdict(cfg).items() if k not in {"qdrant_path", "qdrant_url"}}
    hw = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    manifest = build_dataset_manifest(
        questions_path=q_path,
        evidence_path=ev_path,
        stats=stats,
        ku_retention=ku_retention,
        canonical_fingerprint=can_fp,
        retrieval_config_dict=cfg_dict,
        hardware_info=hw,
        root=root,
    )
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "questions_path": str(q_path),
        "evidence_path": str(ev_path),
        "manifest_path": str(manifest_path),
        "canonical_fingerprint_sha256": can_fp,
        "summary": stats,
        "ku_retention_in_frozen_corpora": ku_retention,
    }


def main() -> int:
    result = build_stage03_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
