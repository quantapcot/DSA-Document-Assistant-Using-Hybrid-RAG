"""Shared schema, canonical source evidence verification, and manifest builder for Stage 03."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.corpus.provenance import (
    CANONICAL_COORDINATE_VERSION,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    CanonicalDocument,
    CanonicalPage,
    canonical_json,
    sha256_bytes,
    sha256_text,
)

STAGE03_DATASET_VERSION = "stage03_dsa_40q_v1"
STAGE03_POOLING_TOP_K = 60
STAGE03_CANDIDATE_POOL = 120
STAGE03_EVAL_KS = (1, 3, 5, 10, 15, 20, 30, 45, 60)
STAGE03_METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")

CANONICAL_FINGERPRINT_SHA256 = "712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d"
RAW_V1_AGGREGATE_SHA256 = "cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6"

FROZEN_CORPORA = {
    PROFILE_A_ID: {
        "corpus_id": "eval-recursive-512-v1",
        "chunk_count": 6366,
        "corpus_sha256": "e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649",
        "rel_chunks_path": "evaluation/artifacts/recursive_512_100/chunks.jsonl",
        "rel_manifest_path": "evaluation/artifacts/recursive_512_100/manifest.json",
    },
    PROFILE_B_ID: {
        "corpus_id": "eval-structure-adaptive-v1",
        "chunk_count": 13390,
        "corpus_sha256": "81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63",
        "rel_chunks_path": "evaluation/artifacts/structure_adaptive/chunks.jsonl",
        "rel_manifest_path": "evaluation/artifacts/structure_adaptive/manifest.json",
    },
}

VALID_TOPICS = (
    "Array & Linked List",
    "Stack & Queue",
    "Tree, BST & AVL",
    "Graph Algorithms",
    "Searching & Sorting",
    "Time & Space Complexity",
    "Hashing",
    "Algorithm Design Techniques",
)
VALID_LANGUAGES = ("vi", "en")
VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_QUESTION_TYPES = (
    "definition",
    "concept_explanation",
    "algorithm_steps",
    "comparison",
    "time_complexity",
    "space_complexity",
    "code_interpretation",
    "pseudocode_interpretation",
    "application_scenario",
    "multi_evidence_reasoning",
)
VALID_APPROVAL_STATUSES = ("DRAFT", "APPROVED", "NEEDS_REVISION")
VALID_EVIDENCE_MODES = ("single", "alternative", "composite")


def compute_canonical_fingerprint(docs: list[CanonicalDocument]) -> str:
    payload = [
        {
            "doc_id": d.document_id,
            "rev": d.source_revision,
            "pages": [
                {
                    "page": p.page,
                    "lang": p.language,
                    "hash": sha256_text(p.canonical_text),
                    "blocks": [(b.block_id, b.raw_kind, b.char_start, b.char_end) for b in p.blocks],
                }
                for p in d.pages
            ],
        }
        for d in docs
    ]
    return sha256_text(canonical_json(payload))


def locate_canonical_spans(
    page: CanonicalPage,
    snippet: str,
    *,
    occurrence: int = 0,
) -> tuple[list[dict[str, Any]], str]:
    """Locate *snippet* verbatim in *page.canonical_text* and return exact block-aligned SourceSpans."""
    text = page.canonical_text
    if not snippet or not snippet.strip():
        raise ValueError(f"Empty evidence snippet for doc={page.document_id} page={page.page}")

    search_from = 0
    found = -1
    for _ in range(occurrence + 1):
        found = text.find(snippet, search_from)
        if found == -1:
            raise ValueError(
                f"Evidence snippet not found verbatim on doc={page.document_id} page={page.page} "
                f"(occurrence={occurrence}): {snippet[:90]!r}"
            )
        search_from = found + 1

    start = found
    end = found + len(snippet)
    spans: list[dict[str, Any]] = []
    for blk in page.blocks:
        os_ = max(start, blk.char_start)
        oe_ = min(end, blk.char_end)
        if oe_ > os_ and text[os_:oe_].strip():
            sub = text[os_:oe_]
            lead = len(sub) - len(sub.lstrip())
            trail = len(sub) - len(sub.rstrip())
            spans.append({
                "page": page.page,
                "block_id": blk.block_id,
                "char_start": os_ + lead,
                "char_end": oe_ - trail,
            })
    if not spans:
        spans.append({
            "page": page.page,
            "block_id": f"{page.document_id}:p{page.page}:raw",
            "char_start": start,
            "char_end": end,
        })
    return spans, text[start:end]


def extract_chunk_source_backed_intervals(chunk: dict[str, Any]) -> list[tuple[int, int, int, str]]:
    """Return all source-backed (page, char_start, char_end, kind) intervals of a frozen chunk.

    Includes original source_spans, coordinate-backed overlap_segments, and source_backed
    synthetic_segments (heading_prefix, table_header, code_signature). Excludes non-source-backed text.
    """
    meta = chunk.get("metadata") or {}
    intervals: list[tuple[int, int, int, str]] = []
    default_page = chunk.get("page")

    for sp in meta.get("source_spans") or []:
        cs, ce = sp.get("char_start"), sp.get("char_end")
        pg = sp.get("page", default_page)
        if type(pg) is int and type(cs) is int and type(ce) is int and ce > cs:
            intervals.append((pg, cs, ce, "source_span"))

    for ov in meta.get("overlap_segments") or []:
        cs, ce = ov.get("char_start"), ov.get("char_end")
        pg = ov.get("page", default_page)
        if type(pg) is int and type(cs) is int and type(ce) is int and ce > cs:
            intervals.append((pg, cs, ce, "overlap_segment"))

    for syn in meta.get("synthetic_segments") or []:
        if not syn.get("source_backed"):
            continue
        cs, ce = syn.get("char_start"), syn.get("char_end")
        pg = syn.get("page", default_page)
        if type(pg) is int and type(cs) is int and type(ce) is int and ce > cs:
            intervals.append((pg, cs, ce, f"synthetic:{syn.get('kind', 'unknown')}"))

    return intervals


def evidence_overlap_threshold(span_len: int) -> int:
    """Uniform minimum character overlap threshold applied identically across Profile A and Profile B."""
    return min(20, max(8, int(0.25 * span_len)))


def chunk_matches_evidence(chunk: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Check whether a frozen chunk's source-backed segments substantially overlap *evidence*."""
    if chunk.get("document_id") != evidence.get("document_id"):
        return False
    ev_page = evidence.get("canonical_page")
    ev_spans = evidence.get("source_spans") or []
    chunk_ivs = extract_chunk_source_backed_intervals(chunk)
    if not chunk_ivs or not ev_spans:
        return False

    for esp in ev_spans:
        es, ee = esp["char_start"], esp["char_end"]
        thresh = evidence_overlap_threshold(ee - es)
        for c_page, cs, ce, _ in chunk_ivs:
            if c_page != ev_page:
                continue
            ov = min(ce, ee) - max(cs, es)
            if ov >= thresh:
                return True
    return False


def validate_dataset_and_evidence(
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    docs: list[CanonicalDocument],
    *,
    expected_question_count: int | None = 40,
    require_draft_status: bool = True,
) -> dict[str, Any]:
    """Validate 40-question dataset and canonical source evidence against live canonical documents."""
    errors: list[str] = []
    doc_by_id = {d.document_id: d for d in docs}
    page_by_key = {(d.document_id, p.page): p for d in docs for p in d.pages}

    if expected_question_count is not None and len(questions) != expected_question_count:
        errors.append(f"Expected {expected_question_count} questions, found {len(questions)}")

    seen_qids: set[str] = set()
    seen_qtexts: set[str] = set()
    ev_by_id: dict[str, dict[str, Any]] = {}
    for ev in evidences:
        eid = ev.get("evidence_id")
        if not isinstance(eid, str) or not eid.strip():
            errors.append("Evidence missing evidence_id")
            continue
        if eid in ev_by_id:
            errors.append(f"Duplicate evidence_id: {eid}")
            continue
        ev_by_id[eid] = ev

    # Validate each evidence record against canonical documents
    for eid, ev in ev_by_id.items():
        did = ev.get("document_id")
        doc = doc_by_id.get(did)
        if doc is None:
            errors.append(f"{eid}: unknown document_id {did!r}")
            continue
        if ev.get("source_revision") != doc.source_revision:
            errors.append(f"{eid}: source_revision mismatch")
        if ev.get("source_path") != doc.source_path:
            errors.append(f"{eid}: source_path mismatch ({ev.get('source_path')!r} != {doc.source_path!r})")
        if ev.get("evidence_mode", "single") not in VALID_EVIDENCE_MODES:
            errors.append(f"{eid}: invalid evidence_mode {ev.get('evidence_mode')!r}")

        pg = ev.get("canonical_page")
        page_obj = page_by_key.get((did, pg))
        if page_obj is None:
            errors.append(f"{eid}: invalid canonical_page {pg} for {doc.source_path}")
            continue

        spans = ev.get("source_spans") or []
        ev_text = ev.get("evidence_text", "")
        if not spans or not isinstance(ev_text, str) or not ev_text.strip():
            errors.append(f"{eid}: empty source_spans or evidence_text")
            continue

        canonical_text = page_obj.canonical_text
        page_len = len(canonical_text)
        for sp in spans:
            cs, ce = sp.get("char_start"), sp.get("char_end")
            if sp.get("page") != pg:
                errors.append(f"{eid}: span page mismatch {sp}")
                continue
            if type(cs) is not int or type(ce) is not int or cs < 0 or ce <= cs or ce > page_len:
                errors.append(f"{eid}: out-of-bounds span {sp} for page_len={page_len}")
                continue
            sub = canonical_text[cs:ce]
            if not sub.strip():
                errors.append(f"{eid}: blank span {sp}")
            elif sub not in ev_text and " ".join(sub.split()) not in " ".join(ev_text.split()):
                errors.append(f"{eid}: canonical span text does not match evidence_text")

        if ev_text not in canonical_text and " ".join(ev_text.split()) not in " ".join(canonical_text.split()):
            errors.append(f"{eid}: evidence_text not found on canonical page {pg}")

    # Validate each question
    topic_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    diff_counts: Counter[str] = Counter()
    qtype_counts: Counter[str] = Counter()
    total_kus = 0

    for q in questions:
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid.strip():
            errors.append("Question missing query_id")
            continue
        if qid in seen_qids:
            errors.append(f"Duplicate query_id: {qid}")
        seen_qids.add(qid)

        qtext = (q.get("query") or "").strip()
        if not qtext:
            errors.append(f"{qid}: empty query text")
        norm_q = " ".join(qtext.lower().split())
        if norm_q in seen_qtexts:
            errors.append(f"{qid}: duplicate normalized query text")
        seen_qtexts.add(norm_q)

        if q.get("topic") not in VALID_TOPICS:
            errors.append(f"{qid}: invalid topic {q.get('topic')!r}")
        if q.get("language") not in VALID_LANGUAGES:
            errors.append(f"{qid}: invalid language {q.get('language')!r}")
        if q.get("difficulty") not in VALID_DIFFICULTIES:
            errors.append(f"{qid}: invalid difficulty {q.get('difficulty')!r}")
        if q.get("question_type") not in VALID_QUESTION_TYPES:
            errors.append(f"{qid}: invalid question_type {q.get('question_type')!r}")

        status = q.get("human_approval_status")
        if status not in VALID_APPROVAL_STATUSES:
            errors.append(f"{qid}: invalid human_approval_status {status!r}")
        if require_draft_status and status != "DRAFT":
            errors.append(f"{qid}: initial human_approval_status must be DRAFT, got {status!r}")

        ref_ans = (q.get("reference_answer") or "").strip()
        if not ref_ans:
            errors.append(f"{qid}: empty reference_answer")

        kus = q.get("required_knowledge_units") or []
        if not kus:
            errors.append(f"{qid}: empty required_knowledge_units")
        ku_ids = set()
        for ku in kus:
            kid = ku.get("ku_id")
            desc = (ku.get("description") or "").strip()
            if not kid or not desc:
                errors.append(f"{qid}: invalid KU entry {ku}")
            ku_ids.add(kid)
        total_kus += len(kus)

        ev_ids = q.get("source_evidence_ids") or []
        if not ev_ids:
            errors.append(f"{qid}: empty source_evidence_ids")
        covered_kus: set[str] = set()
        for eid in ev_ids:
            ev = ev_by_id.get(eid)
            if ev is None:
                errors.append(f"{qid}: unknown evidence_id {eid!r}")
                continue
            if ev.get("query_id") != qid:
                errors.append(f"{qid}: evidence {eid} belongs to {ev.get('query_id')!r}")
            covered_kus.add(ev.get("knowledge_unit_id"))

        missing_ku_ev = ku_ids - covered_kus
        if missing_ku_ev:
            errors.append(f"{qid}: KUs missing source evidence: {sorted(missing_ku_ev)}")

        if not (q.get("source_document_references") or []):
            errors.append(f"{qid}: empty source_document_references")

        topic_counts[q.get("topic", "unknown")] += 1
        lang_counts[q.get("language", "unknown")] += 1
        diff_counts[q.get("difficulty", "unknown")] += 1
        qtype_counts[q.get("question_type", "unknown")] += 1

    if errors:
        raise ValueError(f"Dataset/evidence validation failed with {len(errors)} errors:\n" + "\n".join(errors[:25]))

    return {
        "question_count": len(questions),
        "evidence_count": len(evidences),
        "total_knowledge_units": total_kus,
        "topic_distribution": dict(topic_counts),
        "language_distribution": dict(lang_counts),
        "difficulty_distribution": dict(diff_counts),
        "question_type_distribution": dict(qtype_counts),
    }


def check_ku_retention_in_corpora(
    questions: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    corpus_rows_by_profile: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Check whether each required KU's source evidence is retained in Frozen Corpus A & Profile B.

    Records gaps transparently without dropping any KU or question.
    """
    ev_by_id = {ev["evidence_id"]: ev for ev in evidences}
    # Index chunks by (profile_id, document_id, page) for fast overlap lookup
    chunks_index: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for pid, rows in corpus_rows_by_profile.items():
        for ch in rows:
            did = ch["document_id"]
            pages = {iv[0] for iv in extract_chunk_source_backed_intervals(ch)}
            for pg in pages:
                chunks_index[(pid, did, pg)].append(ch)

    summary: dict[str, Any] = {}
    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        total_kus = 0
        retained_kus = 0
        missing_kus: list[dict[str, Any]] = []
        for q in questions:
            qid = q["query_id"]
            q_evs = [ev_by_id[eid] for eid in q["source_evidence_ids"]]
            evs_by_ku: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for ev in q_evs:
                evs_by_ku[ev["knowledge_unit_id"]].append(ev)

            for ku in q["required_knowledge_units"]:
                total_kus += 1
                kid = ku["ku_id"]
                ku_evs = evs_by_ku.get(kid, [])
                mode = ku.get("evidence_mode", "single")

                def ev_covered(ev: dict[str, Any]) -> bool:
                    cands = chunks_index.get((pid, ev["document_id"], ev["canonical_page"]), [])
                    return any(chunk_matches_evidence(c, ev) for c in cands)

                if mode == "composite":
                    ok = bool(ku_evs) and all(ev_covered(ev) for ev in ku_evs)
                else:
                    # single or alternative: at least one evidence alternative covers the KU
                    ok = any(ev_covered(ev) for ev in ku_evs)
                if ok:
                    retained_kus += 1
                else:
                    missing_kus.append({"query_id": qid, "ku_id": kid, "mode": mode})

        summary[pid] = {
            "total_kus": total_kus,
            "retained_kus": retained_kus,
            "missing_ku_count": len(missing_kus),
            "missing_kus": missing_kus,
        }
    return summary


def build_dataset_manifest(
    *,
    questions_path: Path,
    evidence_path: Path,
    stats: dict[str, Any],
    ku_retention: dict[str, Any],
    canonical_fingerprint: str,
    retrieval_config_dict: dict[str, Any],
    hardware_info: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Build deterministic dataset_manifest.json for Stage 03 reproducibility."""
    q_sha = sha256_bytes(questions_path.read_bytes())
    ev_sha = sha256_bytes(evidence_path.read_bytes())
    return {
        "dataset_version": STAGE03_DATASET_VERSION,
        "python_version": platform.python_version(),
        "canonical_coordinate_version": CANONICAL_COORDINATE_VERSION,
        "canonical_fingerprint_sha256": canonical_fingerprint,
        "raw_aggregate_sha256": RAW_V1_AGGREGATE_SHA256,
        "files": {
            "questions_jsonl": questions_path.relative_to(root).as_posix() if questions_path.is_relative_to(root) else str(questions_path),
            "questions_sha256": q_sha,
            "source_evidence_jsonl": evidence_path.relative_to(root).as_posix() if evidence_path.is_relative_to(root) else str(evidence_path),
            "source_evidence_sha256": ev_sha,
        },
        "summary": stats,
        "ku_retention_in_frozen_corpora": ku_retention,
        "frozen_corpora": FROZEN_CORPORA,
        "experimental_protocol": {
            "shared_questions": 40,
            "profiles": [PROFILE_A_ID, PROFILE_B_ID],
            "retrieval_methods": list(STAGE03_METHODS),
            "pooling_top_k": STAGE03_POOLING_TOP_K,
            "candidate_pool": STAGE03_CANDIDATE_POOL,
            "evaluation_ks": list(STAGE03_EVAL_KS),
            "relevance_scale": [0, 1, 2],
            "blind_order_seed": "sha256(stage03_blind|query_id|profile_id|chunk_id)",
            "evidence_augmentation_policy": "source_backed_overlap >= min(20, max(8, floor(0.25 * evidence_span_len)))",
        },
        "models": {
            "embedding_model": EMBEDDING_MODEL_ID,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
            "embedding_max_seq_length": EMBEDDING_MAX_SEQ_LENGTH,
            "reranker_model": RERANKER_MODEL_ID,
            "reranker_revision": RERANKER_MODEL_REVISION,
            "reranker_max_length": RERANKER_MAX_LENGTH,
        },
        "retrieval_config": retrieval_config_dict,
        "hardware": hardware_info,
    }
