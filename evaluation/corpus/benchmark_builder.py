"""Dual benchmark corpus builder, diagnostics calculator, and isolated Qdrant indexer."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index
from app.retrieval.vector_store import check_model_exhaustive, get_client, scope_filter

from evaluation.chunking.profiles import (
    PROFILE_A_ID,
    PROFILE_B_ID,
    get_profile_config,
    get_profile_config_hash,
)
from evaluation.chunking.recursive_profile import chunk_document_recursive
from evaluation.chunking.structure_adaptive import chunk_document_adaptive, load_embedding_tokenizer
from evaluation.corpus.provenance import (
    CanonicalDocument,
    build_benchmark_manifest,
    build_raw_inventory,
    canonical_json,
    interval_union_length,
    sha256_bytes,
    sha256_text,
    validate_chunk_provenance,
)
from evaluation.corpus.shared_preprocessor import PREPROCESSOR_VERSION, preprocess_raw_dataset

EVAL_WORKSPACE_ID = "evaluation"
EVAL_COLLECTION = "eval_dsa_chunks_v1"
SCOPE_BY_PROFILE = {
    PROFILE_A_ID: "eval-recursive-512-v1",
    PROFILE_B_ID: "eval-structure-adaptive-v1",
}


def _numeric_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    s = sorted(values)
    p95 = quantiles(s, n=20, method="inclusive")[-1] if len(s) > 1 else float(s[0])
    return {
        "min": s[0],
        "mean": round(mean(s), 2),
        "median": round(median(s), 2),
        "p95": round(p95, 2),
        "max": s[-1],
    }


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    valid = sorted((max(0, s), max(0, e)) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for s, e in valid[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            if e > pe:
                merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _interval_difference(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return exact interval set A \\ B (both inputs should be merged & sorted)."""
    out: list[tuple[int, int]] = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def _interval_intersection(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return exact interval set A ∩ B (both inputs should be merged & sorted)."""
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def compute_profile_diagnostics(
    profile_id: str,
    docs: list[CanonicalDocument],
    chunks: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute complete corpus diagnostics including exact non-double-counted canonical coverage reconciliation."""
    page_lookup = {(d.document_id, p.page): p for d in docs for p in d.pages}
    total_canonical_chars = sum(len(p.canonical_text) for d in docs for p in d.pages)
    dropped_dups = sum(p.dropped_duplicate_records for d in docs for p in d.pages)

    # Interval union on original source_spans & source_backed synthetic_segments per (document_id, page)
    intervals_by_page: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    repeated_ctx_by_page: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    provenance_errors = 0
    chunk_ids: list[str] = []
    content_hashes: dict[str, set[str]] = defaultdict(set)
    content_hash_locs: dict[str, set[tuple[str, int, int, int]]] = defaultdict(set)
    overlap_chars = 0
    synthetic_chars = 0
    total_chunk_chars = 0
    char_lens: list[int] = []
    emb_lens: list[int] = []
    ctype_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    forced_splits = 0

    for c in chunks:
        chunk_ids.append(c["chunk_id"])
        meta = c["metadata"]
        page_obj = page_lookup.get((c["document_id"], c["page"]))
        if page_obj is None or validate_chunk_provenance(c, page_obj):
            provenance_errors += 1

        char_lens.append(len(c["text"]))
        total_chunk_chars += len(c["text"])
        emb_lens.append(int(meta["emb_token_count"]))
        ctype_counts[meta.get("content_type", "unknown")] += 1

        fb = meta.get("fallback") or {}
        if fb.get("is_forced_split"):
            forced_splits += 1
        if fb.get("reason"):
            fallback_counts[fb["reason"]] += 1

        spans_list = meta.get("source_spans", [])
        for sp in spans_list:
            intervals_by_page[(c["document_id"], sp["page"])].append((sp["char_start"], sp["char_end"]))
        if spans_list:
            content_hash_locs[meta["content_hash"]].add(
                (c["document_id"], spans_list[0]["page"], spans_list[0]["char_start"], spans_list[-1]["char_end"])
            )
        content_hashes[meta["content_hash"]].add(c["document_id"])

        overlap_chars += sum(int(o.get("text_len", 0)) for o in meta.get("overlap_segments", []))
        for syn in meta.get("synthetic_segments", []):
            synthetic_chars += int(syn.get("text_len", 0))
            if syn.get("source_backed") and syn.get("char_start") is not None and syn.get("char_end") is not None:
                syn_page = syn.get("page", c["page"])
                repeated_ctx_by_page[(c["document_id"], syn_page)].append((syn["char_start"], syn["char_end"]))

    drop_reasons: Counter[str] = Counter(d["reason"] for d in dropped)
    dropped_intervals: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for d in dropped:
        dropped_intervals[(d["document_id"], d["page"])].append((d["char_start"], d["char_end"]))

    # Exact non-double-counted canonical partition across all pages
    retained_chars = 0
    repeated_ctx_total_chars = 0
    repeated_ctx_only_chars = 0
    dropped_chars_union = 0
    dropped_and_uncovered_chars = 0
    dropped_but_covered_chars = 0
    uncovered_whitespace_chars = 0
    uncovered_other_meaningful_chars = 0

    for key, page_obj in page_lookup.items():
        text = page_obj.canonical_text
        page_len = len(text)
        if page_len == 0:
            continue
        cov = _merge_intervals(intervals_by_page.get(key, []))
        rep = _merge_intervals(repeated_ctx_by_page.get(key, []))
        drp = _merge_intervals(dropped_intervals.get(key, []))

        rep_only = _interval_difference(rep, cov)
        kept_all = _merge_intervals([*cov, *rep_only])
        drp_uncov = _interval_difference(drp, kept_all)
        drp_cov = _interval_intersection(drp, kept_all)
        accounted = _merge_intervals([*kept_all, *drp_uncov])
        rem_uncov = _interval_difference([(0, page_len)], accounted)

        retained_chars += sum(e - s for s, e in cov)
        repeated_ctx_total_chars += sum(e - s for s, e in rep)
        repeated_ctx_only_chars += sum(e - s for s, e in rep_only)
        dropped_chars_union += sum(e - s for s, e in drp)
        dropped_and_uncovered_chars += sum(e - s for s, e in drp_uncov)
        dropped_but_covered_chars += sum(e - s for s, e in drp_cov)

        for s, e in rem_uncov:
            seg = text[s:e]
            ws_cnt = sum(1 for ch in seg if ch.isspace())
            uncovered_whitespace_chars += ws_cnt
            uncovered_other_meaningful_chars += len(seg) - ws_cnt

    partition_sum = (
        retained_chars
        + repeated_ctx_only_chars
        + dropped_and_uncovered_chars
        + uncovered_whitespace_chars
        + uncovered_other_meaningful_chars
    )

    trunc_count = sum(1 for v in emb_lens if v > 128)
    same_doc_repeated_groups = sum(
        1 for ch, locs in content_hash_locs.items()
        if len(locs) > 1 and len({loc[0] for loc in locs}) == 1
    )
    cross_doc_repeated_groups = sum(
        1 for ch, doc_set in content_hashes.items() if len(doc_set) > 1
    )

    return {
        "profile_id": profile_id,
        "document_count": len(docs),
        "document_id_collisions": len(docs) - len({d.document_id for d in docs}),
        "chunk_count": len(chunks),
        "duplicate_chunk_ids": len(chunk_ids) - len(set(chunk_ids)),
        "missing_provenance_count": provenance_errors,
        "char_length": _numeric_stats(char_lens),
        "embedding_token_length": _numeric_stats(emb_lens),
        "embedding_truncation_count": trunc_count,
        "embedding_truncation_rate_pct": round(100.0 * trunc_count / max(1, len(chunks)), 2),
        "canonical_total_chars": total_canonical_chars,
        "canonical_retained_chars_union": retained_chars,
        "canonical_source_coverage_pct": round(100.0 * retained_chars / max(1, total_canonical_chars), 2),
        "repeated_context_source_chars_total": repeated_ctx_total_chars,
        "repeated_context_only_chars_union": repeated_ctx_only_chars,
        "canonical_effective_coverage_pct": round(
            100.0 * (retained_chars + repeated_ctx_only_chars) / max(1, total_canonical_chars), 2
        ),
        "dropped_spans_count": len(dropped),
        "dropped_chars_union": dropped_chars_union,
        "dropped_and_uncovered_chars": dropped_and_uncovered_chars,
        "dropped_but_covered_chars": dropped_but_covered_chars,
        "uncovered_whitespace_chars": uncovered_whitespace_chars,
        "uncovered_other_meaningful_chars": uncovered_other_meaningful_chars,
        "coverage_partition_exact_match": partition_sum == total_canonical_chars,
        "dropped_reasons": dict(drop_reasons),
        "dropped_duplicate_records_in_preprocessing": dropped_dups,
        "overlap_chars_total": overlap_chars,
        "overlap_redundancy_pct": round(100.0 * overlap_chars / max(1, total_chunk_chars), 2),
        "synthetic_prefix_chars_total": synthetic_chars,
        "synthetic_prefix_pct": round(100.0 * synthetic_chars / max(1, total_chunk_chars), 2),
        "content_type_distribution": dict(sorted(ctype_counts.items())),
        "forced_split_count": forced_splits,
        "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        "repeated_content_groups_same_doc": same_doc_repeated_groups,
        "repeated_content_groups_cross_doc": cross_doc_repeated_groups,
    }


def _canonical_fingerprint(docs: list[CanonicalDocument]) -> str:
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


def build_dual_corpora(
    *,
    raw_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    verify_determinism: bool = False,
    emb_tokenizer: Any = None,
) -> dict[str, Any]:
    """Preprocess raw dataset once (twice if verify_determinism=True), build Corpus A & B, write manifests & diagnostics."""
    raw_root = raw_dir or (ROOT / "data" / "corpus" / "dataset_raw")
    out_root = artifacts_dir or (ROOT / "evaluation" / "artifacts")
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = emb_tokenizer or load_embedding_tokenizer()
    raw_inventory, raw_agg = build_raw_inventory(raw_root)

    def run_once() -> tuple[str, list[CanonicalDocument], dict[str, dict[str, Any]]]:
        docs = preprocess_raw_dataset(raw_root)
        can_fp = _canonical_fingerprint(docs)
        profile_outputs: dict[str, dict[str, Any]] = {}
        for profile_id, fn in (
            (PROFILE_A_ID, chunk_document_recursive),
            (PROFILE_B_ID, chunk_document_adaptive),
        ):
            all_chunks: list[dict[str, Any]] = []
            all_dropped: list[dict[str, Any]] = []
            for d in docs:
                ch, dr = fn(d, tokenizer)
                all_chunks.extend(ch)
                all_dropped.extend(dr)
            diag = compute_profile_diagnostics(profile_id, docs, all_chunks, all_dropped)
            profile_outputs[profile_id] = {
                "chunks": all_chunks,
                "dropped": all_dropped,
                "diagnostics": diag,
            }
        return can_fp, docs, profile_outputs

    can_fp_1, docs_1, run1 = run_once()
    determinism_verified = False
    if verify_determinism:
        can_fp_2, _, run2 = run_once()
        assert can_fp_1 == can_fp_2, "Canonical preprocessing is not deterministic!"
        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            assert canonical_json(run1[pid]["chunks"]) == canonical_json(run2[pid]["chunks"]), (
                f"Chunk determinism check failed for {pid}"
            )
            assert canonical_json(run1[pid]["diagnostics"]) == canonical_json(run2[pid]["diagnostics"]), (
                f"Diagnostics determinism check failed for {pid}"
            )
        determinism_verified = True

    preproc_sha = sha256_bytes((ROOT / "evaluation" / "corpus" / "shared_preprocessor.py").read_bytes())
    summary: dict[str, Any] = {
        "raw_file_count": len(raw_inventory),
        "raw_aggregate_sha256": raw_agg,
        "canonical_fingerprint_sha256": can_fp_1,
        "determinism_verified_two_runs": determinism_verified,
        "profiles": {},
    }

    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        pdir = out_root / pid
        pdir.mkdir(parents=True, exist_ok=True)
        chunks_path = pdir / "chunks.jsonl"
        jsonl_bytes = ("\n".join(json.dumps(c, ensure_ascii=False) for c in run1[pid]["chunks"]) + "\n").encode("utf-8")
        chunks_path.write_bytes(jsonl_bytes)
        corpus_sha = sha256_bytes(jsonl_bytes)

        manifest = build_benchmark_manifest(
            raw_inventory=raw_inventory,
            raw_aggregate_sha256=raw_agg,
            preprocessor_info={
                "version": PREPROCESSOR_VERSION,
                "shared_preprocessor_sha256": preproc_sha,
                "docx_tables_enabled": True,
                "code_indent_guard": True,
                "language_detector": "deterministic_rule_based_v1",
                "canonical_fingerprint_sha256": can_fp_1,
            },
            profile_id=pid,
            chunking_config=get_profile_config(pid),
            chunking_config_hash=get_profile_config_hash(pid),
            chunks=run1[pid]["chunks"],
            corpus_jsonl_relpath=chunks_path.relative_to(ROOT).as_posix() if chunks_path.is_relative_to(ROOT) else str(chunks_path),
            corpus_sha256=corpus_sha,
            qdrant_scope={
                "qdrant_path": "data/runtime/qdrant_eval",
                "collection": EVAL_COLLECTION,
                "workspace_id": EVAL_WORKSPACE_ID,
                "corpus_id": SCOPE_BY_PROFILE[pid],
            },
        )
        manifest_path = pdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["profiles"][pid] = {
            "chunks_jsonl": str(chunks_path),
            "manifest_json": str(manifest_path),
            "corpus_sha256": corpus_sha,
            "diagnostics": run1[pid]["diagnostics"],
        }

    diag_path = out_root / "diagnostics.json"
    diag_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["diagnostics_json"] = str(diag_path)
    return summary


def build_eval_retrieval_config(
    *,
    base_config_path: Path | None = None,
    qdrant_path: Path | None = None,
    collection: str = EVAL_COLLECTION,
    candidate_top_k: int | None = None,
) -> RetrievalConfig:
    """Load production RetrievalConfig and override only isolated benchmark storage & pool."""
    cfg_path = base_config_path or (ROOT / "configs" / "retrieval.mvp.yaml")
    base = RetrievalConfig.from_yaml(cfg_path)
    eval_storage = str((qdrant_path or (ROOT / "data" / "runtime" / "qdrant_eval")).resolve())
    overrides: dict[str, Any] = {
        "qdrant_mode": "local",
        "qdrant_path": eval_storage,
        "collection": collection,
    }
    if candidate_top_k is not None:
        overrides["candidate_top_k"] = candidate_top_k
    return replace(base, **overrides)


def index_benchmark_corpora(
    *,
    artifacts_dir: Path | None = None,
    qdrant_path: Path | None = None,
    base_config_path: Path | None = None,
    rebuild: bool = True,
    client: Any = None,
) -> dict[str, Any]:
    """Index Corpus A and Corpus B into isolated data/runtime/qdrant_eval/ without touching production."""
    from app.retrieval import vector_store

    art_root = artifacts_dir or (ROOT / "evaluation" / "artifacts")
    eval_storage = qdrant_path or (ROOT / "data" / "runtime" / "qdrant_eval")
    eval_storage.mkdir(parents=True, exist_ok=True)
    cfg = build_eval_retrieval_config(base_config_path=base_config_path, qdrant_path=eval_storage)

    prod_qdrant_dir = (ROOT / "data" / "runtime" / "qdrant_local").resolve()
    assert Path(cfg.qdrant_path).resolve() != prod_qdrant_dir, "Refusing to index into production Qdrant storage!"

    owns_client = client is None
    active_client = client or get_client(config=cfg)
    try:
        results: dict[str, Any] = {"qdrant_path": cfg.qdrant_path, "collection": cfg.collection, "scopes": {}}
        expected_ids: dict[str, set[str]] = {}

        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            chunks_path = art_root / pid / "chunks.jsonl"
            corpus_id = SCOPE_BY_PROFILE[pid]
            lines = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            expected_ids[pid] = {row["chunk_id"] for row in lines}
            idx_res = build_index(
                chunks_path,
                corpus_id=corpus_id,
                workspace_id=EVAL_WORKSPACE_ID,
                client=active_client,
                config=cfg,
                rebuild=rebuild,
            )
            results["scopes"][pid] = {
                "corpus_id": corpus_id,
                "workspace_id": EVAL_WORKSPACE_ID,
                "corpus_count": len(lines),
                **idx_res,
            }

        # Verify post-index isolation across both scopes after BOTH builds finish
        check_model_exhaustive(active_client, cfg)
        actual_ids_by_profile: dict[str, set[str]] = {}
        for pid in (PROFILE_A_ID, PROFILE_B_ID):
            scope = scope_filter(SCOPE_BY_PROFILE[pid], EVAL_WORKSPACE_ID)
            count = active_client.count(cfg.collection, count_filter=scope, exact=True).count
            assert count == results["scopes"][pid]["corpus_count"], (
                f"Scope count mismatch for {pid}: expected {results['scopes'][pid]['corpus_count']}, got {count}"
            )
            found_ids: set[str] = set()
            offset = None
            while True:
                pts, offset = active_client.scroll(
                    cfg.collection, scroll_filter=scope, offset=offset, limit=512,
                    with_payload=True, with_vectors=False,
                )
                for p in pts:
                    pl = p.payload or {}
                    assert pl.get("corpus_id") == SCOPE_BY_PROFILE[pid]
                    assert pl.get("workspace_id") == EVAL_WORKSPACE_ID
                    assert (pl.get("metadata") or {}).get("profile_id") == pid
                    found_ids.add(pl["chunk_id"])
                if offset is None:
                    break
            assert found_ids == expected_ids[pid], f"Indexed chunk IDs mismatch for {pid}"
            actual_ids_by_profile[pid] = found_ids

        overlap = actual_ids_by_profile[PROFILE_A_ID] & actual_ids_by_profile[PROFILE_B_ID]
        assert not overlap, f"Cross-profile chunk_id contamination detected: {len(overlap)}"
        results["cross_profile_overlap"] = len(overlap)
        results["total_collection_points"] = active_client.count(cfg.collection, exact=True).count
        results["model_check_exhaustive"] = "PASSED"
        return results
    finally:
        if owns_client:
            active_client.close()
            resolved_key = str(Path(cfg.qdrant_path).resolve())
            vector_store._LOCAL_CLIENTS.pop(resolved_key, None)
