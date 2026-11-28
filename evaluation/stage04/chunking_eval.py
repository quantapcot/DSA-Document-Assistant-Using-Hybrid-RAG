"""Stage 04 – Step 3: Chunking Evaluation (Structure, Preservation, Dual-Tokenizer Efficiency, Redundancy, Storage)."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.stage03.dataset_schema import FROZEN_CORPORA


def _percentile(sorted_vals: list[float | int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * p))))
    return float(sorted_vals[idx])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_chunking_evaluation(
    *,
    root: Path,
    output_dir: Path,
    retrieval_rankings_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate frozen Profile A & Profile B corpora without rebuilding them."""
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    stage02_diag = json.loads((root / "evaluation/artifacts/diagnostics.json").read_text(encoding="utf-8"))
    ds_manifest = json.loads((root / "evaluation/datasets/dataset_manifest.json").read_text(encoding="utf-8"))
    questions = _load_jsonl(root / "evaluation/datasets/questions_40.jsonl")
    q_by_id = {q["query_id"]: q for q in questions}

    emb_model = ds_manifest["models"]["embedding_model"]
    emb_max = int(ds_manifest["models"]["embedding_max_seq_length"])
    rer_model = ds_manifest["models"]["reranker_model"]
    rer_max = int(ds_manifest["models"]["reranker_max_length"])

    emb_tok = AutoTokenizer.from_pretrained(emb_model, local_files_only=True)
    rer_tok = AutoTokenizer.from_pretrained(rer_model, local_files_only=True)

    rankings_path = retrieval_rankings_path or (root / "evaluation/artifacts/stage03/raw_retrieval_results.jsonl")
    raw_rankings = _load_jsonl(rankings_path) if rankings_path.exists() else []

    qdrant_dir = root / "data/runtime/qdrant_eval"
    combined_qdrant_bytes = sum(p.stat().st_size for p in qdrant_dir.rglob("*") if p.is_file())

    csv_rows: list[dict[str, Any]] = []
    full_diagnostics: dict[str, Any] = {
        "embedding_tokenizer": emb_model,
        "embedding_max_seq_length": emb_max,
        "reranker_tokenizer": rer_model,
        "reranker_max_length": rer_max,
        "combined_qdrant_eval_bytes": combined_qdrant_bytes,
        "storage_note": (
            "Profile A and Profile B share one physical Qdrant local SQLite storage "
            "(data/runtime/qdrant_eval). Per-profile raw vector bytes below are theoretical float32 "
            "estimates (chunk_count * 384 * 4 bytes), NOT physical per-profile disk measurements."
        ),
        "profiles": {},
    }

    for pid in (PROFILE_A_ID, PROFILE_B_ID):
        cpath = root / FROZEN_CORPORA[pid]["rel_chunks_path"]
        chunks = _load_jsonl(cpath)
        chunk_by_id = {c["chunk_id"]: c for c in chunks}
        s2 = stage02_diag["profiles"][pid]["diagnostics"]
        ku_ret = ds_manifest["ku_retention_in_frozen_corpora"][pid]

        char_lens = sorted(len(c["text"]) for c in chunks)
        # Actual embedding token lengths (with special tokens, matching SentenceTransformer encoding)
        emb_lens: list[int] = []
        rer_chunk_lens: list[int] = []
        # Batch tokenize for speed
        texts = [c["text"] for c in chunks]
        for i in range(0, len(texts), 512):
            enc_e = emb_tok(texts[i : i + 512], truncation=False, add_special_tokens=True)
            emb_lens.extend(len(ids) for ids in enc_e["input_ids"])
            enc_r = rer_tok(texts[i : i + 512], truncation=False, add_special_tokens=True)
            rer_chunk_lens.extend(len(ids) for ids in enc_r["input_ids"])

        emb_sorted = sorted(emb_lens)
        rer_sorted = sorted(rer_chunk_lens)
        emb_trunc_cnt = sum(1 for x in emb_lens if x > emb_max)
        emb_total_tokens = sum(emb_lens)
        emb_discarded_tokens = sum(max(0, x - emb_max) for x in emb_lens)

        # Actual reranker query-chunk pair token lengths across unique retrieved (query, chunk) pairs
        pair_lens: list[int] = []
        retrieved_tok_at_k: dict[int, list[int]] = {5: [], 15: [], 60: []}
        seen_pairs: set[tuple[str, str]] = set()
        pair_texts: list[tuple[str, str]] = []
        for rec in raw_rankings:
            if rec["profile_id"] != pid:
                continue
            qid = rec["query_id"]
            qtext = q_by_id[qid]["query"]
            hits = rec["results"]
            if rec["method"] == "hybrid_rerank":
                for k_cut in (5, 15, 60):
                    tok_sum = sum(
                        len(emb_tok.encode(chunk_by_id[h["chunk_id"]]["text"], add_special_tokens=False))
                        for h in hits[:k_cut]
                    )
                    retrieved_tok_at_k[k_cut].append(tok_sum)
            for h in hits:
                cid = h["chunk_id"]
                if (qid, cid) not in seen_pairs:
                    seen_pairs.add((qid, cid))
                    pair_texts.append((qtext, chunk_by_id[cid]["text"]))

        for i in range(0, len(pair_texts), 512):
            batch = pair_texts[i : i + 512]
            enc_p = rer_tok(
                [a for a, _ in batch],
                [b for _, b in batch],
                truncation=False,
                add_special_tokens=True,
            )
            pair_lens.extend(len(ids) for ids in enc_p["input_ids"])

        pair_sorted = sorted(pair_lens)
        pair_trunc_cnt = sum(1 for x in pair_lens if x > rer_max)

        # Content type & duplicate occurrence counts
        ct_counts = dict(sorted(Counter((c.get("metadata") or {}).get("content_type", "unknown") for c in chunks).items()))
        text_counts = Counter(c["text"] for c in chunks)
        dup_text_chunks = sum(v - 1 for v in text_counts.values() if v > 1)
        dup_text_groups = sum(1 for v in text_counts.values() if v > 1)

        # Code/formula/table preservation counts (from chunk content_type or structural heuristics)
        code_chunks = ct_counts.get("code_block", 0) + ct_counts.get("pseudocode", 0)
        formula_chunks = ct_counts.get("formula", 0)
        table_chunks = ct_counts.get("table", 0)
        jsonl_bytes = cpath.stat().st_size
        theoretical_vec_bytes = len(chunks) * 384 * 4

        row = {
            "profile_id": pid,
            "corpus_id": FROZEN_CORPORA[pid]["corpus_id"],
            "total_chunks": len(chunks),
            "char_min": char_lens[0],
            "char_mean": round(mean(char_lens), 2),
            "char_median": round(median(char_lens), 2),
            "char_p95": round(_percentile(char_lens, 0.95), 2),
            "char_max": char_lens[-1],
            "emb_tok_min": emb_sorted[0],
            "emb_tok_mean": round(mean(emb_lens), 2),
            "emb_tok_median": round(median(emb_lens), 2),
            "emb_tok_p95": round(_percentile(emb_sorted, 0.95), 2),
            "emb_tok_max": emb_sorted[-1],
            "emb_limit": emb_max,
            "emb_truncation_count": emb_trunc_cnt,
            "emb_truncation_rate_pct": round(100.0 * emb_trunc_cnt / len(chunks), 2),
            "emb_total_tokens": emb_total_tokens,
            "emb_discarded_tokens": emb_discarded_tokens,
            "emb_discarded_token_rate_pct": round(100.0 * emb_discarded_tokens / max(1, emb_total_tokens), 2),
            "rer_chunk_tok_mean": round(mean(rer_chunk_lens), 2),
            "rer_chunk_tok_p95": round(_percentile(rer_sorted, 0.95), 2),
            "rer_chunk_tok_max": rer_sorted[-1],
            "rer_pair_count": len(pair_lens),
            "rer_pair_tok_mean": round(mean(pair_lens), 2) if pair_lens else 0.0,
            "rer_pair_tok_p95": round(_percentile(pair_sorted, 0.95), 2) if pair_lens else 0.0,
            "rer_pair_tok_max": pair_sorted[-1] if pair_lens else 0,
            "rer_pair_truncation_count": pair_trunc_cnt,
            "rer_pair_truncation_rate_pct": round(100.0 * pair_trunc_cnt / max(1, len(pair_lens)), 2),
            "retrieved_emb_tokens_k5_mean": round(mean(retrieved_tok_at_k[5]), 1) if retrieved_tok_at_k[5] else 0.0,
            "retrieved_emb_tokens_k15_mean": round(mean(retrieved_tok_at_k[15]), 1) if retrieved_tok_at_k[15] else 0.0,
            "retrieved_emb_tokens_k60_mean": round(mean(retrieved_tok_at_k[60]), 1) if retrieved_tok_at_k[60] else 0.0,
            "source_coverage_pct": s2["canonical_source_coverage_pct"],
            "effective_source_coverage_pct": s2["canonical_effective_coverage_pct"],
            "dropped_chars_union": s2["dropped_chars_union"],
            "uncovered_whitespace_chars": s2["uncovered_whitespace_chars"],
            "unexplained_coverage_loss_chars": s2["uncovered_other_meaningful_chars"],
            "ku_retained": ku_ret["retained_kus"],
            "ku_total": ku_ret["total_kus"],
            "provenance_missing_count": s2["missing_provenance_count"],
            "overlap_chars_total": s2["overlap_chars_total"],
            "overlap_redundancy_pct": s2["overlap_redundancy_pct"],
            "repeated_context_source_chars": s2["repeated_context_source_chars_total"],
            "synthetic_prefix_chars": s2["synthetic_prefix_chars_total"],
            "synthetic_prefix_pct": s2["synthetic_prefix_pct"],
            "duplicate_text_groups": dup_text_groups,
            "duplicate_text_excess_chunks": dup_text_chunks,
            "code_pseudocode_chunks": code_chunks,
            "formula_chunks": formula_chunks,
            "table_chunks": table_chunks,
            "corpus_jsonl_bytes": jsonl_bytes,
            "theoretical_vector_bytes_f32_384d": theoretical_vec_bytes,
            "combined_qdrant_eval_bytes": combined_qdrant_bytes,
        }
        csv_rows.append(row)
        full_diagnostics["profiles"][pid] = {
            **row,
            "content_type_distribution": ct_counts,
            "dropped_reasons": s2["dropped_reasons"],
            "char_length_buckets": {
                "lt_100": sum(1 for x in char_lens if x < 100),
                "100_250": sum(1 for x in char_lens if 100 <= x < 250),
                "250_400": sum(1 for x in char_lens if 250 <= x < 400),
                "400_512": sum(1 for x in char_lens if 400 <= x <= 512),
                "gt_512": sum(1 for x in char_lens if x > 512),
            },
            "emb_token_buckets": {
                "lt_32": sum(1 for x in emb_lens if x < 32),
                "32_64": sum(1 for x in emb_lens if 32 <= x < 64),
                "64_128": sum(1 for x in emb_lens if 64 <= x <= 128),
                "129_256": sum(1 for x in emb_lens if 129 <= x <= 256),
                "gt_256": sum(1 for x in emb_lens if x > 256),
            },
        }

    csv_path = output_dir / "chunking_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    diag_path = output_dir / "chunking_diagnostics.json"
    diag_path.write_text(json.dumps(full_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_diagnostics
