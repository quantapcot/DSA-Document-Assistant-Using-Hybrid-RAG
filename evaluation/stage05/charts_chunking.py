"""Group A (Figures 01-05): Chunking Evaluation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluation.stage05.data_loader import Stage04Bundle
from evaluation.stage05.style import PROFILE_COLORS, add_protocol_footer, save_figure


def _mb(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def generate_chunking_figures(
    bundle: Stage04Bundle,
    png_dir: Path,
    svg_dir: Path,
) -> list[dict[str, Any]]:
    """Generate Figures 01 to 05 (Group A: Chunking Evaluation)."""
    diag = bundle.chunking_diagnostics
    p_rec = diag["profiles"]["recursive_512_100"]
    p_ada = diag["profiles"]["structure_adaptive"]
    c_rec = PROFILE_COLORS["recursive_512_100"]
    c_ada = PROFILE_COLORS["structure_adaptive"]
    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fig 01: Chunk Count Comparison & Storage Breakdown
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    fig.subplots_adjust(wspace=0.28, bottom=0.18, top=0.84)

    # Panel A: Total Chunks with Structural Content-Type Breakdown
    profiles = ["Profile A:\nrecursive_512_100", "Profile B:\nstructure_adaptive"]
    x = np.arange(2)
    rec_total = p_rec["total_chunks"]
    ada_code = p_ada["code_pseudocode_chunks"]
    ada_form = p_ada["formula_chunks"]
    ada_tbl = p_ada["table_chunks"]
    ada_other = p_ada["total_chunks"] - (ada_code + ada_form + ada_tbl)

    ax1.bar(x[0], rec_total, width=0.48, color=c_rec, label="Untyped Recursive Segment")
    ax1.bar(x[1], ada_other, width=0.48, color="#F59E0B", label="Adaptive Narrative / Steps / Def / Heading")
    ax1.bar(x[1], ada_code, width=0.48, bottom=ada_other, color="#0072B2", label="Adaptive Code / Pseudocode")
    ax1.bar(x[1], ada_form, width=0.48, bottom=ada_other + ada_code, color="#009E73", label="Adaptive Formula")
    ax1.bar(x[1], ada_tbl, width=0.48, bottom=ada_other + ada_code + ada_form, color="#7B2CBF", label="Adaptive Table")

    ax1.text(x[0], rec_total + 320, f"{rec_total:,}\n(1.00x)", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax1.text(x[1], p_ada["total_chunks"] + 320, f"{p_ada['total_chunks']:,}\n(2.10x)", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax1.set_xticks(x, profiles)
    ax1.set_ylabel("Number of Indexed Chunks")
    ax1.set_ylim(0, 16200)
    ax1.set_title("(A) Total Chunk Count & Structural Typing")
    ax1.legend(loc="upper left", fontsize=7.6)

    # Panel B: Storage Footprint (Corpus JSONL vs Theoretical Vector vs Shared Physical QdrantLocal)
    cats = ["Corpus JSONL\n(Physical)", "Theoretical Vector\n(f32 384d, Est.)", "Shared QdrantLocal\nSQLite (Both Combined)"]
    bx = np.arange(len(cats))
    w = 0.34
    rec_vals = [_mb(p_rec["corpus_jsonl_bytes"]), _mb(p_rec["theoretical_vector_bytes_f32_384d"]), 0.0]
    ada_vals = [_mb(p_ada["corpus_jsonl_bytes"]), _mb(p_ada["theoretical_vector_bytes_f32_384d"]), 0.0]
    shared_mb = _mb(diag["combined_qdrant_eval_bytes"])

    b1 = ax2.bar(bx[:2] - w / 2, rec_vals[:2], width=w, color=c_rec, label="Profile A (recursive_512_100)")
    b2 = ax2.bar(bx[:2] + w / 2, ada_vals[:2], width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B (structure_adaptive)")
    b3 = ax2.bar(bx[2], shared_mb, width=0.45, color="#475569", label="Combined Physical QdrantLocal (A + B)")

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.2f} MiB", ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax2.set_xticks(bx, cats)
    ax2.set_ylabel("Storage Size (MiB)")
    ax2.set_ylim(0, 118)
    ax2.set_title("(B) Corpus, Theoretical Vector & Shared Qdrant Storage")
    ax2.legend(loc="upper left", fontsize=7.8)

    fig.suptitle("Figure 1: Chunk Count, Structural Typing, and Storage Footprint Comparison")
    add_protocol_footer(
        fig,
        "Source: chunking_metrics.csv & chunking_diagnostics.json. Theoretical vector size = chunk_count * 384 * 4 B; QdrantLocal (100.82 MiB) is physically shared by A + B.",
    )
    saved = save_figure(fig, "fig01_chunk_count_and_storage", png_dir, svg_dir)
    records.append({
        "figure_number": 1,
        "figure_id": "fig01_chunk_count_and_storage",
        "data_contract_id": 1,
        "data_contract_name": "Chunk Count Comparison",
        "group": "A",
        "title": "Chunk Count, Structural Typing, and Storage Footprint Comparison",
        "protocol": "Frozen Stage 02 benchmark corpora (Profile A: 6,366 chunks; Profile B: 13,390 chunks; combined QdrantLocal: 19,756 points).",
        "metrics": ["total_chunks", "code_pseudocode_chunks", "formula_chunks", "table_chunks", "corpus_jsonl_bytes", "theoretical_vector_bytes_f32_384d", "combined_qdrant_eval_bytes"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/chunking/chunking_metrics.csv",
            "evaluation/artifacts/stage04/chunking/chunking_diagnostics.json",
        ],
        "caption_vi": (
            "Hình 1: So sánh tổng số lượng chunk, phân loại cấu trúc (code/pseudocode, công thức, bảng) và dung lượng lưu trữ giữa "
            "Profile A (recursive_512_100: 6.366 chunks) và Profile B (structure_adaptive: 13.390 chunks, gấp 2,10×). Dung lượng vector "
            "từng profile là ước lượng lý thuyết (float32 384 chiều), tách biệt với dung lượng vật lý QdrantLocal SQLite dùng chung (100,82 MiB)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 02: Chunk Character Length Distribution
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    fig.subplots_adjust(wspace=0.28, bottom=0.18, top=0.84)

    buckets = ["lt_100", "100_250", "250_400", "400_512", "gt_512"]
    bucket_labels = ["< 100", "100–250", "250–400", "400–512", "> 512"]
    rec_pct = [100.0 * p_rec["char_length_buckets"][k] / p_rec["total_chunks"] for k in buckets]
    ada_pct = [100.0 * p_ada["char_length_buckets"][k] / p_ada["total_chunks"] for k in buckets]

    bx = np.arange(len(buckets))
    w = 0.36
    b1 = ax1.bar(bx - w / 2, rec_pct, width=w, color=c_rec, label="Profile A: recursive_512_100")
    b2 = ax1.bar(bx + w / 2, ada_pct, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B: structure_adaptive")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h >= 0.3:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
    ax1.set_xticks(bx, bucket_labels)
    ax1.set_xlabel("Chunk Length Bucket (Characters)")
    ax1.set_ylabel("Share of Chunks (%)")
    ax1.set_ylim(0, 68)
    ax1.set_title("(A) Character Length Distribution (% of Profile)")
    ax1.legend(loc="upper right", fontsize=8.2)

    # Panel B: Quantile Summary (Min, Median, Mean, P95, Max)
    stat_keys = ["char_min", "char_median", "char_mean", "char_p95", "char_max"]
    stat_labels = ["Min", "Median", "Mean", "P95", "Max"]
    sx = np.arange(len(stat_keys))
    rec_stats = [p_rec[k] for k in stat_keys]
    ada_stats = [p_ada[k] for k in stat_keys]
    b1 = ax2.bar(sx - w / 2, rec_stats, width=w, color=c_rec, label="Profile A: recursive_512_100")
    b2 = ax2.bar(sx + w / 2, ada_stats, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B: structure_adaptive")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{h:.0f}", ha="center", va="bottom", fontsize=7.8)
    ax2.set_xticks(sx, stat_labels)
    ax2.set_ylabel("Chunk Length (Characters)")
    ax2.set_ylim(0, 2120)
    ax2.set_title("(B) Character Length Summary Statistics")
    ax2.legend(loc="upper left", fontsize=8.2)

    fig.suptitle("Figure 2: Chunk Character Length Distribution and Summary Statistics")
    add_protocol_footer(
        fig,
        "Source: chunking_diagnostics.json & chunking_metrics.csv. Profile A caps at 512 chars (median 421); Profile B has shorter median (112) with long intact structural units up to 1,871 chars.",
    )
    saved = save_figure(fig, "fig02_chunk_length_distribution", png_dir, svg_dir)
    records.append({
        "figure_number": 2,
        "figure_id": "fig02_chunk_length_distribution",
        "data_contract_id": 2,
        "data_contract_name": "Chunk Length Distribution",
        "group": "A",
        "title": "Chunk Character Length Distribution and Summary Statistics",
        "protocol": "All indexed chunks across frozen corpora (n=6,366 for Profile A; n=13,390 for Profile B).",
        "metrics": ["char_length_buckets", "char_min", "char_median", "char_mean", "char_p95", "char_max"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/chunking/chunking_diagnostics.json",
            "evaluation/artifacts/stage04/chunking/chunking_metrics.csv",
        ],
        "caption_vi": (
            "Hình 2: Phân bố độ dài ký tự và các thống kê mô tả (Min, Median, Mean, P95, Max) của hai chiến lược phân đoạn. "
            "Profile A tập trung vào nhóm 400–512 ký tự (52,8%, trung vị 421), trong khi 46,8% chunk của Profile B ngắn dưới 100 ký tự "
            "(trung vị 112), nhưng bảo toàn nguyên vẹn các khối code/bảng dài tới 1.871 ký tự."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 03: Token Distribution, Reranker Pair Tokens & Retrieved Context Volume
    # ------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.9))
    fig.subplots_adjust(wspace=0.30, bottom=0.20, top=0.82)

    # Panel A: Embedding Token Distribution
    tbuckets = ["lt_32", "32_64", "64_128", "129_256", "gt_256"]
    tlabels = ["<32", "32–64", "64–128", "129–256", ">256"]
    rec_tpct = [100.0 * p_rec["emb_token_buckets"][k] / p_rec["total_chunks"] for k in tbuckets]
    ada_tpct = [100.0 * p_ada["emb_token_buckets"][k] / p_ada["total_chunks"] for k in tbuckets]
    tx = np.arange(len(tbuckets))
    w = 0.37
    b1 = ax1.bar(tx - w / 2, rec_tpct, width=w, color=c_rec, label="Profile A (Rec)")
    b2 = ax1.bar(tx + w / 2, ada_tpct, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B (Ada)")
    ax1.axvline(2.5, color="#B91C1C", linestyle="--", linewidth=1.3, label="MiniLM limit (128)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h >= 1.5:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.0)
    ax1.set_xticks(tx, tlabels)
    ax1.set_xlabel("Embedding Tokens (MiniLM)")
    ax1.set_ylabel("Share of Chunks (%)")
    ax1.set_ylim(0, 60)
    ax1.set_title("(A) Embedding Token Buckets")
    ax1.legend(loc="upper right", fontsize=7.2)

    # Panel B: Reranker Query-Chunk Pair Tokens (BGE limit=512)
    rkeys = ["rer_pair_tok_mean", "rer_pair_tok_p95", "rer_pair_tok_max"]
    rlabels = ["Mean", "P95", "Max"]
    rx = np.arange(len(rkeys))
    b1 = ax2.bar(rx - w / 2, [p_rec[k] for k in rkeys], width=w, color=c_rec, label="Profile A (n=4,515 pairs)")
    b2 = ax2.bar(rx + w / 2, [p_ada[k] for k in rkeys], width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B (n=4,496 pairs)")
    ax2.axhline(512, color="#B91C1C", linestyle="--", linewidth=1.3, label="BGE Reranker limit (512)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.0f}", ha="center", va="bottom", fontsize=7.6)
    ax2.set_xticks(rx, rlabels)
    ax2.set_ylabel("Query–Chunk Pair Tokens (BGE)")
    ax2.set_ylim(0, 565)
    ax2.set_title("(B) Reranker Pair Diagnostics (0% Trunc.)")
    ax2.legend(loc="upper left", fontsize=7.2)

    # Panel C: Retrieved Context Token Volume per Query at K=5, 15, 60
    ks_ctx = ["K=5", "K=15", "K=60"]
    rec_ctx = [p_rec["retrieved_emb_tokens_k5_mean"], p_rec["retrieved_emb_tokens_k15_mean"], p_rec["retrieved_emb_tokens_k60_mean"]]
    ada_ctx = [p_ada["retrieved_emb_tokens_k5_mean"], p_ada["retrieved_emb_tokens_k15_mean"], p_ada["retrieved_emb_tokens_k60_mean"]]
    kx = np.arange(len(ks_ctx))
    b1 = ax3.bar(kx - w / 2, rec_ctx, width=w, color=c_rec, label="Profile A (Rec)")
    b2 = ax3.bar(kx + w / 2, ada_ctx, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B (Ada)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 140, f"{h:,.0f}", ha="center", va="bottom", fontsize=7.4)
    ax3.set_xticks(kx, ks_ctx)
    ax3.set_ylabel("Mean Retrieved Tokens / Query")
    ax3.set_ylim(0, 8600)
    ax3.set_title("(C) Retrieved Context Token Volume")
    ax3.legend(loc="upper left", fontsize=7.5)

    fig.suptitle("Figure 3: Dual-Tokenizer Token Distribution, Reranker Pair Diagnostics, and Context Volume")
    add_protocol_footer(
        fig,
        "Source: chunking_diagnostics.json & chunking_metrics.csv. Embedding: MiniLM-L12-v2 (max=128); Reranker: BGE-reranker-v2-m3 (max=512, 0.0% pair truncation on both profiles).",
    )
    saved = save_figure(fig, "fig03_token_distribution_and_context_volume", png_dir, svg_dir)
    records.append({
        "figure_number": 3,
        "figure_id": "fig03_token_distribution_and_context_volume",
        "data_contract_id": 3,
        "data_contract_name": "Token Distribution",
        "group": "A",
        "title": "Dual-Tokenizer Token Distribution, Reranker Pair Diagnostics, and Context Volume",
        "protocol": "Dual-tokenizer diagnostics (MiniLM max=128; BGE reranker max=512 on evaluated query-chunk pairs) and mean retrieved tokens at K=5, 15, 60.",
        "metrics": ["emb_token_buckets", "rer_pair_tok_mean", "rer_pair_tok_p95", "rer_pair_tok_max", "retrieved_emb_tokens_k5_mean", "retrieved_emb_tokens_k15_mean", "retrieved_emb_tokens_k60_mean"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/chunking/chunking_diagnostics.json",
            "evaluation/artifacts/stage04/chunking/chunking_metrics.csv",
        ],
        "caption_vi": (
            "Hình 3: Phân bố số token theo hai tokenizer độc lập và dung lượng ngữ cảnh truy xuất. (A) Với MiniLM (giới hạn 128 tokens), "
            "43,2% chunk của Profile A vượt ngưỡng trong khi Profile B = 0%; (B) Với BGE Reranker (giới hạn 512 tokens), cả hai profile "
            "đều đạt 0% cắt cụt cặp câu hỏi–chunk (max 363 vs 190 tokens); (C) Tại cùng K, Profile A mang lượng token ngữ cảnh gấp ~2,1–2,3× Profile B."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 04: Truncation Rate & Discarded Embedding Tokens
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    fig.subplots_adjust(wspace=0.30, bottom=0.18, top=0.83)

    cats = ["Embedding Chunks\nExceeding 128 Tokens", "Discarded Embedding\nTokens (% of Total)", "Reranker Query–Chunk\nPairs > 512 Tokens"]
    rx = np.arange(len(cats))
    w = 0.35
    rec_rates = [p_rec["emb_truncation_rate_pct"], p_rec["emb_discarded_token_rate_pct"], p_rec["rer_pair_truncation_rate_pct"]]
    ada_rates = [p_ada["emb_truncation_rate_pct"], p_ada["emb_discarded_token_rate_pct"], p_ada["rer_pair_truncation_rate_pct"]]

    b1 = ax1.bar(rx - w / 2, rec_rates, width=w, color=c_rec, label="Profile A: recursive_512_100")
    b2 = ax1.bar(rx + w / 2, ada_rates, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B: structure_adaptive")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax1.set_xticks(rx, cats)
    ax1.set_ylabel("Rate (%)")
    ax1.set_ylim(0, 52)
    ax1.set_title("(A) Embedding vs Reranker Truncation Rates")
    ax1.legend(loc="upper right", fontsize=8.2)

    # Panel B: Retained vs Discarded Embedding Tokens (Absolute Counts)
    px = np.arange(2)
    rec_ret = (p_rec["emb_total_tokens"] - p_rec["emb_discarded_tokens"]) / 1000.0
    rec_disc = p_rec["emb_discarded_tokens"] / 1000.0
    ada_ret = (p_ada["emb_total_tokens"] - p_ada["emb_discarded_tokens"]) / 1000.0
    ada_disc = p_ada["emb_discarded_tokens"] / 1000.0

    ax2.bar(px, [rec_ret, ada_ret], width=0.45, color="#0072B2", label="Encoded by MiniLM (<= 128 tok)")
    ax2.bar(px, [rec_disc, ada_disc], width=0.45, bottom=[rec_ret, ada_ret], color="#DC2626", hatch="xx", edgecolor="#7F1D1D", label="Discarded by MiniLM (> 128 tok)")
    ax2.text(px[0], rec_ret + rec_disc + 18, f"Total: {p_rec['emb_total_tokens']:,}\nDiscarded: {p_rec['emb_discarded_tokens']:,} (15.13%)", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.text(px[1], ada_ret + ada_disc + 18, f"Total: {p_ada['emb_total_tokens']:,}\nDiscarded: 0 (0.00%)", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.set_xticks(px, ["Profile A:\nrecursive_512_100", "Profile B:\nstructure_adaptive"])
    ax2.set_ylabel("Embedding Tokens (Thousands)")
    ax2.set_ylim(0, 920)
    ax2.set_title("(B) Encoded vs Discarded Embedding Tokens")
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Figure 4: Tokenizer Truncation Rates and Discarded Embedding Token Volume")
    add_protocol_footer(
        fig,
        "Source: chunking_metrics.csv. Profile A truncates 2,750 / 6,366 chunks (114,961 tokens) at MiniLM embedding, whereas BGE reranker preserves 100% of tokens.",
    )
    saved = save_figure(fig, "fig04_truncation_rate_and_discarded_tokens", png_dir, svg_dir)
    records.append({
        "figure_number": 4,
        "figure_id": "fig04_truncation_rate_and_discarded_tokens",
        "data_contract_id": 4,
        "data_contract_name": "Truncation Rate",
        "group": "A",
        "title": "Tokenizer Truncation Rates and Discarded Embedding Token Volume",
        "protocol": "Exact token counts under MiniLM-L12-v2 (128 limit) and BGE-reranker-v2-m3 (512 limit).",
        "metrics": ["emb_truncation_rate_pct", "emb_discarded_tokens", "emb_discarded_token_rate_pct", "rer_pair_truncation_rate_pct"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/chunking/chunking_metrics.csv",
        ],
        "caption_vi": (
            "Hình 4: Tỷ lệ cắt cụt (truncation) và lượng token bị loại bỏ ở giai đoạn Embedding so với Reranking. Profile B loại bỏ hoàn toàn "
            "cắt cụt embedding (0,00% so với 43,20% và 114.961 token bị bỏ của Profile A), giải thích ưu thế cấu trúc của Adaptive ở lớp mã hóa, "
            "trong khi BGE Reranker xử lý trọn vẹn cả hai profile (0,00% cắt cụt)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 05: Source Coverage, KU Retention & Overlap Redundancy
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(wspace=0.28, bottom=0.18, top=0.83)

    cov_labels = ["Raw Source\nCoverage (%)", "Effective Source\nCoverage (%)", "Knowledge Unit\nRetention (91 KUs)"]
    cx = np.arange(len(cov_labels))
    w = 0.35
    rec_cov = [p_rec["source_coverage_pct"], p_rec["effective_source_coverage_pct"], 100.0 * p_rec["ku_retained"] / p_rec["ku_total"]]
    ada_cov = [p_ada["source_coverage_pct"], p_ada["effective_source_coverage_pct"], 100.0 * p_ada["ku_retained"] / p_ada["ku_total"]]

    b1 = ax1.bar(cx - w / 2, rec_cov, width=w, color=c_rec, label="Profile A: recursive_512_100")
    b2 = ax1.bar(cx + w / 2, ada_cov, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B: structure_adaptive")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax1.set_xticks(cx, cov_labels)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 114)
    ax1.set_title("(A) Source Coverage & Knowledge Unit Retention")
    ax1.legend(loc="lower right", fontsize=8.0)

    red_labels = ["Overlap\nRedundancy (%)", "Synthetic Prefix\nCharacters (%)", "Excess Duplicate\nChunks (% of Total)"]
    rx = np.arange(len(red_labels))
    rec_red = [p_rec["overlap_redundancy_pct"], p_rec["synthetic_prefix_pct"], 100.0 * p_rec["duplicate_text_excess_chunks"] / p_rec["total_chunks"]]
    ada_red = [p_ada["overlap_redundancy_pct"], p_ada["synthetic_prefix_pct"], 100.0 * p_ada["duplicate_text_excess_chunks"] / p_ada["total_chunks"]]

    b1 = ax2.bar(rx - w / 2, rec_red, width=w, color=c_rec, label="Profile A: recursive_512_100")
    b2 = ax2.bar(rx + w / 2, ada_red, width=w, color=c_ada, hatch="///", edgecolor="#78350F", label="Profile B: structure_adaptive")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax2.set_xticks(rx, red_labels)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 25.5)
    ax2.set_title("(B) Overlap Redundancy, Prefixes & Text Duplication")
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Figure 5: Source Coverage, Knowledge Unit Retention, and Redundancy Profile")
    add_protocol_footer(
        fig,
        "Source: chunking_metrics.csv & chunking_diagnostics.json. Unexplained coverage loss = 0 chars and missing provenance = 0 on both profiles; 91/91 KUs retained.",
    )
    saved = save_figure(fig, "fig05_source_coverage_ku_and_redundancy", png_dir, svg_dir)
    records.append({
        "figure_number": 5,
        "figure_id": "fig05_source_coverage_ku_and_redundancy",
        "data_contract_id": 5,
        "data_contract_name": "Source Coverage",
        "group": "A",
        "title": "Source Coverage, Knowledge Unit Retention, and Redundancy Profile",
        "protocol": "Provenance-verified character coverage across 1,733 canonical pages and 91 benchmark Knowledge Units.",
        "metrics": ["source_coverage_pct", "effective_source_coverage_pct", "ku_retained", "overlap_redundancy_pct", "synthetic_prefix_pct", "duplicate_text_excess_chunks"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/chunking/chunking_metrics.csv",
            "evaluation/artifacts/stage04/chunking/chunking_diagnostics.json",
        ],
        "caption_vi": (
            "Hình 5: Độ phủ văn bản nguồn, tỷ lệ bảo toàn đơn vị kiến thức (91/91 KU = 100% ở cả hai profile) và đặc trưng dư thừa. "
            "Profile A có độ phủ nguồn hiệu dụng cao hơn (91,36% vs 88,93%) nhưng dư thừa chồng lấn lớn (20,33%); ngược lại Profile B "
            "giảm chồng lấn xuống 1,06% song có tỷ lệ nhóm văn bản ngắn lặp lại cao hơn (15,03% vs 3,25% số chunk)."
        ),
        **saved,
    })

    return records
