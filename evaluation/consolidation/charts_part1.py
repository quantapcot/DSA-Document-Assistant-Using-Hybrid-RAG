"""Consolidated Vietnamese Multi-Panel Figures 01–05 (Chunking, Retrieval, Evidence, Performance, Generation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.consolidation.style import (
    COMBO_ORDER,
    COMBO_STYLE,
    K_POS,
    K_VALUES,
    PROFILE_COLORS,
    apply_consolidated_style,
    save_consolidated_figure,
    shared_combo_handles,
    style_axis_clean,
)
from evaluation.stage05.data_loader import Stage04Bundle


def generate_fig01_danh_gia_chunking(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 01: Đánh giá chiến lược Chunking (tổng hợp từ Stage 05 Hình 01–05)."""
    apply_consolidated_style()
    cm = bundle.chunking_metrics.set_index("profile_id")
    diag = bundle.chunking_diagnostics
    rec = cm.loc["recursive_512_100"]
    adp = cm.loc["structure_adaptive"]

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.26, top=0.87, bottom=0.08, left=0.05, right=0.98)

    fig.suptitle(
        "HÌNH 01 — ĐÁNH GIÁ CHIẾN LƯỢC CHUNKING: RECURSIVE (PROFILE A) SO VỚI STRUCTURE-ADAPTIVE (PROFILE B)",
        fontsize=23, fontweight="bold", y=0.955, color="#111827",
    )
    fig.text(
        0.5, 0.915,
        "Nguồn: 54 tài liệu DSA đóng băng (1.733 trang chuẩn hóa, 91 Knowledge Units) | Profile A: 6.366 chunks vs Profile B: 13.390 chunks (gấp 2,10×)",
        ha="center", fontsize=13.5, color="#374151",
    )

    c_rec, c_adp = PROFILE_COLORS["recursive_512_100"], PROFILE_COLORS["structure_adaptive"]

    # (a) Chunk Count & Structural Typing
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    cats = ["Tổng số\nchunk", "Code /\nPseudocode", "Công thức\ntoán", "Bảng\ndữ liệu"]
    v_rec = [rec["total_chunks"], rec["code_pseudocode_chunks"], rec["formula_chunks"], rec["table_chunks"]]
    v_adp = [adp["total_chunks"], adp["code_pseudocode_chunks"], adp["formula_chunks"], adp["table_chunks"]]
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(x - w / 2, v_rec, w, color=c_rec, label="Profile A: Recursive (512/100)")
    b2 = ax.bar(x + w / 2, v_adp, w, color=c_adp, hatch="//", edgecolor="#FFFFFF", label="Profile B: Structure-Adaptive")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 220, f"{int(h):,}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 15600)
    ax.set_ylabel("Số lượng chunk")
    ax.set_title("(a) Tổng số chunk và phân loại cấu trúc")
    ax.legend(loc="upper right", fontsize=11)

    # (b) Character Length Distribution
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    buckets_rec = diag["profiles"]["recursive_512_100"]["char_length_buckets"]
    buckets_adp = diag["profiles"]["structure_adaptive"]["char_length_buckets"]
    blabels = list(buckets_rec.keys())
    pct_rec = [buckets_rec[k] / rec["total_chunks"] * 100.0 for k in blabels]
    pct_adp = [buckets_adp[k] / adp["total_chunks"] * 100.0 for k in blabels]
    xb = np.arange(len(blabels))
    ax.bar(xb - w / 2, pct_rec, w, color=c_rec, label="Profile A")
    ax.bar(xb + w / 2, pct_adp, w, color=c_adp, hatch="//", edgecolor="#FFFFFF", label="Profile B")
    for i, (r_v, a_v) in enumerate(zip(pct_rec, pct_adp)):
        if r_v >= 2.0:
            ax.text(i - w / 2, r_v + 1.1, f"{r_v:.1f}%", ha="center", va="bottom", fontsize=9.5, color=c_rec, fontweight="bold")
        if a_v >= 2.0:
            ax.text(i + w / 2, a_v + 1.1, f"{a_v:.1f}%", ha="center", va="bottom", fontsize=9.5, color=c_adp, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels(blabels, rotation=18, ha="right")
    ax.set_ylim(0, 70)
    ax.set_ylabel("Tỷ lệ chunk (%)")
    ax.set_title("(b) Phân bố độ dài ký tự (Median: 421 vs 112)")
    ax.text(
        0.98, 0.96,
        f"Profile A: Min={int(rec['char_min'])}, Med={int(rec['char_median'])}, P95={int(rec['char_p95'])}, Max={int(rec['char_max'])}\n"
        f"Profile B: Min={int(adp['char_min'])}, Med={int(adp['char_median'])}, P95={int(adp['char_p95'])}, Max={int(adp['char_max'])}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.94),
    )

    # (c) Dual-Tokenizer Distribution & Reranker Pair Tokens
    ax = fig.add_subplot(gs[0, 2])
    style_axis_clean(ax)
    tok_metrics = ["Emb\nMean", "Emb\nP95", "Emb\nMax", "Rer Pair\nMean", "Rer Pair\nP95", "Rer Pair\nMax"]
    tr = [rec["emb_tok_mean"], rec["emb_tok_p95"], rec["emb_tok_max"], rec["rer_pair_tok_mean"], rec["rer_pair_tok_p95"], rec["rer_pair_tok_max"]]
    ta = [adp["emb_tok_mean"], adp["emb_tok_p95"], adp["emb_tok_max"], adp["rer_pair_tok_mean"], adp["rer_pair_tok_p95"], adp["rer_pair_tok_max"]]
    xt = np.arange(len(tok_metrics))
    ax.bar(xt - w / 2, tr, w, color=c_rec, label="Profile A")
    ax.bar(xt + w / 2, ta, w, color=c_adp, hatch="//", edgecolor="#FFFFFF", label="Profile B")
    ax.axhline(128, color="#DC2626", linestyle="--", linewidth=1.6, label="Giới hạn MiniLM (128 tok)")
    ax.axhline(512, color="#059669", linestyle=":", linewidth=1.6, label="Giới hạn BGE Reranker (512 tok)")
    for i, (r_v, a_v) in enumerate(zip(tr, ta)):
        ax.text(i - w / 2, r_v + 10, f"{int(round(r_v))}", ha="center", va="bottom", fontsize=9.3, fontweight="bold", color=c_rec,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
        ax.text(i + w / 2, a_v + 10, f"{int(round(a_v))}", ha="center", va="bottom", fontsize=9.3, fontweight="bold", color=c_adp,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xticks(xt)
    ax.set_xticklabels(tok_metrics)
    ax.set_ylim(0, 560)
    ax.set_ylabel("Số token")
    ax.set_title("(c) Thống kê Dual-Tokenizer (MiniLM & BGE Reranker)")
    ax.legend(loc="upper left", fontsize=9.8, ncol=2)

    # (d) Truncation Rate & Discarded Embedding Tokens
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    trunc_cats = ["Cắt cụt\nEmbedding (%)", "Token Emb\nbị loại bỏ (%)", "Cắt cụt cặp\nReranker (%)"]
    t_rec = [rec["emb_truncation_rate_pct"], rec["emb_discarded_token_rate_pct"], rec["rer_pair_truncation_rate_pct"]]
    t_adp = [adp["emb_truncation_rate_pct"], adp["emb_discarded_token_rate_pct"], adp["rer_pair_truncation_rate_pct"]]
    xtr = np.arange(len(trunc_cats))
    ax.bar(xtr - w / 2, t_rec, w, color=c_rec, label="Profile A")
    ax.bar(xtr + w / 2, t_adp, w, color=c_adp, hatch="//", edgecolor="#FFFFFF", label="Profile B")
    for i, (r_v, a_v) in enumerate(zip(t_rec, t_adp)):
        ax.text(i - w / 2, r_v + 1.0, f"{r_v:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_rec)
        ax.text(i + w / 2, a_v + 1.0, f"{a_v:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_adp)
    ax.set_xticks(xtr)
    ax.set_xticklabels(trunc_cats)
    ax.set_ylim(0, 54)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("(d) Tỷ lệ cắt cụt (Truncation) và token bị loại bỏ")
    ax.text(
        0.98, 0.92,
        f"Profile A: {int(rec['emb_truncation_count']):,} chunks vượt 128 tok\n"
        f"({int(rec['emb_discarded_tokens']):,} tokens bị bỏ)\n"
        f"Profile B: 0 chunks cắt cụt (0,00%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#EF4444", alpha=0.92),
    )

    # (e) Effective Source Coverage, KU Retention & Redundancy
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    cov_cats = ["Độ phủ\nnguồn (%)", "Độ phủ\nhiệu dụng (%)", "Bảo toàn\nKU (%)", "Dư thừa\noverlap (%)", "Tiền tố\ntổng hợp (%)"]
    c_r = [rec["source_coverage_pct"], rec["effective_source_coverage_pct"], rec["ku_retained"] / rec["ku_total"] * 100.0, rec["overlap_redundancy_pct"], rec["synthetic_prefix_pct"]]
    c_a = [adp["source_coverage_pct"], adp["effective_source_coverage_pct"], adp["ku_retained"] / adp["ku_total"] * 100.0, adp["overlap_redundancy_pct"], adp["synthetic_prefix_pct"]]
    xc = np.arange(len(cov_cats))
    ax.bar(xc - w / 2, c_r, w, color=c_rec)
    ax.bar(xc + w / 2, c_a, w, color=c_adp, hatch="//", edgecolor="#FFFFFF")
    for i, (r_v, a_v) in enumerate(zip(c_r, c_a)):
        ax.text(i - w / 2, r_v + 1.5, f"{r_v:.1f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=c_rec)
        ax.text(i + w / 2, a_v + 1.5, f"{a_v:.1f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=c_adp)
    ax.set_xticks(xc)
    ax.set_xticklabels(cov_cats)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("(e) Độ phủ nguồn, bảo toàn 91/91 KU và dư thừa")

    # (f) Retrieved Context Volume & Storage Footprint
    ax = fig.add_subplot(gs[1, 2])
    style_axis_clean(ax)
    vol_cats = ["Ngữ cảnh\nTop-5 (tok)", "Ngữ cảnh\nTop-15 (tok)", "Ngữ cảnh\nTop-60 (tok)"]
    vr = [rec["retrieved_emb_tokens_k5_mean"], rec["retrieved_emb_tokens_k15_mean"], rec["retrieved_emb_tokens_k60_mean"]]
    va = [adp["retrieved_emb_tokens_k5_mean"], adp["retrieved_emb_tokens_k15_mean"], adp["retrieved_emb_tokens_k60_mean"]]
    xv = np.arange(len(vol_cats))
    ax.bar(xv - w / 2, vr, w, color=c_rec, label="Profile A (Recursive)")
    ax.bar(xv + w / 2, va, w, color=c_adp, hatch="//", edgecolor="#FFFFFF", label="Profile B (Adaptive)")
    for i, (r_v, a_v) in enumerate(zip(vr, va)):
        ax.text(i - w / 2, r_v + 150, f"{int(round(r_v)):,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=c_rec)
        ax.text(i + w / 2, a_v + 150, f"{int(round(a_v)):,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=c_adp)
    ax.set_xticks(xv)
    ax.set_xticklabels(vol_cats)
    ax.set_ylim(0, 8600)
    ax.set_ylabel("Trung bình số token ngữ cảnh")
    ax.set_title("(f) Dung lượng ngữ cảnh truy xuất & lưu trữ")
    mb_rec_j = rec["corpus_jsonl_bytes"] / (1024 * 1024)
    mb_adp_j = adp["corpus_jsonl_bytes"] / (1024 * 1024)
    mb_rec_v = rec["theoretical_vector_bytes_f32_384d"] / (1024 * 1024)
    mb_adp_v = adp["theoretical_vector_bytes_f32_384d"] / (1024 * 1024)
    mb_qdrant = rec["combined_qdrant_eval_bytes"] / (1024 * 1024)
    ax.text(
        0.02, 0.95,
        f"Lưu trữ tách biệt:\n"
        f"• JSONL: A={mb_rec_j:.2f} MiB | B={mb_adp_j:.2f} MiB\n"
        f"• Vector ước lượng (f32 384d): A={mb_rec_v:.2f} | B={mb_adp_v:.2f} MiB\n"
        f"• QdrantLocal SQLite dùng chung (thực tế): {mb_qdrant:.2f} MiB",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#3B82F6", alpha=0.94),
    )

    fig.text(
        0.5, 0.018,
        "Đánh đổi cốt lõi: Profile A (chunk lớn hơn, trung vị 421 ký tự) mang ngữ cảnh gấp ~2,1–2,3× mỗi K nhưng chịu 43,20% cắt cụt MiniLM và 20,33% dư thừa overlap; "
        "Profile B loại bỏ hoàn toàn cắt cụt embedding (0,00%) và giảm overlap còn 1,06% nhưng phân mảnh thành 13.390 chunk ngắn.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig01_danh_gia_chunking")


def generate_fig02_chat_luong_retrieval(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 02: Đánh giá chất lượng Retrieval (tổng hợp từ Stage 05 Hình 06–12)."""
    apply_consolidated_style()
    df = bundle.retrieval_summary
    fig = plt.figure(figsize=(24, 13.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.36, wspace=0.22, top=0.82, bottom=0.08, left=0.045, right=0.985)

    fig.suptitle(
        "HÌNH 02 — ĐÁNH GIÁ CHẤT LƯỢNG RETRIEVAL TRÊN 8 TỔ HỢP THỰC NGHIỆM VÀ 9 MỐC TOP-K (K = 1..60)",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.922,
        "Giao thức chính thức: Top-60 / candidate_top_k=120 trên 40 câu hỏi DSA dùng chung (320 lượt truy xuất; 0 ranking drift; Judged@60 = 1,000)",
        ha="center", fontsize=13.5, color="#374151",
    )
    fig.legend(
        handles=shared_combo_handles(False),
        loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=4, fontsize=11.5, frameon=True,
    )

    panels = [
        ("precision_at_k", "(a) Precision@K", "Precision@K", (0.0, 1.05)),
        ("pooled_recall_at_k", "(b) Pooled Recall@K", "Pooled Recall@K", (0.0, 1.05)),
        ("mrr_at_k", "(c) MRR@K", "MRR@K", (0.35, 1.05)),
        ("ndcg_at_k", "(d) nDCG@K (Phân bậc 0/1/2)", "nDCG@K", (0.20, 0.90)),
        ("map_at_k", "(e) MAP@K", "MAP@K", (0.15, 0.93)),
        ("f1_at_k", "(f) F1@K (Precision & Pooled Recall)", "F1@K", (0.05, 0.62)),
        ("hit_rate_at_k", "(g) Hit Rate@K", "Hit Rate@K", (0.35, 1.05)),
    ]

    for idx, (col, title, ylabel, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        style_axis_clean(ax)
        for cid in COMBO_ORDER:
            sub = df[df["combo_id"] == cid].sort_values("k")
            st = COMBO_STYLE[cid]
            ax.plot(
                K_POS, sub[col].values,
                color=st["color"], linestyle=st["ls"], linewidth=2.1,
                marker=st["marker"], markersize=5.8, markerfacecolor=st["mfc"],
                markeredgecolor=st["color"], markeredgewidth=1.4,
            )
        ax.set_xticks(K_POS)
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=10.5)
        ax.set_xlabel("Top-K")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title)

        # Annotate K=5 leader concisely in clean empty corner
        k5 = df[df["k"] == 5].set_index("combo_id")[col]
        best_cid = k5.idxmax()
        if idx in (0, 4):
            tx, ty, ha_b = 0.97, 0.90, "right"
        elif idx == 5:
            tx, ty, ha_b = 0.03, 0.89, "left"
        else:
            tx, ty, ha_b = 0.03, 0.06, "left"
        ax.text(
            tx, ty,
            f"K=5 cao nhất: {best_cid}={k5[best_cid]:.4f}",
            transform=ax.transAxes, ha=ha_b, va="bottom",
            fontsize=9.3, fontweight="bold", color="#111827",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.92),
        )

    # (h) Judged@K verification + compact K=5 & K=15 summary table
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.set_title("(h) Judged@60 = 1,000 & Bảng tóm tắt K=5 / K=15", pad=10)
    k5_df = df[df["k"] == 5].set_index("combo_id")
    k15_df = df[df["k"] == 15].set_index("combo_id")
    rows = []
    for cid in COMBO_ORDER:
        rows.append([
            cid,
            f"{k5_df.loc[cid, 'ndcg_at_k']:.4f}",
            f"{k5_df.loc[cid, 'precision_at_k']:.3f}",
            f"{k5_df.loc[cid, 'mrr_at_k']:.4f}",
            f"{k15_df.loc[cid, 'ndcg_at_k']:.4f}",
            f"{k15_df.loc[cid, 'pooled_recall_at_k']:.3f}",
            f"{k15_df.loc[cid, 'judged_at_k']:.3f}",
        ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["Tổ hợp", "nDCG@5", "P@5", "MRR@5", "nDCG@15", "Rec@15", "Judged"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 4:  # A4 highlight
            cell.set_facecolor("#FEF3C7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F9FAFB")
        cell.set_edgecolor("#D1D5DB")
    ax.text(
        0.5, 0.04,
        "Kiểm chứng gán nhãn: 0 vị trí chưa gán nhãn trên cả 2.880 ô (40×8×9).\n"
        "A4 dẫn đầu nDCG@5 (0,7053), MRR@5 (0,9875), nDCG@15 (0,7335).",
        ha="center", va="bottom", fontsize=9.8, color="#065F46", fontweight="semibold",
    )

    fig.text(
        0.5, 0.015,
        "Lưu ý phương pháp: Pooled Recall@K tính trên tập ứng viên đã gán nhãn Stage 03 (không phải toàn corpus). "
        "BM25 vượt rõ rệt Dense thuần ở cả hai profile; BGE Reranker tạo bước nhảy lớn nhất ở mọi thước đo xếp hạng.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig02_chat_luong_retrieval")


def generate_fig03_chat_luong_evidence(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 03: Đánh giá chất lượng Evidence (tổng hợp từ Stage 05 Hình 13–15)."""
    apply_consolidated_style()
    ev = bundle.evidence_summary
    fig = plt.figure(figsize=(23, 13.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.25, top=0.81, bottom=0.08, left=0.05, right=0.98)

    fig.suptitle(
        "HÌNH 03 — ĐÁNH GIÁ CHẤT LƯỢNG EVIDENCE NGUỒN XÁC THỰC VÀ ĐỘ PHỦ KNOWLEDGE UNIT",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.922,
        "Giao khoảng nguồn xác thực trên 91 Knowledge Units / 107 khoảng chuẩn hóa (tách biệt độ phủ nguồn source-backed với liên quan ngữ nghĩa LLM)",
        ha="center", fontsize=13.5, color="#374151",
    )
    fig.legend(
        handles=shared_combo_handles(False),
        loc="upper center", bbox_to_anchor=(0.5, 0.892), ncol=4, fontsize=11.5, frameon=True,
    )

    # (a) Evidence Recall@K
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = COMBO_STYLE[cid]
        ax.plot(K_POS, sub["evidence_recall_at_k"], color=st["color"], ls=st["ls"], lw=2.2,
                marker=st["marker"], ms=6, mfc=st["mfc"], mec=st["color"], mew=1.4)
    ax.set_xticks(K_POS)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Evidence Recall@K")
    ax.set_ylim(0.15, 1.03)
    ax.set_title("(a) Evidence Recall@K (Source-Backed)")
    ax.annotate("A4@15 = 0,9083", xy=(4, 0.9083), xytext=(1.2, 0.93), fontsize=10.2, fontweight="bold",
                color="#9A3412", arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFF7ED", edgecolor="#FB923C"))

    # (b) Evidence Precision@K
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = COMBO_STYLE[cid]
        ax.plot(K_POS, sub["evidence_precision_at_k"], color=st["color"], ls=st["ls"], lw=2.2,
                marker=st["marker"], ms=6, mfc=st["mfc"], mec=st["color"], mew=1.4)
    ax.set_xticks(K_POS)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Evidence Precision@K")
    ax.set_ylim(0.0, 0.82)
    ax.set_title("(b) Evidence Precision@K (Tỷ lệ chunk mang bằng chứng)")
    k5_ep = ev[ev["k"] == 5].set_index("combo_id")["evidence_precision_at_k"]
    ax.text(0.97, 0.93, f"K=5: A4={k5_ep['A4']:.4f} | B4={k5_ep['B4']:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor="#F59E0B"))

    # (c) Complete Evidence Rate@K
    ax = fig.add_subplot(gs[0, 2])
    style_axis_clean(ax)
    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = COMBO_STYLE[cid]
        ax.plot(K_POS, sub["complete_evidence_rate_at_k"], color=st["color"], ls=st["ls"], lw=2.2,
                marker=st["marker"], ms=6, mfc=st["mfc"], mec=st["color"], mew=1.4)
    ax.set_xticks(K_POS)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Complete Evidence Rate@K")
    ax.set_ylim(0.05, 1.03)
    ax.set_title("(c) Tỷ lệ câu hỏi đạt 100% bằng chứng (Complete Rate)")
    ax.annotate("A4@15 = 80,0%\nvs B4@15 = 47,5%", xy=(4, 0.80), xytext=(0.4, 0.82), fontsize=9.8, fontweight="bold",
                color="#9A3412", arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFF7ED", edgecolor="#FB923C"))

    # (d) Partial vs Complete Coverage Hierarchy at K=5
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    k5 = ev[ev["k"] == 5].set_index("combo_id").loc[COMBO_ORDER]
    x = np.arange(len(COMBO_ORDER))
    w = 0.21
    ax.bar(x - 1.5 * w, k5["complete_evidence_rate_at_k"], w, color="#1D4ED8", label="Complete Evidence")
    ax.bar(x - 0.5 * w, k5["evidence_recall_at_k"], w, color="#059669", label="Evidence Recall")
    ax.bar(x + 0.5 * w, k5["partial_evidence_coverage_at_k"], w, color="#F59E0B", label="Partial Coverage")
    ax.bar(x + 1.5 * w, k5["ku_coverage_at_k"], w, color="#9333EA", label="KU Any-Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tỷ lệ tại K = 5")
    ax.set_title("(d) Phân tầng độ phủ tại K=5 (Complete ≤ Recall ≤ Partial ≤ KU)")
    ax.legend(loc="upper right", fontsize=9.2, ncol=2)

    # (e) Source-Backed Evidence Recall@5 vs Semantic KU Coverage@5
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    w2 = 0.36
    b1 = ax.bar(x - w2 / 2, k5["evidence_recall_at_k"], w2, color="#0072B2", label="Source-Backed Evidence Recall@5")
    b2 = ax.bar(x + w2 / 2, k5["semantic_ku_recall_at_k"], w2, color="#009E73", hatch="//", edgecolor="#FFFFFF", label="Semantic KU Coverage@5 (Nhãn LLM)")
    for i, (v1, v2) in enumerate(zip(k5["evidence_recall_at_k"], k5["semantic_ku_recall_at_k"])):
        ax.text(i - w2 / 2, v1 + 0.015, f"{v1:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0072B2")
        ax.text(i + w2 / 2, v2 + 0.015, f"{v2:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#009E73")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Tỷ lệ phủ tại K = 5")
    ax.set_title("(e) Bằng chứng nguồn xác thực vs Độ phủ ngữ nghĩa KU")
    ax.legend(loc="upper right", fontsize=9.8)

    # (f) Evidence Redundancy@K
    ax = fig.add_subplot(gs[1, 2])
    style_axis_clean(ax)
    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = COMBO_STYLE[cid]
        ax.plot(K_POS, sub["evidence_redundancy_at_k"], color=st["color"], ls=st["ls"], lw=2.2,
                marker=st["marker"], ms=6, mfc=st["mfc"], mec=st["color"], mew=1.4)
    ax.set_xticks(K_POS)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Evidence Redundancy@K")
    ax.set_ylim(-0.005, 0.22)
    ax.set_title("(f) Trùng lặp bằng chứng (Evidence Redundancy@K)")
    ax.text(
        0.04, 0.92,
        "Profile B (nét đứt) gần như loại bỏ\ntrùng lặp bằng chứng (0,000–0,008 tại K=5)\nso với Profile A (0,038–0,061 tại K=5).",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ECFDF5", edgecolor="#10B981", alpha=0.92),
    )

    fig.text(
        0.5, 0.015,
        "Phát hiện chính: A4 đạt Evidence Recall@15 = 0,9083 và Complete Evidence@15 = 80,0%, vượt rõ B4 (47,5%) do chunk ngắn làm phân mảnh bằng chứng đa câu; "
        "Semantic KU Coverage cao hơn Source-Backed Recall vì corpus chứa nhiều đoạn giải thích cùng kiến thức ngoài khoảng gốc đã đăng ký.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig03_chat_luong_evidence")


def generate_fig04_hieu_nang_va_tai_nguyen(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 04: Hiệu năng và tài nguyên Retrieval (tổng hợp từ Stage 05 Hình 16–20)."""
    apply_consolidated_style()
    lat = bundle.latency_summary
    stg = bundle.stage_latency
    cold = bundle.cold_start_latency
    prod = bundle.production_profile_latency.set_index("combo_id").loc[COMBO_ORDER]
    res = bundle.resource_usage.set_index("combo_id").loc[COMBO_ORDER]

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.25, top=0.81, bottom=0.08, left=0.05, right=0.98)

    fig.suptitle(
        "HÌNH 04 — HIỆU NĂNG ĐỘ TRỄ RETRIEVAL (TÁCH BIỆT 4 GIAO THỨC) VÀ TÀI NGUYÊN HỆ THỐNG",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.922,
        "Cảnh báo giao thức: Toàn bộ số liệu dưới đây là độ trễ TRUY XUẤT NỘI BỘ (retrieval-only), KHÔNG phải độ trễ HTTP End-to-End (xem Hình 09) | RTX 5060 (7,96 GB VRAM), 15,71 GB RAM",
        ha="center", fontsize=13.2, color="#991B1B", fontweight="semibold",
    )
    fig.legend(
        handles=shared_combo_handles(False),
        loc="upper center", bbox_to_anchor=(0.5, 0.892), ncol=4, fontsize=11.5, frameon=True,
    )

    # (a) Protocol 1: Main Warm Mean Latency@K (14,400 calls)
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    for cid in COMBO_ORDER:
        sub = lat[lat["combo_id"] == cid].sort_values("k")
        st = COMBO_STYLE[cid]
        ax.plot(K_POS, sub["mean_ms"], color=st["color"], ls=st["ls"], lw=2.1,
                marker=st["marker"], ms=5.8, mfc=st["mfc"], mec=st["color"], mew=1.4)
    ax.set_xticks(K_POS)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Độ trễ trung bình — Mean (ms)")
    ax.set_ylim(0, 4200)
    ax.set_title("(a) Giao thức 1: Main Warm Mean Latency (14.400 lượt, pool=120)")
    ax.text(0.03, 0.93, "Độ trễ gần như phẳng theo K vì toàn bộ\npool=120 được chấm điểm trước khi cắt K.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    # (b) Protocol 1: Main Warm Tail Latency (P50 / Mean / P95 / P99 at K=5)
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    k5_lat = lat[lat["k"] == 5].set_index("combo_id").loc[COMBO_ORDER]
    x = np.arange(len(COMBO_ORDER))
    w = 0.21
    ax.bar(x - 1.5 * w, k5_lat["p50_ms"], w, color="#0072B2", label="P50")
    ax.bar(x - 0.5 * w, k5_lat["mean_ms"], w, color="#009E73", label="Mean")
    ax.bar(x + 0.5 * w, k5_lat["p95_ms"], w, color="#E69F00", label="P95")
    ax.bar(x + 1.5 * w, k5_lat["p99_ms"], w, color="#D55E00", label="P99")
    for i, cid in enumerate(COMBO_ORDER):
        v = k5_lat.loc[cid, "p95_ms"]
        ax.text(i + 0.5 * w, v + 120, f"{int(round(v))}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#92400E")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylabel("Độ trễ tại K = 5 (ms)")
    ax.set_ylim(0, max(k5_lat["p99_ms"]) * 1.18)
    ax.set_title("(b) Giao thức 1: Phân vị đuôi P50/Mean/P95/P99 tại K=5")
    ax.legend(loc="upper left", ncol=4, fontsize=9.8)
    ax.annotate("B4 P99 = 23,1s\n(tái tạo cache BM25 TTL 300s)", xy=(7 + 1.5 * w, k5_lat.loc["B4", "p99_ms"]),
                xytext=(3.5, 19000), fontsize=9.2, fontweight="bold", color="#991B1B",
                arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#EF4444"))

    # (c) Protocol 2: Stage-Level Breakdown (640 calls, K=5)
    ax = fig.add_subplot(gs[0, 2])
    style_axis_clean(ax)
    s5 = stg[stg["k"] == 5].set_index("combo_id").loc[COMBO_ORDER]
    comps = [
        ("query_embed_mean_ms", "Query Embed", "#56B4E9"),
        ("bm25_search_mean_ms", "BM25 Search", "#0072B2"),
        ("qdrant_dense_search_mean_ms", "Qdrant Dense", "#E69F00"),
        ("rrf_fusion_mean_ms", "RRF Fusion", "#009E73"),
        ("rerank_mean_ms", "BGE Rerank", "#D55E00"),
    ]
    bottom = np.zeros(len(COMBO_ORDER))
    for col, lbl, clr in comps:
        vals = s5[col].values
        ax.bar(x, vals, 0.58, bottom=bottom, color=clr, label=lbl)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylabel("Thời gian công đoạn trung bình (ms)")
    ax.set_ylim(0, 4250)
    ax.set_title("(c) Giao thức 2: Phân rã công đoạn (640 lượt, K=5, pool=120)")
    ax.legend(loc="upper left", fontsize=9.2, ncol=2)
    a4_rr = s5.loc["A4", "rerank_mean_ms"]
    b4_rr = s5.loc["B4", "rerank_mean_ms"]
    ax.text(
        0.98, 0.94,
        f"Riêng công đoạn Reranking:\nB4 = {b4_rr:.1f} ms vs A4 = {a4_rr:.1f} ms\n(B4 nhanh hơn 41,2% ở bước rerank,\ntổng P95 chỉ giảm ~13,0%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.94),
    )

    # (d) Protocol 4: Cold-Start Latency (24 trials, SECONDS)
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    cg = cold.groupby("combo_id")[["model_load_ms", "bm25_build_ms", "first_retrieval_ms", "total_cold_start_ms"]].mean().loc[COMBO_ORDER] / 1000.0
    b_m = cg["model_load_ms"].values
    b_b = cg["bm25_build_ms"].values
    b_f = cg["first_retrieval_ms"].values
    stack_top = b_m + b_b + b_f
    ax.bar(x, b_m, 0.56, color="#6366F1", label="Nạp mô hình (s)")
    ax.bar(x, b_b, 0.56, bottom=b_m, color="#0072B2", label="Dựng chỉ mục BM25 (s)")
    ax.bar(x, b_f, 0.56, bottom=b_m + b_b, color="#F59E0B", label="Truy xuất đầu tiên (s)")
    for i, (top_h, tot) in enumerate(zip(stack_top, cg["total_cold_start_ms"].values)):
        ax.text(i, top_h + 0.8, f"{tot:.1f}s", ha="center", va="bottom", fontsize=9.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylabel("Thời gian khởi động nguội (giây — s)")
    ax.set_ylim(0, max(stack_top) * 1.18)
    ax.set_title("(d) Giao thức 4: Cold-Start (24 thử nghiệm, ĐƠN VỊ GIÂY)")
    ax.legend(loc="upper left", fontsize=9.6)

    # (e) Protocol 3: Production-Parameter Retrieval Latency (1,600 calls, K=5/pool=10)
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    w3 = 0.27
    ax.bar(x - w3, prod["p50_ms"], w3, color="#0072B2", label="Prod-Param P50")
    ax.bar(x, prod["mean_ms"], w3, color="#009E73", label="Prod-Param Mean")
    ax.bar(x + w3, prod["p95_ms"], w3, color="#D55E00", label="Prod-Param P95")
    for i, cid in enumerate(COMBO_ORDER):
        v = prod.loc[cid, "p95_ms"]
        ax.text(i + w3, v + 35, f"{int(round(v))}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#9A3412")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylabel("Độ trễ truy xuất (ms)")
    ax.set_ylim(0, max(prod["p95_ms"]) * 1.34)
    ax.set_title("(e) Giao thức 3: Tham số Production (1.600 lượt, K=5, pool=10)")
    ax.legend(loc="upper left", fontsize=9.8)
    ax.text(
        0.98, 0.96,
        f"A4 P95 giảm: {k5_lat.loc['A4', 'p95_ms']:.0f} → {prod.loc['A4', 'p95_ms']:.0f} ms\n(pool 120 → pool 10)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF5", edgecolor="#10B981"),
    )

    # (f) Resource Usage: GPU VRAM & Process RAM
    ax = fig.add_subplot(gs[1, 2])
    style_axis_clean(ax)
    w4 = 0.27
    ax.bar(x - w4, res["peak_gpu_allocated_max_mb"], w4, color="#D55E00", label="GPU VRAM Allocated (MB)")
    ax.bar(x, res["peak_gpu_reserved_max_mb"], w4, color="#E69F00", label="GPU VRAM Reserved (MB)")
    ax.bar(x + w4, res["process_rss_mean_mb"], w4, color="#0072B2", label="Process RAM RSS (MB)")
    for i, cid in enumerate(COMBO_ORDER):
        vg = res.loc[cid, "peak_gpu_reserved_max_mb"]
        if vg > 0:
            ax.text(i, vg + 60, f"{int(round(vg))}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#92400E")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylabel("Bộ nhớ tiêu thụ (MB)")
    ax.set_ylim(0, 3850)
    ax.set_title("(f) Tài nguyên GPU VRAM và RAM tiến trình (0 CUDA OOM)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(
        0.98, 0.96,
        "BM25 thuần (A1, B1): 0 MB VRAM\nReranker đỉnh ~2,69 GB (~34% RTX 5060 8 GB)\nRAM RSS ổn định: 965–993 MB",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#EFF6FF", edgecolor="#3B82F6"),
    )

    fig.text(
        0.5, 0.015,
        "Tách biệt bắt buộc: Main Warm (14.400 lượt, ms) ≠ Stage Breakdown (640 lượt, ms) ≠ Production-Param Retrieval (1.600 lượt, ms) ≠ Cold-Start (24 thử nghiệm, giây). "
        "Cải thiện tốc độ ~41,2% của B4 chỉ áp dụng cho công đoạn BGE Reranking cô lập.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig04_hieu_nang_va_tai_nguyen")


def generate_fig05_chat_luong_cau_tra_loi(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 05: Chất lượng sinh câu trả lời — 10 thước đo Controlled Agent Generation (Stage 05 Hình 21)."""
    apply_consolidated_style()
    df = bundle.answer_summary.set_index("combo_id").loc[COMBO_ORDER + ["NO_RAG"]]

    metrics_meta = [
        ("answer_correctness_norm", "Correctness\n(↑)", False),
        ("answer_relevancy_norm", "Relevancy\n(↑)", False),
        ("faithfulness_norm", "Faithfulness\n(↑)", False),
        ("answer_completeness_norm", "Completeness\n(↑)", False),
        ("citation_accuracy_norm", "Citation Acc.\n(↑)", False),
        ("citation_validity", "Citation Val.\n(↑)", False),
        ("unsupported_claim_rate", "Unsupported\n(↓)", True),
        ("contradiction_rate", "Contradiction\n(↓)", True),
        ("complexity_correctness_norm", "Complexity\n(↑)", False),
        ("code_correctness_norm", "Code\n(↑)", False),
    ]

    fig = plt.figure(figsize=(23, 11.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.48, 1.0], wspace=0.20, top=0.84, bottom=0.11, left=0.055, right=0.98)

    fig.suptitle(
        "HÌNH 05 — CHẤT LƯỢNG SINH CÂU TRẢ LỜI TRÊN 10 THƯỚC ĐO (CONTROLLED AGENT GENERATION)",
        fontsize=23, fontweight="bold", y=0.955, color="#111827",
    )
    fig.text(
        0.5, 0.905,
        "Giao thức Controlled Agent (antigravity_agent_flash, ngữ cảnh 2.048 tokens, 40 câu × 9 cấu hình = 360 câu chính, chấm mù antigravity_agent_pro) | KHÔNG đại diện Live Groq production (xem Hình 09)",
        ha="center", fontsize=12.8, color="#374151",
    )

    # (a) Full 9x10 Heatmap preserving N/A verbatim
    ax = fig.add_subplot(gs[0, 0])
    rows = COMBO_ORDER + ["NO_RAG"]
    mat_color = np.full((len(rows), len(metrics_meta)), np.nan)
    mat_text = [["" for _ in metrics_meta] for _ in rows]

    for i, cid in enumerate(rows):
        for j, (col, _, lower_better) in enumerate(metrics_meta):
            raw = df.loc[cid, col]
            if pd.isna(raw) or str(raw).strip().upper() == "N/A":
                mat_text[i][j] = "N/A"
                mat_color[i, j] = np.nan
            else:
                val = float(raw)
                mat_text[i][j] = f"{val:.3f}"
                mat_color[i, j] = (1.0 - val) if lower_better else val

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax.imshow(mat_color, cmap=cmap, aspect="auto", vmin=0.55, vmax=1.0)
    for i in range(len(rows)):
        for j in range(len(metrics_meta)):
            txt = mat_text[i][j]
            if txt == "N/A":
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#4B5563")
            else:
                score_c = mat_color[i, j]
                tc = "#FFFFFF" if score_c >= 0.86 else "#111827"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10.2, fontweight="bold", color=tc)

    ax.set_xticks(range(len(metrics_meta)))
    ax.set_xticklabels([m[1] for m in metrics_meta], fontsize=10.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([COMBO_STYLE[r]["label"] for r in rows], fontsize=11)
    ax.set_title("(a) Ma trận toàn bộ 10 thước đo (bảo toàn N/A của NO_RAG; thang màu hướng tốt)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Điểm chất lượng chuẩn hóa (1,0 = tốt nhất)", fontsize=10.5)

    # (b) Explicit comparison highlighting NO_RAG > A4 Correctness and A1 > A4 Faithfulness / Citation Accuracy
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    focus_combos = ["A1", "A4", "B4", "NO_RAG"]
    focus_metrics = [
        ("answer_correctness_norm", "Correctness"),
        ("answer_relevancy_norm", "Relevancy"),
        ("faithfulness_norm", "Faithfulness"),
        ("citation_accuracy_norm", "Citation Acc."),
        ("answer_completeness_norm", "Completeness"),
    ]
    xf = np.arange(len(focus_metrics))
    wf = 0.19
    for idx_c, cid in enumerate(focus_combos):
        st = COMBO_STYLE[cid]
        vals = []
        is_na = []
        for col, _ in focus_metrics:
            raw = df.loc[cid, col]
            if pd.isna(raw) or str(raw).strip().upper() == "N/A":
                vals.append(0.0)
                is_na.append(True)
            else:
                vals.append(float(raw))
                is_na.append(False)
        offset = (idx_c - 1.5) * wf
        bars = ax.bar(
            xf + offset, vals, wf,
            color=st["color"], hatch=st["hatch"], edgecolor="#FFFFFF" if st["hatch"] else st["color"],
            label=st["label"],
        )
        for b, v, na in zip(bars, vals, is_na):
            if na:
                ax.text(b.get_x() + b.get_width() / 2, 0.03, "N/A", ha="center", va="bottom",
                        fontsize=8.8, fontweight="bold", color="#4B5563", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.6, fontweight="bold", rotation=90)

    ax.set_xticks(xf)
    ax.set_xticklabels([m[1] for m in focus_metrics], fontsize=11.5)
    ax.set_ylim(0.0, 1.38)
    ax.set_ylabel("Điểm chuẩn hóa (0 – 1)")
    ax.set_title("(b) So sánh trọng tâm: A4 KHÔNG thắng mọi thước đo")
    ax.legend(loc="upper right", fontsize=9.8)
    ax.text(
        0.02, 0.97,
        "Công bố trung thực:\n"
        "• NO_RAG > A4 về Correctness (0,969 vs 0,925) & Relevancy (1,000 vs 0,956)\n"
        "• A1 > A4 về Faithfulness (0,925 vs 0,919) & Citation Acc. (0,850 vs 0,819)\n"
        "• A4 đạt Contradiction = 0,000 vs NO_RAG = 0,018",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.95),
    )

    fig.text(
        0.5, 0.02,
        "Lưu ý khoa học: Citation Validity = 1,000 ở mọi cấu hình RAG chỉ phản ánh mã trích dẫn hợp lệ về mặt hình thức, không đồng nghĩa 100% Citation Accuracy (0,750–0,850). "
        "Kiểm toán chấm mù giữ nguyên human_validated = false.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig05_chat_luong_cau_tra_loi")
