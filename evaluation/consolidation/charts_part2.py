"""Consolidated Vietnamese Multi-Panel Figures 06–10 (Ablation/Sensitivity, Statistics/Trade-off, Reliability/Error, Live Groq/HTTP E2E, Live Features/API Stability)."""

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
    apply_consolidated_style,
    save_consolidated_figure,
    style_axis_clean,
)
from evaluation.path_resolver import resolve_ac04_file
from evaluation.stage05.data_loader import Stage04Bundle


def generate_fig06_ablation_va_do_nhay(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 06: Ablation và Phân tích độ nhạy (tổng hợp từ Stage 05 Hình 22–24)."""
    apply_consolidated_style()
    ab = bundle.ablation_results
    rag = bundle.rag_contribution
    cp = bundle.candidate_pool_sensitivity
    tk = bundle.top_k_sensitivity
    cb = bundle.context_budget_sensitivity

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.25, top=0.88, bottom=0.08, left=0.07, right=0.97)

    fig.suptitle(
        "HÌNH 06 — NGHIÊN CỨU ABLATION VÀ PHÂN TÍCH ĐỘ NHẠY (TOP-K, CANDIDATE POOL, CONTEXT BUDGET)",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.92,
        "Kiểm định Bootstrap ghép cặp (10.000 lượt, seed=42, hiệu chỉnh Holm-Bonferroni) | Tách biệt độ nhạy Candidate Pool (single-pass) và Context Budget thăm dò (n=10)",
        ha="center", fontsize=13, color="#374151",
    )

    # (a) Retrieval & Chunking Ablation (nDCG@5)
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    sub_ab = ab[(ab["k"] == 5) & (ab["metric"] == "ndcg_at_k")].copy()
    order_ids = ["R1_A4_vs_A3", "R2_B4_vs_B3", "H2_A3_vs_A2", "H4_B3_vs_B2", "H1_A3_vs_A1", "H3_B3_vs_B1", "C1_B1_vs_A1", "C2_B2_vs_A2", "C3_B3_vs_A3", "C4_B4_vs_A4"]
    sub_ab = sub_ab.set_index("comparison_id").loc[order_ids].reset_index()
    labels_ab = [
        "Rerank Rec (A4−A3)", "Rerank Adap (B4−B3)", "RRF vs Dense (A3−A2)", "RRF vs Dense (B3−B2)",
        "RRF vs BM25 (A3−A1)", "RRF vs BM25 (B3−B1)", "Adap vs Rec BM25 (B1−A1)", "Adap vs Rec Dense (B2−A2)",
        "Adap vs Rec RRF (B3−A3)", "Adap vs Rec Rerank (B4−A4)",
    ]
    y = np.arange(len(sub_ab))
    vals = sub_ab["abs_improvement"].values
    err_lo = vals - sub_ab["ci_95_low"].values
    err_hi = sub_ab["ci_95_high"].values - vals
    colors = ["#009E73" if v >= 0 else "#D55E00" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.88, height=0.65)
    ax.errorbar(vals, y, xerr=[np.maximum(err_lo, 0), np.maximum(err_hi, 0)], fmt="none", ecolor="#111827", elinewidth=1.4, capsize=3.5)
    ax.axvline(0, color="#111827", lw=1.2)
    for i, row in sub_ab.iterrows():
        v = row["abs_improvement"]
        p = row["p_value_holm"]
        sig = "*" if row["significant_holm_005"] else "ns"
        xpos = row["ci_95_high"] + 0.012 if v >= 0 else row["ci_95_low"] - 0.012
        ha = "left" if v >= 0 else "right"
        ax.text(xpos, i, f"{v:+.3f} (p={p:.3f}, {sig})", va="center", ha=ha, fontsize=9.2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_ab, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.34, 0.42)
    ax.set_xlabel("Chênh lệch nDCG@5 ghép cặp (Δ ± 95% CI)")
    ax.set_title("(a) Ablation trên nDCG@5 (BM25 vs Dense, RRF, Reranker, Chunking)")

    # (b) RAG vs NO_RAG Paired Correctness & Contradiction
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    rc_corr = rag[rag["metric"] == "answer_correctness_norm"].set_index("combo_id").loc[COMBO_ORDER]
    x = np.arange(len(COMBO_ORDER))
    diffs = rc_corr["abs_diff"].values
    lo = diffs - rc_corr["ci_95_low"].values
    hi = rc_corr["ci_95_high"].values - diffs
    clrs = [COMBO_STYLE[c]["color"] for c in COMBO_ORDER]
    hatches = [COMBO_STYLE[c]["hatch"] for c in COMBO_ORDER]
    bars = ax.bar(x, diffs, 0.56, color=clrs, alpha=0.88)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            b.set_edgecolor("#FFFFFF")
    ax.errorbar(x, diffs, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor="#111827", elinewidth=1.4, capsize=3.5)
    ax.axhline(0, color="#111827", lw=1.2)
    for i, cid in enumerate(COMBO_ORDER):
        v = rc_corr.loc[cid, "abs_diff"]
        p = rc_corr.loc[cid, "p_value_raw"]
        ax.text(i, rc_corr.loc[cid, "ci_95_low"] - 0.015, f"{v:+.3f}\np={p:.2f}", ha="center", va="top", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylim(-0.68, 0.08)
    ax.set_ylabel("Δ Correctness (RAG − NO_RAG) ± 95% CI")
    ax.set_title("(b) Đóng góp RAG vs NO_RAG (Controlled Agent, n=40)")
    ax.text(
        0.02, 0.03,
        "A4 − NO_RAG = −0,044 (p=0,238, không có ý nghĩa ở α=0,05; KHÔNG chứng minh tương đương).\n"
        "Lợi ích RAG: giảm Contradiction từ 0,018 xuống 0,000 ở A4 và có trích dẫn nguồn xác thực.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor="#F59E0B"),
    )

    # (c) Candidate Pool Sensitivity (640 single-pass calls) + Top-K Operating Curve
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    ax2 = ax.twinx()
    pools = [10, 30, 60, 120]
    p_pos = np.arange(len(pools))
    for cid in ["A4", "B4"]:
        sub = cp[cp["combo_id"] == cid].sort_values("candidate_top_k")
        st = COMBO_STYLE[cid]
        ax.plot(p_pos, sub["ndcg_at_5"], color=st["color"], ls=st["ls"], lw=2.4, marker="o", ms=7, mfc=st["mfc"], label=f"{cid} nDCG@5 (trái)")
        ax2.plot(p_pos, sub["mean_latency_ms"], color="#0072B2" if cid == "A4" else "#009E73", ls=st["ls"], lw=2.0, marker="s", ms=6, label=f"{cid} Mean Latency ms (phải)")
    ax.set_xticks(p_pos)
    ax.set_xticklabels([f"pool={p}" for p in pools])
    ax.set_ylabel("nDCG@5 (trục trái)", color="#B45309")
    ax2.set_ylabel("Single-Pass Mean Latency (ms, trục phải)", color="#0072B2")
    ax.set_ylim(0.50, 0.78)
    ax2.set_ylim(0, 4200)
    ax.set_title("(c) Độ nhạy Candidate Pool (640 lượt single-pass; tách khỏi Main Warm)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9.6)
    ax.annotate(
        "Điểm cân bằng thăm dò A4 pool=30:\ngiữ 99,8% nDCG@5 (0,7039 vs 0,7053)\ngiảm độ trễ 3.486 → 1.628 ms",
        xy=(1, 0.7039), xytext=(1.2, 0.725), fontsize=9.6, fontweight="bold", color="#9A3412",
        arrowprops=dict(arrowstyle="->", color="#9A3412", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFF7ED", edgecolor="#FB923C"),
    )

    # (d) Exploratory Context Budget Sensitivity (1024 / 2048 / 4096, n=10)
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    budgets = [1024, 2048, 4096]
    xb = np.arange(len(budgets))
    a4_cb = cb[cb["combo_id"] == "A4"].sort_values("token_budget")
    b4_cb = cb[cb["combo_id"] == "B4"].sort_values("token_budget")
    ax.plot(xb, a4_cb["answer_correctness_norm"], color="#D55E00", ls="-", lw=2.4, marker="D", ms=7, label="A4 Correctness")
    ax.plot(xb, a4_cb["faithfulness_norm"], color="#0072B2", ls="-", lw=2.4, marker="o", ms=7, label="A4 Faithfulness")
    ax.plot(xb, a4_cb["citation_accuracy_norm"], color="#009E73", ls="-", lw=2.2, marker="^", ms=7, label="A4 Citation Acc.")
    ax.plot(xb, b4_cb["answer_correctness_norm"], color="#D55E00", ls="--", lw=2.0, marker="D", ms=6.5, mfc="white", label="B4 Correctness")
    ax.plot(xb, b4_cb["faithfulness_norm"], color="#0072B2", ls="--", lw=2.0, marker="o", ms=6.5, mfc="white", label="B4 Faithfulness")
    for i, row in enumerate(a4_cb.itertuples()):
        ax.text(i, row.answer_correctness_norm + 0.015, f"{row.answer_correctness_norm:.3f}", ha="center", fontsize=9.5, fontweight="bold", color="#D55E00")
        ax.text(i, row.citation_accuracy_norm - 0.032, f"{row.citation_accuracy_norm:.3f}", ha="center", fontsize=9.2, fontweight="bold", color="#009E73")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{b} tokens\n(thực tế A4≈{int(round(t))})" for b, t in zip(budgets, a4_cb["mean_actual_context_tokens"])])
    ax.set_ylim(0.58, 1.15)
    ax.set_ylabel("Điểm chất lượng chuẩn hóa (0 – 1)")
    ax.set_title("(d) Độ nhạy Context Budget THĂM DÒ (n=10 câu, Controlled Agent)")
    ax.legend(loc="lower left", ncol=2, fontsize=9.5)
    ax.text(
        0.98, 0.97,
        "2.048 tokens tối ưu cho A4 (Corr=0,950, Faith=0,925);\n4.096 tokens gây suy giảm Faithfulness (0,825) & CitAcc (0,700).\nCỡ mẫu nhỏ (n=10) — mang tính thăm dò.",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#EFF6FF", edgecolor="#3B82F6"),
    )

    fig.text(
        0.5, 0.015,
        "Ghi chú Top-K sensitivity (toàn bộ 9 mốc K=1..60 xem Hình 02 & Hình 03): K=5–10 tối ưu cho Chat ngắn gọn; K=15 đạt 80,0% Complete Evidence ở A4 trước khi bão hòa sau K=30.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig06_ablation_va_do_nhay")


def generate_fig07_thong_ke_va_tradeoff(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 07: Thống kê và Cân bằng Hiệu năng (tổng hợp từ Stage 05 Hình 25–26)."""
    apply_consolidated_style()
    pf = bundle.pareto_frontier
    ci = bundle.confidence_intervals
    wtl = bundle.win_tie_loss
    ann = bundle.annotation_sensitivity
    subg = bundle.subgroup_metrics

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.26, top=0.88, bottom=0.08, left=0.055, right=0.98)

    fig.suptitle(
        "HÌNH 07 — PHÂN TÍCH THỐNG KÊ VÀ BIÊN TỐI ƯU CÂN BẰNG CHẤT LƯỢNG – ĐỘ TRỄ – TÀI NGUYÊN",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.92,
        "Bootstrap 10.000 mẫu (seed=42, n=40) | Biên Pareto khớp cùng giao thức Main Warm (pool=120), KHÔNG dùng điểm trọng số tổng hợp tùy ý",
        ha="center", fontsize=13.2, color="#374151",
    )

    # (a) Pareto Frontier: nDCG@5 vs P95 Latency
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    p5 = pf[pf["k"] == 5].set_index("combo_id").loc[COMBO_ORDER]
    for cid in COMBO_ORDER:
        row = p5.loc[cid]
        st = COMBO_STYLE[cid]
        size = 110 + row["peak_gpu_allocated_max_mb"] * 0.14
        ax.scatter(row["p95_latency_ms"], row["ndcg_at_k"], s=size, color=st["color"],
                   marker=st["marker"], edgecolors="#111827" if row["is_pareto_optimal"] else st["color"],
                   linewidths=2.2 if row["is_pareto_optimal"] else 1.2, alpha=0.9, zorder=4)
        dx = 95 if cid not in ("A4", "B4") else -110
        dy_map = {"A1": 0.018, "A2": -0.025, "A3": 0.016, "A4": 0.018, "B1": 0.018, "B2": 0.012, "B3": 0.014, "B4": 0.018}
        dy = dy_map.get(cid, 0.015)
        ha = "left" if dx > 0 else "right"
        ax.text(row["p95_latency_ms"] + dx, row["ndcg_at_k"] + dy, f"{cid} ({row['ndcg_at_k']:.3f})",
                fontsize=9.8, fontweight="bold" if row["is_pareto_optimal"] else "normal", ha=ha)
    pareto_pts = p5[p5["is_pareto_optimal"]].sort_values("p95_latency_ms")
    ax.plot(pareto_pts["p95_latency_ms"], pareto_pts["ndcg_at_k"], color="#111827", ls="--", lw=1.6, label="Biên Pareto (A1, B4, A4)", zorder=3)
    ax.set_xlabel("Main Warm P95 Latency tại K=5 (ms)")
    ax.set_ylabel("nDCG@5")
    ax.set_xlim(200, 4200)
    ax.set_ylim(0.28, 0.78)
    ax.set_title("(a) Biên Pareto: nDCG@5 vs Độ trễ P95 (cỡ điểm = VRAM)")
    ax.legend(loc="lower right", fontsize=10)

    # (b) Pareto Frontier: Evidence Recall@15 vs P95 Latency
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    p15 = pf[pf["k"] == 15].set_index("combo_id").loc[COMBO_ORDER]
    for cid in COMBO_ORDER:
        row = p15.loc[cid]
        st = COMBO_STYLE[cid]
        ax.scatter(row["p95_latency_ms"], row["evidence_recall_at_k"], s=140, color=st["color"],
                   marker=st["marker"], edgecolors="#111827" if cid in ("A1", "B4", "A4") else st["color"],
                   linewidths=2.0 if cid in ("A1", "B4", "A4") else 1.2, zorder=4)
        dx = 95 if cid not in ("A4", "B4") else -110
        dy = 0.014 if cid != "A3" else -0.028
        ax.text(row["p95_latency_ms"] + dx, row["evidence_recall_at_k"] + dy, f"{cid} ({row['evidence_recall_at_k']:.3f})",
                fontsize=9.8, fontweight="bold" if cid in ("A1", "B4", "A4") else "normal", ha="left" if dx > 0 else "right")
    p15_front = p15.loc[["A1", "A4"]].sort_values("p95_latency_ms")
    ax.plot(p15_front["p95_latency_ms"], p15_front["evidence_recall_at_k"], color="#111827", ls="--", lw=1.6, label="Biên Pareto EvRecall@15 (A1, A4)")
    ax.set_xlabel("Main Warm P95 Latency tại K=15 (ms)")
    ax.set_ylabel("Evidence Recall@15")
    ax.set_xlim(200, 4200)
    ax.set_ylim(0.32, 0.98)
    ax.set_title("(b) Cân bằng Evidence Recall@15 vs Độ trễ P95")
    ax.legend(loc="lower right", fontsize=10)

    # (c) Bootstrap 95% CIs (nDCG@5 and EvRecall@15)
    ax = fig.add_subplot(gs[0, 2])
    style_axis_clean(ax)
    ci_ndcg = ci[(ci["k"] == 5) & (ci["metric"] == "ndcg_at_k")].set_index("combo_id").loc[COMBO_ORDER]
    ci_ev = ci[(ci["k"] == 15) & (ci["metric"] == "evidence_recall_at_k")].set_index("combo_id").loc[COMBO_ORDER]
    y = np.arange(len(COMBO_ORDER))
    ax.errorbar(ci_ndcg["mean"], y - 0.16, xerr=[ci_ndcg["mean"] - ci_ndcg["ci_95_low"], ci_ndcg["ci_95_high"] - ci_ndcg["mean"]],
                fmt="o", color="#D55E00", capsize=4, lw=2.0, ms=6.5, label="nDCG@5 (95% CI)")
    ax.errorbar(ci_ev["mean"], y + 0.16, xerr=[ci_ev["mean"] - ci_ev["ci_95_low"], ci_ev["ci_95_high"] - ci_ev["mean"]],
                fmt="s", color="#0072B2", capsize=4, lw=2.0, ms=6.5, label="EvRecall@15 (95% CI)")
    ax.set_yticks(y)
    ax.set_yticklabels(COMBO_ORDER, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.20, 1.02)
    ax.set_xlabel("Ước lượng trung bình và khoảng tin cậy 95% Bootstrap")
    ax.set_title("(c) Khoảng tin cậy 95% Bootstrap (10.000 mẫu, n=40)")
    ax.legend(loc="lower left", fontsize=10)

    # (d) Paired Win–Tie–Loss (n=40, nDCG@5)
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    wtl_sel = wtl[(wtl["k"] == 5) & (wtl["metric"] == "ndcg_at_k")].set_index("comparison_id").loc[
        ["R1_A4_vs_A3", "R2_B4_vs_B3", "H1_A3_vs_A1", "H2_A3_vs_A2", "C1_B1_vs_A1", "C4_B4_vs_A4"]
    ]
    w_lbls = ["A4 vs A3", "B4 vs B3", "A3 vs A1", "A3 vs A2", "B1 vs A1", "B4 vs A4"]
    yw = np.arange(len(w_lbls))
    ax.barh(yw, wtl_sel["wins"], color="#009E73", label="Thắng (Wins)")
    ax.barh(yw, wtl_sel["ties"], left=wtl_sel["wins"], color="#9CA3AF", label="Hòa (Ties)")
    ax.barh(yw, wtl_sel["losses"], left=wtl_sel["wins"] + wtl_sel["ties"], color="#D55E00", label="Thua (Losses)")
    for i, row in enumerate(wtl_sel.itertuples()):
        ax.text(20, i, f"{int(row.wins)}W – {int(row.ties)}T – {int(row.losses)}L", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#FFFFFF",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#111827", alpha=0.65, edgecolor="none"))
    ax.set_yticks(yw)
    ax.set_yticklabels(w_lbls, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel("Số câu hỏi (n = 40, tol = 1e-4)")
    ax.set_title("(d) Tỷ số Thắng – Hòa – Thua ghép cặp (nDCG@5)")
    ax.legend(loc="lower right", ncol=3, fontsize=9.5)

    # (e) Annotation Adjudication Sensitivity (|Δ nDCG@5|)
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    ann5 = ann[(ann["k"] == 5) & (ann["metric"] == "ndcg_at_k")].set_index("combo_id").loc[COMBO_ORDER]
    xa = np.arange(len(COMBO_ORDER))
    shifts = ann5["abs_shift_final_minus_primary"].values
    ax.bar(xa, shifts, 0.56, color=["#0072B2" if v >= 0 else "#D55E00" for v in shifts])
    ax.axhline(0, color="#111827", lw=1.1)
    for i, v in enumerate(shifts):
        ax.text(i, v + (0.0008 if v >= 0 else -0.0018), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9.2, fontweight="bold")
    ax.set_xticks(xa)
    ax.set_xticklabels(COMBO_ORDER)
    ax.set_ylim(-0.015, 0.012)
    ax.set_ylabel("Δ nDCG@5 (Final Adjudicated − Primary AI)")
    ax.set_title("(e) Độ nhạy gán nhãn AI (|Δ| ≤ 0,011; không đổi xếp hạng)")
    ax.text(0.03, 0.94, "Lưu ý: Độ nhạy thấp giữa Primary và Final AI\nKHÔNG thay thế xác nhận con người (human_validated=false).",
            transform=ax.transAxes, va="top", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF2F2", edgecolor="#EF4444"))

    # (f) Language (VI/EN) & Difficulty (Easy/Med/Hard) Subgroups for A1, A4, B4
    ax = fig.add_subplot(gs[1, 2])
    style_axis_clean(ax)
    sg5 = subg[subg["k"] == 5]
    groups_spec = [
        ("language", "vi", "VI\n(n=24)"),
        ("language", "en", "EN\n(n=16)"),
        ("difficulty", "easy", "Dễ\n(Easy)"),
        ("difficulty", "medium", "TB\n(Medium)"),
        ("difficulty", "hard", "Khó\n(Hard)"),
    ]
    xs = np.arange(len(groups_spec))
    ws = 0.25
    for idx_c, cid in enumerate(["A1", "A4", "B4"]):
        vals = []
        for dim, val, _ in groups_spec:
            row = sg5[(sg5["combo_id"] == cid) & (sg5["subgroup_dim"] == dim) & (sg5["subgroup_value"] == val)]
            vals.append(float(row["ndcg_at_k"].iloc[0]) if len(row) else np.nan)
        st = COMBO_STYLE[cid]
        ax.bar(xs + (idx_c - 1) * ws, vals, ws, color=st["color"], hatch=st["hatch"],
               edgecolor="#FFFFFF" if st["hatch"] else st["color"], label=st["label"])
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i + (idx_c - 1) * ws, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[2] for g in groups_spec])
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("nDCG@5 theo phân nhóm")
    ax.set_title("(f) Hiệu năng theo ngôn ngữ (VI/EN) và độ khó")
    ax.legend(loc="upper right", fontsize=9.5)

    fig.text(
        0.5, 0.015,
        "Kết luận thống kê: A1, B4 và A4 tạo thành biên tối ưu Pareto; sự khác biệt không có ý nghĩa thống kê (p ≥ 0,05) KHÔNG được diễn giải là chứng minh tương đương.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig07_thong_ke_va_tradeoff")


def generate_fig08_do_tin_cay_va_phan_tich_loi(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 08: Độ tin cậy và Phân tích lỗi thực nghiệm (tổng hợp từ Stage 05 Hình 27–28)."""
    apply_consolidated_style()
    rob = bundle.robustness_metrics
    study = bundle.study_summary.set_index("feature")
    ft_res = bundle.functional_tests
    ftax = bundle.failure_taxonomy.sort_values("occurrence_count", ascending=True)
    err_rows = bundle.error_records

    fig = plt.figure(figsize=(23, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.25, top=0.88, bottom=0.08, left=0.065, right=0.97)

    fig.suptitle(
        "HÌNH 08 — ĐỘ TIN CẬY (CONTROLLED ROBUSTNESS), TÍNH NĂNG HỌC TẬP VÀ PHÂN TÍCH LỖI THỰC NGHIỆM",
        fontsize=22.5, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.92,
        "Giao thức Stage 04 Controlled (tách biệt hoàn toàn khỏi Live HTTP Testing ở Hình 09–10) | Bảo toàn nguyên cỡ mẫu và số đếm lỗi thực đo",
        ha="center", fontsize=13, color="#374151",
    )

    # (a) Controlled Robustness (n=20)
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    cat_map = {
        "out_of_domain": "Ngoài miền\n(OOD, n=4)",
        "unanswerable_in_corpus": "Không thể trả lời\n(n=4)",
        "prompt_injection": "Prompt Injection*\n(Context-lvl, n=3)",
        "distractor_context": "Ngữ cảnh nhiễu\n(n=3)",
        "semantic_paraphrase": "Diễn đạt lại\n(n=3)",
        "typo_noisy": "Tiếng Việt\nkhông dấu (n=3)",
    }
    rob_order = list(cat_map.keys())
    rsub = rob.set_index("category").loc[rob_order]
    xr = np.arange(len(rob_order))
    w = 0.36
    b1 = ax.bar(xr - w / 2, rsub["correct_handling_rate"] * 100, w, color="#0072B2", label="Xử lý đúng (%)")
    b2 = ax.bar(xr + w / 2, rsub["mean_answer_quality_norm"] * 100, w, color="#009E73", hatch="//", edgecolor="#FFFFFF", label="Chất lượng trả lời (%)")
    for i, (v1, v2) in enumerate(zip(rsub["correct_handling_rate"] * 100, rsub["mean_answer_quality_norm"] * 100)):
        ax.text(i - w / 2, v1 + 2.0, f"{v1:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0072B2")
        ax.text(i + w / 2, v2 + 2.0, f"{v2:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#009E73")
    ax.set_xticks(xr)
    ax.set_xticklabels([cat_map[c] for c in rob_order], fontsize=10)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Tỷ lệ / Điểm (%)")
    ax.set_title("(a) Controlled Robustness (n=20, A4 retrieval + Controlled Agent)")
    ax.legend(loc="upper right", fontsize=10)
    ax.annotate("Thất bại 0% (0/3)\nEvRecall@5 giảm −0,67\n(accent_insensitive=false)",
                xy=(5 - w / 2, 2), xytext=(3.82, 74), fontsize=9.2, fontweight="bold", color="#B91C1C",
                arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#FEF2F2", edgecolor="#EF4444"))

    # (b) Controlled Study Features Quality vs Real Backend Contract Checks
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    sm_cols = ["schema_valid_rate", "content_correctness_norm", "source_grounding_norm", "pedagogical_quality_norm"]
    sm_lbls = ["Hợp lệ Schema", "Độ đúng nội dung", "Grounding nguồn", "Chất lượng sư phạm"]
    xs = np.arange(len(sm_cols))
    qz_v = [float(study.loc["quiz_generation_controlled_agent", c]) * 100 for c in sm_cols]
    rm_v = [float(study.loc["study_roadmap_controlled_agent", c]) * 100 for c in sm_cols]
    ax.bar(xs - w / 2, qz_v, w, color="#0072B2", label=f"Controlled Quiz (n={int(study.loc['quiz_generation_controlled_agent', 'scenarios'])})")
    ax.bar(xs + w / 2, rm_v, w, color="#E69F00", hatch="//", edgecolor="#FFFFFF", label=f"Controlled Roadmap (n={int(study.loc['study_roadmap_controlled_agent', 'scenarios'])})")
    for i, (v1, v2) in enumerate(zip(qz_v, rm_v)):
        ax.text(i - w / 2, v1 + 2.0, f"{v1:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0072B2")
        ax.text(i + w / 2, v2 + 2.0, f"{v2:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#B45309")
    ax.set_xticks(xs)
    ax.set_xticklabels(sm_lbls, fontsize=10.5)
    ax.set_ylim(0, 134)
    ax.set_ylabel("Điểm / Tỷ lệ (%)")
    ax.set_title("(b) Controlled Study Features vs Hợp đồng Backend thật")
    ax.legend(loc="lower right", fontsize=10)
    passed_ft = int(ft_res.get("passed", 11))
    total_ft = int(ft_res.get("total_checks", 11))
    ax.text(
        0.02, 0.96,
        f"Kiểm thử hợp đồng module backend thật: {passed_ft}/{total_ft} PASSED\n"
        f"(Tách biệt với điểm Controlled Agent ở trên; Live HTTP xem Hình 10)",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5", edgecolor="#10B981"),
    )

    # (c) Failure Taxonomy (7 classes)
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    tax_vi = {
        "2_relevant_evidence_ranked_too_low": "Bằng chứng trong Top-60 nhưng xếp ngoài Top-5",
        "1_relevant_evidence_not_retrieved": "Không truy xuất được bằng chứng trong Top-60",
        "9_missing_knowledge_unit": "Thiếu đơn vị kiến thức bắt buộc (KU)",
        "10_irrelevant_context_overload": "Quá tải ngữ cảnh nhiễu không liên quan",
        "7_fragmented_evidence": "Bằng chứng bị phân mảnh qua nhiều chunk",
        "4_semantic_mismatch": "Lệch khớp ngữ nghĩa câu hỏi – tài liệu",
        "8_duplicate_context": "Trùng lặp nội dung trong ngữ cảnh",
    }
    yt = np.arange(len(ftax))
    ax.barh(yt, ftax["occurrence_count"], color="#D55E00", alpha=0.88, height=0.62)
    for i, cnt in enumerate(ftax["occurrence_count"]):
        ax.text(cnt + 0.5, i, f"{int(cnt)} lượt", va="center", ha="left", fontsize=10.2, fontweight="bold")
    ax.set_yticks(yt)
    ax.set_yticklabels([tax_vi.get(c, c) for c in ftax["failure_category"]], fontsize=10.5)
    ax.set_xlim(0, 35)
    ax.set_xlabel("Số lượt xuất hiện (trên 40 câu hỏi × {A4, B4})")
    ax.set_title("(c) Phân loại 7 dạng lỗi thực đo (Failure Taxonomy)")

    # (d) Representative Success & Failure Cases: Per-Query A4 vs B4 nDCG@5 (from retrieval_metrics_per_query.csv)
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    from evaluation.path_resolver import resolve_stage04_file

    pq_df = pd.read_csv(resolve_stage04_file(root, "retrieval/retrieval_metrics_per_query.csv"))
    pq5 = pq_df[pq_df["k"] == 5]
    q_a4 = pq5[pq5["combo_id"] == "A4"].set_index("query_id")["ndcg_at_k"].to_dict()
    q_b4 = pq5[pq5["combo_id"] == "B4"].set_index("query_id")["ndcg_at_k"].to_dict()
    qids = sorted(set(q_a4) & set(q_b4))
    xa4 = np.array([q_a4[q] for q in qids])
    yb4 = np.array([q_b4[q] for q in qids])
    ax.plot([0, 1], [0, 1], color="#9CA3AF", ls="--", lw=1.4, label="Đường cân bằng A4 = B4")
    ax.scatter(xa4, yb4, s=72, color="#0072B2", alpha=0.82, edgecolors="#111827", linewidths=0.8)
    highlight_q = {"Q007", "Q022", "Q029", "Q011", "Q014", "Q015"}
    for q, vx, vy in zip(qids, xa4, yb4):
        if q in highlight_q:
            clr = "#DC2626" if (vx < 0.45 and vy < 0.45) else "#059669"
            ax.scatter([vx], [vy], s=120, color=clr, edgecolors="#111827", linewidths=1.3, zorder=5)
            ax.text(vx + 0.018, vy + 0.016, q, fontsize=9.8, fontweight="bold", color=clr)
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("A4 (Recursive + Rerank) nDCG@5")
    ax.set_ylabel("B4 (Adaptive + Rerank) nDCG@5")
    ax.set_title("(d) Bản đồ chẩn đoán từng câu hỏi (A4 vs B4, n=40) & ca điển hình")
    ax.legend(loc="lower right", fontsize=10)
    ax.text(0.03, 0.94, "Đỏ: Cả hai yếu (Q007, Q022, Q029)\nXanh: B4 vượt A4 (Q011, Q014, Q015)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.8,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F9FAFB", edgecolor="#9CA3AF"))

    fig.text(
        0.5, 0.015,
        "*Lưu ý cỡ mẫu: Prompt Injection ở Stage 04 là context-level (n=3), không chứng minh an toàn bảo mật toàn hệ thống. "
        "Kết quả kiểm thử HTTP thực tế trên backend production xem Hình 09 và Hình 10.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig08_do_tin_cay_va_phan_tich_loi")


def generate_fig09_live_groq_va_http_e2e(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 09: Đánh giá Live Groq và HTTP End-to-End (tổng hợp từ Supplementary Hình 01–04)."""
    apply_consolidated_style()
    aq = pd.read_csv(resolve_ac04_file(root, "answer_quality_summary.csv"), keep_default_na=False)
    hl = pd.read_csv(resolve_ac04_file(root, "http_latency_summary.csv")).set_index("experiment_group")
    s4_ans = bundle.answer_summary.set_index("combo_id")
    s4_prod = bundle.production_profile_latency.set_index("combo_id")
    s4_lat = bundle.latency_summary

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.24, top=0.87, bottom=0.08, left=0.06, right=0.97)

    fig.suptitle(
        "HÌNH 09 — ĐÁNH GIÁ LIVE GROQ API (openai/gpt-oss-120b) VÀ HTTP END-TO-END THỰC TẾ",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.918,
        "Môi trường cô lập (127.0.0.1:8015, Qdrant copy xác thực SHA-256, 0 sửa đổi backend) | Backend KHÔNG hỗ trợ Context Budget 2.048 tokens hay Top-K > 5",
        ha="center", fontsize=13.2, color="#991B1B", fontweight="semibold",
    )

    def _get_aq(proto: str, subset: str) -> pd.Series:
        return aq[(aq["protocol"] == proto) & (aq["subset"] == subset)].iloc[0]

    # (a) Live Groq Answer Quality (All-40 vs Live-LLM-Only vs Direct NO_RAG vs Stage 04 Controlled A4)
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    m_cols = [
        ("answer_correctness_norm", "Correctness"),
        ("answer_relevancy_norm", "Relevancy"),
        ("faithfulness_norm", "Faithfulness"),
        ("answer_completeness_norm", "Completeness"),
        ("citation_accuracy_norm", "Citation Acc."),
    ]
    series_spec = [
        ("HTTP Config A (All 40)", _get_aq("http_e2e_config_a_prod_default_pool10", "all_40_queries"), "#0072B2", ""),
        ("HTTP Config A (Live LLM n=4)", _get_aq("http_e2e_config_a_prod_default_pool10", "live_llm_only"), "#56B4E9", "//"),
        ("HTTP Config B (All 40)", _get_aq("http_e2e_config_b_exploratory_pool30", "all_40_queries"), "#D55E00", ""),
        ("Direct Live Groq NO_RAG (n=40)", _get_aq("direct_live_groq_no_rag", "all_40_queries"), "#595959", ".."),
        ("Đối chiếu Stage 04 Agent A4 (tách biệt)", s4_ans.loc["A4"], "#009E73", "xx"),
    ]
    xm = np.arange(len(m_cols))
    wm = 0.165
    for idx_s, (lbl, row, clr, h) in enumerate(series_spec):
        vals, nas = [], []
        for col, _ in m_cols:
            raw = row[col]
            if pd.isna(raw) or str(raw).strip().upper() == "N/A":
                vals.append(0.0)
                nas.append(True)
            else:
                vals.append(float(raw))
                nas.append(False)
        bars = ax.bar(xm + (idx_s - 2) * wm, vals, wm, color=clr, hatch=h, edgecolor="#FFFFFF" if h else clr, label=lbl)
        for b, v, na in zip(bars, vals, nas):
            if na:
                ax.text(b.get_x() + b.get_width() / 2, 0.03, "N/A", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#374151", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", rotation=90)
    ax.set_xticks(xm)
    ax.set_xticklabels([m[1] for m in m_cols], fontsize=11)
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("Điểm chuẩn hóa (0 – 1)")
    ax.set_title("(a) Chất lượng Live Groq (All-40 vs Live-LLM n=4 vs NO_RAG vs Controlled A4)")
    ax.legend(loc="upper center", ncol=2, fontsize=9.0)

    # (b) Generation Mode Distribution (exposing 82.5% false-positive abstention)
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    gm_rows = [
        ("HTTP Config A\n(pool=10, n=40)", _get_aq("http_e2e_config_a_prod_default_pool10", "all_40_queries")),
        ("HTTP Config B\n(pool=30, n=40)", _get_aq("http_e2e_config_b_exploratory_pool30", "all_40_queries")),
        ("Direct Live Groq\nNO_RAG (n=40)", _get_aq("direct_live_groq_no_rag", "all_40_queries")),
    ]
    xg = np.arange(len(gm_rows))
    llm_pct = [int(r["n_live_llm"]) / int(r["n_total"]) * 100 for _, r in gm_rows]
    ext_pct = [int(r["n_extractive"]) / int(r["n_total"]) * 100 for _, r in gm_rows]
    abs_pct = [int(r["n_abstained"]) / int(r["n_total"]) * 100 for _, r in gm_rows]
    ax.bar(xg, llm_pct, 0.52, color="#009E73", label="Sinh Live LLM (mode=llm)")
    ax.bar(xg, ext_pct, 0.52, bottom=llm_pct, color="#E69F00", label="Extractive Fallback")
    ax.bar(xg, abs_pct, 0.52, bottom=np.array(llm_pct) + np.array(ext_pct), color="#DC2626", hatch="//", edgecolor="#FFFFFF", label="Từ chối sai — Abstained")
    for i, (l_v, e_v, a_v) in enumerate(zip(llm_pct, ext_pct, abs_pct)):
        if l_v > 0:
            ax.text(i, l_v / 2, f"LLM\n{l_v:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="#FFFFFF")
        if e_v > 0:
            ax.text(i, l_v + e_v / 2, f"Ext {e_v:.1f}%", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#111827")
        if a_v > 0:
            ax.text(i, l_v + e_v + a_v / 2, f"Abstained\n{a_v:.1f}% (33/40)", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#FFFFFF",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#991B1B", edgecolor="none", alpha=0.88))
    ax.set_xticks(xg)
    ax.set_xticklabels([g[0] for g in gm_rows], fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Tỷ lệ phản hồi (%)")
    ax.set_title("(b) Phân bổ chế độ sinh: 100% HTTP 200 nhưng 82,5% từ chối sai")
    ax.legend(loc="upper center", ncol=3, fontsize=9.6)

    # (c) Client-Observed HTTP End-to-End Latency (P50, Mean, P95)
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    hl_order = ["HTTP_Config_A_pool10", "HTTP_Config_B_pool30", "Direct_Live_Groq_NO_RAG", "HTTP_Robustness_20", "HTTP_Quiz_8_Topics", "HTTP_Roadmap_4_Scenarios"]
    hl_lbls = ["Chat Config A\n(pool=10, n=40)", "Chat Config B\n(pool=30, n=40)", "Direct Groq\nNO_RAG (n=40)", "Robustness\n(/chat, n=20)", "Live Quiz\n(/quiz, n=8)", "Live Roadmap\n(/roadmap, n=4)"]
    xh = np.arange(len(hl_order))
    wh = 0.26
    sub_hl = hl.loc[hl_order]
    ax.bar(xh - wh, sub_hl["p50_ms"] / 1000.0, wh, color="#0072B2", label="P50 (s)")
    ax.bar(xh, sub_hl["mean_ms"] / 1000.0, wh, color="#009E73", label="Mean (s)")
    ax.bar(xh + wh, sub_hl["p95_ms"] / 1000.0, wh, color="#D55E00", label="P95 (s)")
    for i, row in enumerate(sub_hl.itertuples()):
        ax.text(i + wh, row.p95_ms / 1000.0 + 0.35, f"{row.p95_ms/1000.0:.2f}s", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#9A3412")
    ax.set_xticks(xh)
    ax.set_xticklabels(hl_lbls, fontsize=9.8)
    ax.set_ylim(0, 22.5)
    ax.set_ylabel("Độ trễ HTTP End-to-End phía Client (giây — s)")
    ax.set_title("(c) Độ trễ HTTP End-to-End thực đo phía Client (P50 / Mean / P95)")
    ax.legend(loc="upper left", ncol=3, fontsize=10)

    # (d) E2E Latency Decomposition vs Stage 04 Retrieval-Only Reference
    ax = fig.add_subplot(gs[1, 1])
    style_axis_clean(ax)
    dec_groups = ["HTTP_Config_A_pool10", "HTTP_Config_B_pool30", "Direct_Live_Groq_NO_RAG", "HTTP_Quiz_8_Topics", "HTTP_Roadmap_4_Scenarios"]
    dec_lbls = ["HTTP Chat A\n(pool=10)", "HTTP Chat B\n(pool=30)", "Direct Groq\nNO_RAG", "HTTP Quiz\n(n=8)", "HTTP Roadmap\n(n=4)", "S4 Prod-Param\nA4 (Retr-Only)", "S4 Warm A4\n(Retr-Only)"]
    ret_vals = [hl.loc[g, "retrieval_mean_ms"] for g in dec_groups] + [float(s4_prod.loc["A4", "mean_ms"]), float(s4_lat[(s4_lat["combo_id"] == "A4") & (s4_lat["k"] == 5)]["mean_ms"].iloc[0])]
    llm_vals = [hl.loc[g, "llm_generation_mean_ms"] for g in dec_groups] + [0.0, 0.0]
    ovh_vals = [hl.loc[g, "http_overhead_mean_ms"] for g in dec_groups] + [0.0, 0.0]
    xd = np.arange(len(dec_lbls))
    ax.bar(xd, ret_vals, 0.56, color="#0072B2", label="Retrieval Mean (ms)")
    ax.bar(xd, llm_vals, 0.56, bottom=ret_vals, color="#D55E00", label="Live Groq Generation Mean (ms)")
    ax.bar(xd, ovh_vals, 0.56, bottom=np.array(ret_vals) + np.array(llm_vals), color="#9CA3AF", label="HTTP / App Overhead (ms)")
    ax.axvline(4.5, color="#111827", ls="--", lw=1.6)
    for i, tot in enumerate(np.array(ret_vals) + np.array(llm_vals) + np.array(ovh_vals)):
        ax.text(i, tot + 200, f"{tot/1000.0:.2f}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xd)
    ax.set_xticklabels(dec_lbls, fontsize=9.4)
    ax.set_ylim(0, 11800)
    ax.set_ylabel("Độ trễ trung bình (ms)")
    ax.set_title("(d) Phân rã Retrieval vs Live Groq Generation vs Overhead (tách Stage 04)")
    ax.legend(loc="upper left", fontsize=9.6)
    ax.text(5.5, 9800, "Đối chiếu Stage 04\n(Chỉ truy xuất —\nkhông có LLM/HTTP)", ha="center", va="top", fontsize=9.4, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F3F4F6", edgecolor="#6B7280"))

    fig.text(
        0.5, 0.015,
        "Giải thích: Trong /api/chat, trung bình LLM toàn tập 40 câu chỉ là 281–328 ms vì 33/40 câu bị từ chối trước khi gọi Groq; trên nhóm thực sự gọi Groq (n=7), thời gian gọi LLM trung bình là ~1,6–1,9s.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig09_live_groq_va_http_e2e")


def generate_fig10_chuc_nang_va_do_on_dinh_api(bundle: Stage04Bundle, root: Path) -> dict[str, Any]:
    """Hình 10: Chức năng thực tế và Độ ổn định API (tổng hợp từ Supplementary Hình 05–06)."""
    apply_consolidated_style()
    st_live = pd.read_csv(resolve_ac04_file(root, "study_features_summary_live.csv")).set_index("feature")
    st_s4 = bundle.study_summary.set_index("feature")
    rb_live = pd.read_csv(resolve_ac04_file(root, "robustness_summary_live.csv")).set_index("category")
    hl = pd.read_csv(resolve_ac04_file(root, "http_latency_summary.csv")).set_index("experiment_group")
    tok = pd.read_csv(resolve_ac04_file(root, "token_usage_summary.csv")).set_index("experiment_group")

    fig = plt.figure(figsize=(24, 14.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.25, top=0.87, bottom=0.08, left=0.06, right=0.97)

    fig.suptitle(
        "HÌNH 10 — CHỨC NĂNG THỰC TẾ (QUIZ, ROADMAP, ROBUSTNESS) VÀ ĐỘ ỔN ĐỊNH API SẢN PHẨM",
        fontsize=23, fontweight="bold", y=0.96, color="#111827",
    )
    fig.text(
        0.5, 0.918,
        "Tổng 166 yêu cầu HTTP/API (157 thuộc 8 nhóm chính + 9 Edge Cases) ≠ 90 lệnh gọi Groq thực tế (84 HTTP 200, 6 HTTP 429 tầng provider; 117.091 tokens) | Kết luận: NOT PRODUCTION-READY",
        ha="center", fontsize=13.0, color="#991B1B", fontweight="semibold",
    )

    # (a) Live Quiz & Roadmap vs Controlled Agent (separating HTTP 200, Partial Live, Error-Free Full)
    ax = fig.add_subplot(gs[0, 0])
    style_axis_clean(ax)
    cats_a = ["HTTP 200\nTransport", "Có sinh Live\n(≥1 mục)", "Sinh đủ\nsạch lỗi", "Correctness\n(All topics)", "Grounding\n(All topics)"]
    qz_live = [
        st_live.loc["live_http_quiz", "http_200_count"] / 8 * 100,
        st_live.loc["live_http_quiz", "any_live_llm_output_count"] / 8 * 100,
        st_live.loc["live_http_quiz", "functional_live_llm_success_count"] / 8 * 100,
        st_live.loc["live_http_quiz", "content_correctness_norm"] * 100,
        st_live.loc["live_http_quiz", "source_grounding_norm"] * 100,
    ]
    rm_live = [
        st_live.loc["live_http_roadmap", "http_200_count"] / 4 * 100,
        st_live.loc["live_http_roadmap", "any_live_llm_output_count"] / 4 * 100,
        st_live.loc["live_http_roadmap", "functional_live_llm_success_count"] / 4 * 100,
        st_live.loc["live_http_roadmap", "content_correctness_norm"] * 100,
        st_live.loc["live_http_roadmap", "source_grounding_norm"] * 100,
    ]
    qz_ctrl = [np.nan, np.nan, np.nan, st_s4.loc["quiz_generation_controlled_agent", "content_correctness_norm"] * 100, st_s4.loc["quiz_generation_controlled_agent", "source_grounding_norm"] * 100]
    xa = np.arange(len(cats_a))
    wa = 0.26
    ax.bar(xa - wa, qz_live, wa, color="#0072B2", label="Live HTTP /api/quiz (n=8 chuẩn hóa)")
    ax.bar(xa, rm_live, wa, color="#D55E00", hatch="//", edgecolor="#FFFFFF", label="Live HTTP /api/roadmap (n=4)")
    ax.bar(xa + wa, [0 if np.isnan(v) else v for v in qz_ctrl], wa, color="#009E73", alpha=0.8, label="Đối chiếu Stage 04 Controlled Quiz (n=8)")
    for i, (vq, vr) in enumerate(zip(qz_live, rm_live)):
        ax.text(i - wa, vq + 2.0, f"{vq:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0072B2")
        ax.text(i, vr + 2.0, f"{vr:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#D55E00")
    for i in (3, 4):
        ax.text(i + wa, qz_ctrl[i] + 2.0, f"{qz_ctrl[i]:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#009E73")
    ax.set_xticks(xa)
    ax.set_xticklabels(cats_a, fontsize=10.2)
    ax.set_ylim(0, 140)
    ax.set_ylabel("Tỷ lệ / Điểm (%)")
    ax.set_title("(a) Live Quiz (5/8 có câu, 1/8 sạch lỗi) & Roadmap (0/4 sinh mục)")
    ax.legend(loc="upper right", fontsize=9.0)
    ax.text(0.02, 0.97, "Quiz: 8/8 HTTP 200 nhưng 3/8 gặp lỗi 429 tầng provider Groq\n(8.093–9.142 tok > 8.000 TPM); raw QZ_03 trả HTTP 422.\nRoadmap: 4/4 HTTP 200 abstained (0/4 mục).",
            transform=ax.transAxes, va="top", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#F59E0B"))

    # (b) Live HTTP Robustness (n=20: 8/8 OOD/Unanswerable vs 0/12 Answerable)
    ax = fig.add_subplot(gs[0, 1])
    style_axis_clean(ax)
    rcats = ["out_of_domain", "unanswerable_in_corpus", "prompt_injection", "distractor_context", "semantic_paraphrase", "typo_noisy"]
    rlbls = ["OOD\n(n=4)", "Không thể\ntrả lời (n=4)", "Prompt Inj.\nQuery-lvl (n=3)", "Nhiễu\n(n=3)", "Paraphrase\n(n=3)", "Không dấu\nVI (n=3)"]
    rsub = rb_live.loc[rcats]
    xr = np.arange(len(rcats))
    wr = 0.36
    ax.bar(xr - wr / 2, rsub["correct_handling_rate"] * 100, wr, color="#009E73", label="Xử lý đúng (%)")
    ax.bar(xr + wr / 2, rsub["abstention_rate"] * 100, wr, color="#DC2626", hatch="//", edgecolor="#FFFFFF", label="Tỷ lệ Abstained (%)")
    for i, (vc, va) in enumerate(zip(rsub["correct_handling_rate"] * 100, rsub["abstention_rate"] * 100)):
        ax.text(i - wr / 2, vc + 2.0, f"{vc:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#065F46")
        ax.text(i + wr / 2, va + 2.0, f"{va:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#991B1B")
    ax.set_xticks(xr)
    ax.set_xticklabels(rlbls, fontsize=10.2)
    ax.set_ylim(0, 138)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("(b) Live HTTP Robustness (n=20): 8/8 OOD/Unans đúng, 0/12 câu hợp lệ")
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)
    ax.text(0.98, 0.97, "20/20 ca trả HTTP 200 + abstained:\n• 8/8 ca cần từ chối → đúng\n• 0/12 ca DSA hợp lệ → thất bại",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#EF4444"))

    # (c) Reliability Matrix: HTTP 200 vs Error-Free Live LLM vs Provider 429 / Errors / Edge Cases
    ax = fig.add_subplot(gs[1, 0])
    style_axis_clean(ax)
    rel_groups = ["HTTP_Config_A_pool10", "HTTP_Config_B_pool30", "Direct_Live_Groq_NO_RAG", "HTTP_Robustness_20", "HTTP_Quiz_8_Topics", "HTTP_Roadmap_4_Scenarios"]
    rel_lbls = ["Chat A\n(40r / 7c)", "Chat B\n(40r / 7c)", "NO_RAG\n(40r / 40c)", "Robust.\n(20r / 1c)", "Quiz\n(8r / 33c)", "Roadmap\n(4r / 0c)"]
    sub_r = hl.loc[rel_groups]
    xrel = np.arange(len(rel_groups))
    wrel = 0.25
    http200_pct = sub_r["http_success_rate"] * 100
    live_ok_pct = sub_r["live_llm_count"] / sub_r["measured_requests"] * 100
    err_pct = sub_r["error_rate"] * 100
    ax.bar(xrel - wrel, http200_pct, wrel, color="#0072B2", label="Ứng dụng HTTP 200 (%)")
    ax.bar(xrel, live_ok_pct, wrel, color="#009E73", label="Sinh Live LLM sạch lỗi (%)")
    ax.bar(xrel + wrel, err_pct, wrel, color="#DC2626", hatch="//", edgecolor="#FFFFFF", label="Tỷ lệ lỗi / Provider 429 (%)")
    for i, (v1, v2, v3) in enumerate(zip(http200_pct, live_ok_pct, err_pct)):
        ax.text(i - wrel, v1 + 1.8, f"{v1:.0f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0072B2")
        ax.text(i, v2 + 1.8, f"{v2:.0f}%" if v2 == 100 else f"{v2:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#065F46")
        if v3 > 0:
            ax.text(i + wrel, v3 + 1.8, f"{v3:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#991B1B")
    ax.set_xticks(xrel)
    ax.set_xticklabels(rel_lbls, fontsize=9.8)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("(c) Ma trận độ tin cậy: HTTP 200 ≠ Sinh LLM thành công (r=req, c=Groq calls)")
    ax.legend(loc="upper center", ncol=3, fontsize=9.2)

    # (d) Compact Summary Table of 4 Production Defects, Edge Cases (9/9) & Token Accounting
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("(d) Tổng hợp 4 lỗi Backend thật, Edge Cases (9/9) & Tiêu thụ Token Groq", pad=10)
    defect_rows = [
        ["Lỗi #1: /api/chat", "82,5% (33/40) từ chối dương tính giả (reason=off_topic)\ntrong intent_router.py / chat_retrieval.py dù score > 0,90."],
        ["Lỗi #2: llm_client.py", "Ném LLMError('truncated_json') khi plain-text chat\nchạm max_tokens=1.024 (1 ca trong Config B → fallback)."],
        ["Lỗi #3: /api/quiz", "Ứng dụng trả HTTP 200 nhưng 6 lệnh gọi tầng provider Groq\ngặp 429 (3/8 chủ đề) do >8.000 TPM; raw QZ_03 trả 422."],
        ["Lỗi #4: /api/roadmap", "0/4 kịch bản sinh lộ trình (4/4 HTTP 200 abstained)\ndo bộ lọc catalog không khớp câu mục tiêu dài."],
        ["Edge Cases & Timeout", "9/9 PASSED (/health, /ready, 3× empty, 400, 422,\nclient timeout, server 504 timeout; 0 lệnh gọi Groq)."],
        ["Kế toán API & Token", "166 yêu cầu HTTP/API (157 nhóm chính + 9 edge cases)\n→ 90 lệnh gọi Groq (84×200, 6×429); 117.091 tokens."],
    ]
    tbl = ax.table(
        cellText=defect_rows,
        colLabels=["Hạng mục kiểm định", "Bằng chứng thực đo (KHÔNG sửa mã production)"],
        loc="center",
        cellLoc="left",
        colWidths=[0.28, 0.72],
        bbox=[0.0, 0.04, 1.0, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif r in (1, 2, 3, 4):
            cell.set_facecolor("#FEF2F2" if c == 0 else "#FFFFFF")
            if c == 0:
                cell.set_text_props(color="#991B1B", fontweight="bold")
        else:
            cell.set_facecolor("#ECFDF5" if c == 0 else "#F9FAFB")
            if c == 0:
                cell.set_text_props(color="#065F46", fontweight="bold")
        cell.set_edgecolor("#D1D5DB")

    fig.text(
        0.5, 0.015,
        "Tách biệt quản trị bắt buộc: Cổng hoàn thành thực nghiệm Stage 04 AC-04 đạt PASS sau bổ sung (bảo toàn lịch sử gốc PARTIAL), "
        "nhưng sản phẩm ở trạng thái NOT PRODUCTION-READY và human_validated = false.",
        ha="center", fontsize=12, style="italic", color="#1F2937",
    )
    return save_consolidated_figure(fig, root, "fig10_chuc_nang_va_do_on_dinh_api")
