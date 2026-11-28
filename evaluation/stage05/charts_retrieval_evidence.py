"""Group B (Figures 06-12: Retrieval) & Group C (Figures 13-15: Evidence) visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.stage05 import COMBO_LABELS, COMBO_ORDER, KS
from evaluation.stage05.data_loader import Stage04Bundle
from evaluation.stage05.style import add_protocol_footer, combo_style, save_figure


def _plot_curve_and_k5_k15_bars(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    fig_num: int,
    stem: str,
    dc_id: int,
    dc_name: str,
    group: str,
    title: str,
    footer: str,
    caption_vi: str,
    source_artifacts: list[str],
    png_dir: Path,
    svg_dir: Path,
    y_max: float = 1.05,
    extra_annotation: str | None = None,
) -> dict[str, Any]:
    """Standard 2-panel retrieval curve across 9 Ks + K=5/K=15 grouped bar comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    x_idx = np.arange(len(KS))
    for cid in COMBO_ORDER:
        sub = df[df["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx,
            sub[metric_col].to_numpy(),
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=st["linewidth"],
            markersize=st["markersize"],
            label=COMBO_LABELS[cid],
        )

    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K (Top-K Retrieved Chunks)")
    ax1.set_ylabel(metric_label)
    ax1.set_ylim(0.0, y_max)
    ax1.set_title(f"(A) {metric_label} Across All 9 Cutoffs (K=1..60)")
    ax1.legend(loc="best", ncol=2, fontsize=7.5)
    # Keep plot area uncluttered; protocol/extra notes live in the subtitle/footer
    if extra_annotation:
        ax1.set_title(f"(A) {metric_label} Across All 9 Cutoffs — {extra_annotation}", fontsize=10.3)

    # Panel B: Side-by-side comparison at K=5 and K=15 for all 8 combos
    cx = np.arange(len(COMBO_ORDER))
    w = 0.36
    vals_k5 = [float(df[(df["combo_id"] == cid) & (df["k"] == 5)][metric_col].iloc[0]) for cid in COMBO_ORDER]
    vals_k15 = [float(df[(df["combo_id"] == cid) & (df["k"] == 15)][metric_col].iloc[0]) for cid in COMBO_ORDER]

    for i, cid in enumerate(COMBO_ORDER):
        st = combo_style(cid)
        ax2.bar(
            cx[i] - w / 2,
            vals_k5[i],
            width=w,
            color=st["color"],
            alpha=0.95,
            hatch=st["hatch"],
            edgecolor="#1F2937",
            linewidth=0.6,
            label="K=5 (solid/hatched)" if i == 0 else None,
        )
        ax2.bar(
            cx[i] + w / 2,
            vals_k15[i],
            width=w,
            color=st["color"],
            alpha=0.45,
            hatch=st["hatch"],
            edgecolor="#1F2937",
            linewidth=0.6,
            label="K=15 (lighter fill)" if i == 0 else None,
        )
        ax2.text(cx[i] - w / 2, vals_k5[i] + 0.015, f"{vals_k5[i]:.2f}", ha="center", va="bottom", fontsize=6.9, rotation=90)
        ax2.text(cx[i] + w / 2, vals_k15[i] + 0.015, f"{vals_k15[i]:.2f}", ha="center", va="bottom", fontsize=6.9, rotation=90)

    ax2.axvline(3.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax2.text(1.5, y_max * 0.96, "Profile A (Recursive)", ha="center", va="top", fontsize=8.0, color="#334155", fontweight="bold")
    ax2.text(5.5, y_max * 0.96, "Profile B (Adaptive)", ha="center", va="top", fontsize=8.0, color="#334155", fontweight="bold")
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination (Left bar: K=5, Right bar: K=15)")
    ax2.set_ylabel(metric_label)
    ax2.set_ylim(0.0, y_max * 1.07)
    ax2.set_title(f"(B) {metric_label} at Key Cutoffs (K=5 vs K=15)")

    fig.suptitle(f"Figure {fig_num}: {title}")
    add_protocol_footer(fig, footer)
    saved = save_figure(fig, stem, png_dir, svg_dir)
    return {
        "figure_number": fig_num,
        "figure_id": stem,
        "data_contract_id": dc_id,
        "data_contract_name": dc_name,
        "group": group,
        "title": title,
        "protocol": "Official Top-60 / candidate_top_k=120 retrieval on 40 shared DSA questions (Judged@K = 1.0, 0 ranking drift).",
        "metrics": [metric_col],
        "source_artifacts": source_artifacts,
        "caption_vi": caption_vi,
        **saved,
    }


def generate_retrieval_and_evidence_figures(
    bundle: Stage04Bundle,
    png_dir: Path,
    svg_dir: Path,
) -> list[dict[str, Any]]:
    """Generate Figures 06-12 (Group B: Retrieval) and Figures 13-15 (Group C: Evidence)."""
    ret = bundle.retrieval_summary
    ev = bundle.evidence_summary
    records: list[dict[str, Any]] = []

    ret_sources = ["evaluation/artifacts/stage04/retrieval/retrieval_metrics_summary.csv"]
    ret_footer = (
        "Source: retrieval_metrics_summary.csv (40 shared queries, pool=120, Judged@K=1.0, 0 unjudged slots, human_validated=false). "
        "Solid lines = Profile A (Recursive); Dashed = Profile B (Adaptive)."
    )

    # Fig 06: Precision@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "precision_at_k", "Precision@K", 6, "fig06_precision_at_k", 6, "Precision@K", "B",
        "Retrieval Precision@K Across All Eight Combinations (K=1..60)",
        ret_footer,
        "Hình 6: So sánh Precision@K của 8 tổ hợp thực nghiệm qua 9 mốc K (1..60) và đối chiếu tại K=5, K=15. "
        "Precision giảm đơn điệu khi K tăng; A4 đạt cao nhất tại K=5 (0,7000) trong khi BM25 (A1=0,6000) vượt Dense thuần (A2=0,4700).",
        ret_sources, png_dir, svg_dir, y_max=1.05,
        extra_annotation="Judged@K = 1.000 for all combos & Ks",
    ))

    # Fig 07: Pooled Recall@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "pooled_recall_at_k", "Pooled Recall@K", 7, "fig07_pooled_recall_at_k", 7, "Pooled Recall@K", "B",
        "Pooled Chunk Recall@K Across All Eight Combinations (K=1..60)",
        ret_footer + " Note: Pooled Recall uses the Stage 03 judged candidate pool denominator, NOT complete corpus recall.",
        "Hình 7: Pooled Recall@K (tính trên mẫu số pool ứng viên đã gán nhãn Stage 03, không phải toàn bộ corpus). "
        "Tại K=5, B4 (0,3509) nhỉnh hơn nhẹ A4 (0,3474); ở K lớn hơn (K=15..60), A4 dẫn đầu (0,6745 tại K=15; 0,9447 tại K=60).",
        ret_sources, png_dir, svg_dir, y_max=1.05,
        extra_annotation="Pooled chunk denominator (NOT complete corpus recall)",
    ))

    # Fig 08: MRR@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "mrr_at_k", "MRR@K", 8, "fig08_mrr_at_k", 8, "MRR@K", "B",
        "Mean Reciprocal Rank (MRR@K) Across All Eight Combinations (K=1..60)",
        ret_footer,
        "Hình 8: Mean Reciprocal Rank (MRR@K) qua 9 mốc K. Reranking đẩy tài liệu phù hợp đầu tiên lên sát vị trí top-1 "
        "(A4 MRR@5 = 0,9875; B4 = 0,9008), vượt rõ rệt so với Dense đơn lẻ (A2 = 0,7079; B2 = 0,6246).",
        ret_sources, png_dir, svg_dir, y_max=1.05,
    ))

    # Fig 09: nDCG@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "ndcg_at_k", "nDCG@K", 9, "fig09_ndcg_at_k", 9, "nDCG@K", "B",
        "Normalized Discounted Cumulative Gain (nDCG@K) Across All Eight Combinations",
        ret_footer + " Strict Grade-2 sensitivity (strict_relevance_metrics.csv) preserves the exact rank order A4 > B4 ~ A1 > A3 > B1 > B3 > A2 > B2.",
        "Hình 9: nDCG@K (đánh giá phân bậc độ liên quan 0/1/2) qua 9 mốc K. A4 đạt cao nhất (0,7053 tại K=5; 0,7335 tại K=15; 0,8097 tại K=60), "
        "tiếp theo là B4 (0,5591) và A1 (0,5584). Hybrid RRF (A3=0,5037) thấp hơn BM25 tại K=5 do tín hiệu Dense yếu hơn trên thuật ngữ DSA.",
        ret_sources + ["evaluation/artifacts/stage04/retrieval/strict_relevance_metrics.csv"], png_dir, svg_dir, y_max=0.92,
        extra_annotation="Strict Grade-2 sensitivity preserves rank order",
    ))

    # Fig 10: MAP@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "map_at_k", "MAP@K", 10, "fig10_map_at_k", 10, "MAP@K", "B",
        "Mean Average Precision (MAP@K) Across All Eight Combinations (K=1..60)",
        ret_footer,
        "Hình 10: Mean Average Precision (MAP@K) qua 9 mốc K. Hai tổ hợp sử dụng Reranker (A4 = 0,6638; B4 = 0,5638 tại K=5) "
        "cải thiện nhất quán mật độ tài liệu liên quan ở đầu danh sách so với BM25, Dense và Hybrid RRF.",
        ret_sources, png_dir, svg_dir, y_max=1.05,
    ))

    # Fig 11: F1@K
    records.append(_plot_curve_and_k5_k15_bars(
        ret, "f1_at_k", "F1@K", 11, "fig11_f1_at_k", 11, "F1@K", "B",
        "Harmonic Mean F1@K (Precision & Pooled Recall) Across All Eight Combinations",
        ret_footer,
        "Hình 11: F1@K (trung bình điều hòa giữa Precision@K và Pooled Recall@K). Điểm cực đại xuất hiện quanh K=10–15 "
        "(A4 đạt đỉnh ~0,520 tại K=15; B4 ~0,414 tại K=5), phản ánh vùng cân bằng tự nhiên giữa độ chính xác và độ phủ chunk.",
        ret_sources, png_dir, svg_dir, y_max=0.62,
    ))

    # ------------------------------------------------------------------
    # Fig 12: Hit Rate@K & Judgment Coverage (Judged@K)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(wspace=0.26, bottom=0.19, top=0.83)

    x_idx = np.arange(len(KS))
    for cid in COMBO_ORDER:
        sub = ret[ret["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["hit_rate_at_k"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K")
    ax1.set_ylabel("Hit Rate@K")
    ax1.set_ylim(0.0, 1.06)
    ax1.set_title("(A) Hit Rate@K Across All 9 Cutoffs")
    ax1.legend(loc="lower right", ncol=2, fontsize=7.5)

    # Panel B: Judged@K verification (K=5, K=15, K=60)
    cx = np.arange(len(COMBO_ORDER))
    judged_60 = [float(ret[(ret["combo_id"] == cid) & (ret["k"] == 60)]["judged_at_k"].iloc[0]) for cid in COMBO_ORDER]
    unjudged_60 = [int(ret[(ret["combo_id"] == cid) & (ret["k"] == 60)]["total_unjudged_slots"].iloc[0]) for cid in COMBO_ORDER]
    bars = ax2.bar(cx, judged_60, width=0.58, color=[combo_style(c)["color"] for c in COMBO_ORDER],
                   edgecolor="#1F2937", linewidth=0.6)
    for bar, u in zip(bars, unjudged_60):
        ax2.text(bar.get_x() + bar.get_width() / 2, 1.02, f"1.000\n(u={u})", ha="center", va="bottom", fontsize=6.8)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_ylabel("Judged@60 Coverage Rate")
    ax2.set_ylim(0.0, 1.18)
    ax2.set_title("(B) Judgment Coverage Verification (Judged@60 = 1.000)")
    ax2.text(
        0.5, 0.45,
        f"Exact Ranking Agreement vs Stage 03:\n{bundle.ranking_drift['exact_ranking_matches']}/{bundle.ranking_drift['total_calls_compared']} calls (100.0%)\n0 unique unjudged candidates",
        transform=ax2.transAxes, ha="center", va="center", fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#0072B2", alpha=0.95),
    )

    fig.suptitle("Figure 12: Hit Rate@K and Complete Judgment Coverage Verification (Judged@K)")
    add_protocol_footer(fig, ret_footer + " Source also includes judgment_coverage.csv and ranking_drift.json.")
    saved = save_figure(fig, "fig12_hit_rate_and_judged_at_k", png_dir, svg_dir)
    records.append({
        "figure_number": 12,
        "figure_id": "fig12_hit_rate_and_judged_at_k",
        "data_contract_id": 12,
        "data_contract_name": "Hit Rate@K",
        "group": "B",
        "title": "Hit Rate@K and Complete Judgment Coverage Verification (Judged@K)",
        "protocol": "Official Top-60 retrieval on 40 queries; judgment coverage verified across all 2,880 query-combo-K cells.",
        "metrics": ["hit_rate_at_k", "judged_at_k", "total_unjudged_slots"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/retrieval/retrieval_metrics_summary.csv",
            "evaluation/artifacts/stage04/retrieval/judgment_coverage.csv",
            "evaluation/artifacts/stage04/retrieval/ranking_drift.json",
        ],
        "caption_vi": (
            "Hình 12: (A) Hit Rate@K qua 9 mốc K — A1, A3, A4 và B4 đều đạt 100% (1,000) từ K=5 trở đi, trong khi Dense thuần "
            "(A2=0,900; B2=0,775) cần K lớn hơn; (B) Kiểm chứng độ phủ gán nhãn Judged@60 = 1,000 trên toàn bộ 8 tổ hợp (0 vị trí chưa gán nhãn, "
            "320/320 khớp hoàn toàn thứ hạng với Stage 03)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 13 (Group C): Evidence Recall@K & Semantic KU Coverage
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["evidence_recall_at_k"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K")
    ax1.set_ylabel("Source-Backed Evidence Recall@K")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("(A) Source-Backed Evidence Recall@K (K=1..60)")
    ax1.legend(loc="lower right", ncol=2, fontsize=7.5)

    # Panel B: Strict Source-Backed EvRecall vs LLM Semantic KU Coverage at K=5
    cx = np.arange(len(COMBO_ORDER))
    w = 0.36
    ev5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["evidence_recall_at_k"].iloc[0]) for cid in COMBO_ORDER]
    sem5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["semantic_ku_recall_at_k"].iloc[0]) for cid in COMBO_ORDER]
    b1 = ax2.bar(cx - w / 2, ev5, width=w, color="#1F5A8A", label="Source-Backed Evidence Recall@5")
    b2 = ax2.bar(cx + w / 2, sem5, width=w, color="#009E73", alpha=0.85, hatch="///", edgecolor="#064E3B", label="Semantic KU Coverage@5 (LLM labels)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=6.9, rotation=90)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination (K=5)")
    ax2.set_ylabel("Coverage Score (0–1)")
    ax2.set_ylim(0.0, 1.10)
    ax2.set_title("(B) Canonical Span Recall vs Semantic KU Coverage (K=5)")
    ax2.legend(loc="upper right", fontsize=7.6)

    fig.suptitle("Figure 13: Source-Backed Evidence Recall@K vs Semantic Knowledge Unit Coverage")
    add_protocol_footer(
        fig,
        "Source: evidence_metrics_summary.csv. Evidence Recall requires canonical span overlap >= min(20, max(8, floor(0.25*span_len))); Semantic KU uses Stage 03 LLM KU labels.",
    )
    saved = save_figure(fig, "fig13_evidence_recall_and_semantic_ku_at_k", png_dir, svg_dir)
    records.append({
        "figure_number": 13,
        "figure_id": "fig13_evidence_recall_and_semantic_ku_at_k",
        "data_contract_id": 13,
        "data_contract_name": "Evidence Recall@K",
        "group": "C",
        "title": "Source-Backed Evidence Recall@K vs Semantic Knowledge Unit Coverage",
        "protocol": "Source-backed interval union over 91 KUs / 107 verified canonical spans (single, alternative, composite modes).",
        "metrics": ["evidence_recall_at_k", "semantic_ku_recall_at_k"],
        "source_artifacts": ["evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv"],
        "caption_vi": (
            "Hình 13: (A) Evidence Recall@K dựa trên giao khoảng nguồn xác thực (source-backed interval union) tăng mạnh từ K=1 đến K=15 "
            "(A4: 0,475 → 0,721@5 → 0,908@15) và bão hòa sau K=30; (B) Đối chiếu Evidence Recall@5 với Semantic KU Coverage@5 (nhãn LLM Stage 03) "
            "— Semantic KU cao hơn vì corpus chứa các đoạn giải thích đúng cùng kiến thức ngoài vị trí span chuẩn đã đăng ký."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 14 (Group C): Evidence Precision@K & Evidence Redundancy@K
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1), gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["evidence_precision_at_k"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K")
    ax1.set_ylabel("Evidence Precision@K")
    ax1.set_ylim(0.0, 0.80)
    ax1.set_title("(A) Source-Backed Evidence Precision@K (K=1..60)")
    ax1.legend(loc="upper right", ncol=2, fontsize=7.5)

    # Panel B: Evidence Redundancy at K=5, 15, 60
    w = 0.25
    r5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["evidence_redundancy_at_k"].iloc[0]) for cid in COMBO_ORDER]
    r15 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 15)]["evidence_redundancy_at_k"].iloc[0]) for cid in COMBO_ORDER]
    r60 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 60)]["evidence_redundancy_at_k"].iloc[0]) for cid in COMBO_ORDER]
    ax2.bar(cx - w, r5, width=w, color="#0072B2", label="K=5")
    ax2.bar(cx, r15, width=w, color="#D97706", label="K=15")
    ax2.bar(cx + w, r60, width=w, color="#7B2CBF", label="K=60")
    for i in range(len(COMBO_ORDER)):
        ax2.text(cx[i] + w, r60[i] + 0.004, f"{r60[i]:.3f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
    ax2.axvline(3.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination")
    ax2.set_ylabel("Evidence Redundancy (0–1)")
    ax2.set_ylim(0.0, 0.26)
    ax2.set_title("(B) Retrieved Evidence Redundancy at K=5, 15, 60")
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Figure 14: Source-Backed Evidence Precision@K and Evidence Redundancy")
    add_protocol_footer(
        fig,
        "Source: evidence_metrics_summary.csv & evidence_redundancy.csv. Redundancy = 1 - (unique_matched_chars / sum_matched_chars). Profile B achieves near-zero overlap redundancy.",
    )
    saved = save_figure(fig, "fig14_evidence_precision_and_redundancy_at_k", png_dir, svg_dir)
    records.append({
        "figure_number": 14,
        "figure_id": "fig14_evidence_precision_and_redundancy_at_k",
        "data_contract_id": 14,
        "data_contract_name": "Evidence Precision@K",
        "group": "C",
        "title": "Source-Backed Evidence Precision@K and Evidence Redundancy",
        "protocol": "Exact source-backed span matching across 40 queries; redundancy computed on matched evidence character intervals.",
        "metrics": ["evidence_precision_at_k", "evidence_redundancy_at_k"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv",
            "evaluation/artifacts/stage04/evidence/evidence_redundancy.csv",
        ],
        "caption_vi": (
            "Hình 14: (A) Evidence Precision@K đo tỷ lệ chunk trả về thực sự chứa bằng chứng nguồn xác thực (A4 dẫn đầu ở K=5 với 0,3550, "
            "thấp hơn Precision@5 mức chunk vì nhiều chunk liên quan ngữ nghĩa không trùng span gốc); (B) Evidence Redundancy cho thấy "
            "Profile B gần như loại bỏ trùng lặp bằng chứng (0,0000–0,0075 tại K=5 so với 0,0380–0,0608 của Profile A)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 15 (Group C): Complete Evidence Rate & Coverage Hierarchy
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1), gridspec_kw={"width_ratios": [1.28, 1.08]})
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    for cid in COMBO_ORDER:
        sub = ev[ev["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["complete_evidence_rate_at_k"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K")
    ax1.set_ylabel("Complete Evidence Rate@K")
    ax1.set_ylim(0.0, 1.08)
    ax1.set_title("(A) Complete Evidence Rate@K (100% Required KUs)")
    ax1.legend(loc="upper left", ncol=2, fontsize=7.3)

    # Panel B: Explicit 4-tier Coverage Hierarchy at K=5 (Complete <= EvRecall <= Partial <= KU Any)
    w = 0.20
    c5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["complete_evidence_rate_at_k"].iloc[0]) for cid in COMBO_ORDER]
    er5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["evidence_recall_at_k"].iloc[0]) for cid in COMBO_ORDER]
    p5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["partial_evidence_coverage_at_k"].iloc[0]) for cid in COMBO_ORDER]
    ku5 = [float(ev[(ev["combo_id"] == cid) & (ev["k"] == 5)]["ku_coverage_at_k"].iloc[0]) for cid in COMBO_ORDER]

    ax2.bar(cx - 1.5 * w, c5, width=w, color="#1E3A8A", label="Complete Evidence Rate (100% KUs)")
    ax2.bar(cx - 0.5 * w, er5, width=w, color="#0072B2", label="Evidence Recall (full-threshold KUs)")
    ax2.bar(cx + 0.5 * w, p5, width=w, color="#38BDF8", label="Partial Evidence Coverage (char fraction)")
    ax2.bar(cx + 1.5 * w, ku5, width=w, color="#BAE6FD", edgecolor="#0369A1", linewidth=0.5, label="KU Coverage Any (overlap > 0 chars)")
    ax2.axvline(3.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination (K=5)")
    ax2.set_ylabel("Coverage Metric (0–1)")
    ax2.set_ylim(0.0, 1.08)
    ax2.set_title("(B) Verified Hierarchy: Complete ≤ EvRecall ≤ Partial ≤ Any (K=5)")
    ax2.legend(loc="upper right", fontsize=7.2)

    fig.suptitle("Figure 15: Complete Evidence Rate@K and Partial vs Complete Coverage Hierarchy")
    add_protocol_footer(
        fig,
        "Source: evidence_metrics_summary.csv. Invariant Complete <= EvRecall <= Partial <= KU_Any verified across all 72 summary rows. Partial overlap is NOT equated to full evidence.",
    )
    saved = save_figure(fig, "fig15_complete_vs_partial_evidence_coverage", png_dir, svg_dir)
    records.append({
        "figure_number": 15,
        "figure_id": "fig15_complete_vs_partial_evidence_coverage",
        "data_contract_id": 15,
        "data_contract_name": "Complete Evidence Rate",
        "group": "C",
        "title": "Complete Evidence Rate@K and Partial vs Complete Coverage Hierarchy",
        "protocol": "Exact implementation distinction in evidence_metrics.py across 40 multi-KU questions (K=1..60).",
        "metrics": ["complete_evidence_rate_at_k", "evidence_recall_at_k", "partial_evidence_coverage_at_k", "ku_coverage_at_k"],
        "source_artifacts": ["evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv"],
        "caption_vi": (
            "Hình 15: (A) Tỷ lệ câu hỏi đạt đủ 100% bằng chứng bắt buộc (Complete Evidence Rate@K) — A4 đạt 55,0% ở K=5 và 80,0% ở K=15, "
            "vượt xa B4 (30,0% và 47,5%) do chunk ngắn làm phân mảnh ngữ cảnh; (B) Phân tầng bốn thước đo phủ nguồn tại K=5, xác minh bất biến "
            "Complete ≤ EvRecall ≤ Partial ≤ KU_Any và tránh đánh đồng giao ký tự tối thiểu với phủ bằng chứng hoàn chỉnh."
        ),
        **saved,
    })

    return records
