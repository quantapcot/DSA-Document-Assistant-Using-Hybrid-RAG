"""Groups E, F, G (Figures 21-24): Answer Generation, Ablation Study & Sensitivity Analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.stage05 import ALL_GEN_ORDER, COMBO_ORDER, KS
from evaluation.stage05.data_loader import Stage04Bundle
from evaluation.stage05.style import add_protocol_footer, combo_style, save_figure


def generate_generation_ablation_sensitivity_figures(
    bundle: Stage04Bundle,
    png_dir: Path,
    svg_dir: Path,
) -> list[dict[str, Any]]:
    """Generate Figures 21-24 (Groups E, F, G)."""
    ans = bundle.answer_summary.set_index("combo_id")
    rag_c = bundle.rag_contribution
    abl = bundle.ablation_results
    pool_df = bundle.candidate_pool_sensitivity
    topk_df = bundle.top_k_sensitivity
    bud_df = bundle.context_budget_sensitivity
    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fig 21 (Group E): Answer Generation Quality Comparison (All 10 Metrics + N/A)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(wspace=0.22, bottom=0.19, top=0.82)

    core_metrics = [
        ("answer_correctness_norm", "Correctness"),
        ("answer_relevancy_norm", "Relevancy"),
        ("faithfulness_norm", "Faithfulness"),
        ("answer_completeness_norm", "Completeness"),
        ("citation_accuracy_norm", "Citation Acc."),
        ("citation_validity", "Citation Valid."),
    ]
    mat1 = np.full((len(ALL_GEN_ORDER), len(core_metrics)), np.nan)
    for r, cid in enumerate(ALL_GEN_ORDER):
        for c, (col, _) in enumerate(core_metrics):
            val = ans.loc[cid, col]
            if pd.notna(val):
                mat1[r, c] = float(val)

    cmap1 = plt.cm.YlGnBu.copy()
    cmap1.set_bad(color="#E5E7EB")
    im1 = ax1.imshow(np.ma.masked_invalid(mat1), cmap=cmap1, vmin=0.30, vmax=1.00, aspect="auto")
    for r in range(mat1.shape[0]):
        for c in range(mat1.shape[1]):
            v = mat1[r, c]
            if np.isnan(v):
                ax1.text(c, r, "N/A", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#4B5563")
            else:
                txt_clr = "#FFFFFF" if v >= 0.78 else "#111827"
                ax1.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8.1, fontweight="bold", color=txt_clr)
    ax1.set_xticks(range(len(core_metrics)), [lbl for _, lbl in core_metrics], rotation=18, ha="right")
    ax1.set_yticks(range(len(ALL_GEN_ORDER)), list(ALL_GEN_ORDER))
    ax1.axhline(7.5, color="#111827", linewidth=1.6)
    ax1.set_title("(A) Core Answer & Citation Quality (0–1, Higher = Better)")
    fig.colorbar(im1, ax=ax1, fraction=0.038, pad=0.02, label="Score (0–1)")

    # Panel B: Error Rates (lower = better) + DSA-specific Complexity & Code Correctness
    sub_metrics = [
        ("unsupported_claim_rate", "Unsupported\n(Lower Better)"),
        ("contradiction_rate", "Contradiction\n(Lower Better)"),
        ("complexity_correctness_norm", "Complexity\nCorrectness"),
        ("code_correctness_norm", "Code\nCorrectness"),
    ]
    from matplotlib.patches import Patch

    x = np.arange(len(ALL_GEN_ORDER))
    w = 0.20
    colors_b = ["#DC2626", "#991B1B", "#0072B2", "#009E73"]
    legend_patches = []
    for idx, (col, lbl) in enumerate(sub_metrics):
        vals = [float(ans.loc[c, col]) if pd.notna(ans.loc[c, col]) else np.nan for c in ALL_GEN_ORDER]
        offsets = x + (idx - 1.5) * w
        for xi, v in zip(offsets, vals):
            if np.isnan(v):
                ax2.text(xi, 0.03, "N/A", ha="center", va="bottom", fontsize=6.6, rotation=90, color="#4B5563", fontweight="bold")
            else:
                ax2.bar(xi, v, width=w, color=colors_b[idx], edgecolor="#1F2937", linewidth=0.4)
        legend_patches.append(Patch(facecolor=colors_b[idx], edgecolor="#1F2937", linewidth=0.5, label=lbl))

    ax2.axvline(7.5, color="#111827", linestyle="--", linewidth=1.1)
    ax2.set_xticks(x, list(ALL_GEN_ORDER), rotation=25, ha="right")
    ax2.set_ylabel("Rate / Score (0–1)")
    ax2.set_ylim(0.0, 1.28)
    ax2.set_title("(B) Error Rates & DSA Technical Correctness")
    ax2.legend(handles=legend_patches, loc="upper center", ncol=2, fontsize=7.5)

    fig.suptitle("Figure 21: Controlled Agent Answer Generation Quality (All 10 Metrics; NO_RAG N/A Preserved)")
    add_protocol_footer(
        fig,
        "Source: answer_metrics_summary.csv (360 main answers, budget=2048, flash gen + blind pro judge, human_validated=false). NOT live Groq production. NO_RAG > A4 in Correctness (0.969 vs 0.925); A1 > A4 in Faithfulness & Cit. Acc.",
    )
    saved = save_figure(fig, "fig21_answer_quality_comparison", png_dir, svg_dir)
    records.append({
        "figure_number": 21,
        "figure_id": "fig21_answer_quality_comparison",
        "data_contract_id": 21,
        "data_contract_name": "Answer Quality Comparison",
        "group": "E",
        "title": "Controlled Agent Answer Generation Quality (All 10 Metrics; NO_RAG N/A Preserved)",
        "protocol": "Controlled Agent Generation (antigravity_agent_flash + blind antigravity_agent_pro judge, 2,048 token budget, 40 queries x 9 configs = 360 main answers). Production Groq API BLOCKED.",
        "metrics": [
            "answer_correctness_norm", "answer_relevancy_norm", "faithfulness_norm", "answer_completeness_norm",
            "citation_accuracy_norm", "citation_validity", "unsupported_claim_rate", "contradiction_rate",
            "complexity_correctness_norm", "code_correctness_norm",
        ],
        "source_artifacts": [
            "evaluation/artifacts/stage04/generation/answer_metrics_summary.csv",
            "evaluation/artifacts/stage04/generation/answer_quality_audit.json",
        ],
        "caption_vi": (
            "Hình 21: So sánh toàn bộ 10 thước đo chất lượng sinh câu trả lời trong thí nghiệm Controlled Agent (không đại diện hiệu năng "
            "Groq production). Bảo toàn nguyên giá trị N/A của NO_RAG ở các thước đo gắn với ngữ cảnh. Công bố rõ: NO_RAG đạt Correctness "
            "(0,969) và Relevancy (1,000) cao hơn A4 (0,925 và 0,956); A1 nhỉnh hơn A4 về Faithfulness (0,925 vs 0,919) và Citation Accuracy "
            "(0,850 vs 0,819); 100% Citation Validity không đồng nghĩa với 100% Citation Accuracy."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 22 (Groups E & F): RAG vs Without-RAG & Component Ablation Study
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.0, 1.32]})
    fig.subplots_adjust(wspace=0.32, bottom=0.19, top=0.82)

    # Panel A: RAG - NO_RAG Paired Differences on Correctness with 95% Bootstrap CI
    rc_corr = rag_c[rag_c["metric"] == "answer_correctness_norm"].set_index("combo_id")
    y_rag = np.arange(len(COMBO_ORDER))
    diffs = np.array([float(rc_corr.loc[c, "abs_diff"]) for c in COMBO_ORDER])
    lo = np.array([float(rc_corr.loc[c, "ci_95_low"]) for c in COMBO_ORDER])
    hi = np.array([float(rc_corr.loc[c, "ci_95_high"]) for c in COMBO_ORDER])
    pvals = [float(rc_corr.loc[c, "p_value_raw"]) for c in COMBO_ORDER]

    bar_clrs = ["#0072B2" if p >= 0.05 else "#D55E00" for p in pvals]
    ax1.axvline(0.0, color="#111827", linewidth=1.2)
    ax1.barh(y_rag, diffs, height=0.55, color=bar_clrs, alpha=0.88, edgecolor="#1F2937", linewidth=0.5)
    ax1.errorbar(diffs, y_rag, xerr=[diffs - lo, hi - diffs], fmt="none", ecolor="#111827", elinewidth=1.3, capsize=3.5)
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        sig = "ns" if p >= 0.05 else "*"
        xpos = lo[i] - 0.015
        ax1.text(xpos, y_rag[i], f"{d:+.3f} (p={p:.3f}, {sig})", va="center", ha="right", fontsize=7.5)
    ax1.set_yticks(y_rag, list(COMBO_ORDER))
    ax1.invert_yaxis()
    ax1.set_xlim(-0.88, 0.14)
    ax1.set_xlabel("Paired Δ Answer Correctness (RAG minus NO_RAG) with 95% CI")
    ax1.set_title("(A) RAG vs NO_RAG Correctness (Contradiction: A4=0.000 vs NO_RAG=0.018)")

    # Panel B: Component Ablation Study (BM25 vs Dense, RRF vs Singles, Reranker, Adaptive vs Recursive)
    # Include direct BM25 vs Dense computed from retrieval_metrics_summary + official paired_bootstrap/ablation rows
    abl_k5 = abl[(abl["k"] == 5) & (abl["metric"] == "ndcg_at_k")].set_index("comparison_id")
    selected_ids = [
        ("R1_A4_vs_A3", "Rerank (Rec): A4 − A3"),
        ("R2_B4_vs_B3", "Rerank (Ada): B4 − B3"),
        ("H2_A3_vs_A2", "RRF vs Dense (Rec): A3 − A2"),
        ("H4_B3_vs_B2", "RRF vs Dense (Ada): B3 − B2"),
        ("H1_A3_vs_A1", "RRF vs BM25 (Rec): A3 − A1"),
        ("H3_B3_vs_B1", "RRF vs BM25 (Ada): B3 − B1"),
        ("C1_B1_vs_A1", "Chunk (BM25): B1 − A1"),
        ("C2_B2_vs_A2", "Chunk (Dense): B2 − A2"),
        ("C3_B3_vs_A3", "Chunk (RRF): B3 − A3"),
        ("C4_B4_vs_A4", "Chunk (Rerank): B4 − A4"),
    ]
    # Also compute BM25 vs Dense directly from per-query retrieval_metrics_per_query.csv for exact paired CI!
    pq = pd.read_csv(bundle.stage04_dir / "retrieval/retrieval_metrics_per_query.csv")
    pq5 = pq[pq["k"] == 5].pivot(index="query_id", columns="combo_id", values="ndcg_at_k").sort_index()
    rng = np.random.default_rng(42)

    def _paired_ci(a_col: str, b_col: str) -> tuple[float, float, float, float]:
        d = (pq5[a_col] - pq5[b_col]).to_numpy()
        idx_b = rng.integers(0, len(d), size=(10000, len(d)))
        means = d[idx_b].mean(axis=1)
        rel = float(d.mean() / pq5[b_col].mean())
        return float(d.mean()), rel, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    bm25_vs_dense_rec = _paired_ci("A1", "A2")
    bm25_vs_dense_ada = _paired_ci("B1", "B2")

    rows_b: list[tuple[str, float, float, float, float, str]] = [
        ("BM25 vs Dense (Rec): A1 − A2", bm25_vs_dense_rec[0], bm25_vs_dense_rec[1], bm25_vs_dense_rec[2], bm25_vs_dense_rec[3], "boot"),
        ("BM25 vs Dense (Ada): B1 − B2", bm25_vs_dense_ada[0], bm25_vs_dense_ada[1], bm25_vs_dense_ada[2], bm25_vs_dense_ada[3], "boot"),
    ]
    for cid_key, lbl in selected_ids:
        r = abl_k5.loc[cid_key]
        sig = f"Holm p={r['p_value_holm']:.3f}*" if bool(r["significant_holm_005"]) else f"Holm p={r['p_value_holm']:.2f} ns"
        rows_b.append((lbl, float(r["abs_improvement"]), float(r["rel_improvement"]), float(r["ci_95_low"]), float(r["ci_95_high"]), sig))

    yb = np.arange(len(rows_b))
    vals_b = np.array([x[1] for x in rows_b])
    lo_b = np.array([x[3] for x in rows_b])
    hi_b = np.array([x[4] for x in rows_b])
    clrs_b = ["#009E73" if v > 0 else "#DC2626" for v in vals_b]

    ax2.axvline(0.0, color="#111827", linewidth=1.2)
    ax2.barh(yb, vals_b, height=0.60, color=clrs_b, alpha=0.86, edgecolor="#1F2937", linewidth=0.45)
    ax2.errorbar(vals_b, yb, xerr=[vals_b - lo_b, hi_b - vals_b], fmt="none", ecolor="#111827", elinewidth=1.15, capsize=3.0)
    for i, (_, v, rel, l_ci, h_ci, sig) in enumerate(rows_b):
        tag = f"{v:+.3f} ({rel * 100:+.1f}%) [{sig}]" if sig != "boot" else f"{v:+.3f} ({rel * 100:+.1f}%)"
        if v >= 0:
            ax2.text(h_ci + 0.008, yb[i], tag, va="center", ha="left", fontsize=7.0)
        else:
            ax2.text(l_ci - 0.008, yb[i], tag, va="center", ha="right", fontsize=7.0)
    ax2.set_yticks(yb, [x[0] for x in rows_b])
    ax2.invert_yaxis()
    ax2.set_xlim(-0.58, 0.56)
    ax2.set_xlabel("Paired Δ nDCG@5 (Absolute Diff, Relative %, 95% Bootstrap CI)")
    ax2.set_title("(B) Component Ablation Study on nDCG@5 (10,000-Resample CI)")

    fig.suptitle("Figure 22: RAG vs Without-RAG Contribution and Retrieval/Chunking Ablation Study")
    add_protocol_footer(
        fig,
        "Source: rag_contribution.csv, ablation_results.csv & retrieval_metrics_per_query.csv (n=40 queries, 10,000 bootstrap resamples, seed=42). Non-significance (ns) does NOT prove equivalence.",
    )
    saved = save_figure(fig, "fig22_rag_vs_norag_and_ablation_study", png_dir, svg_dir)
    records.append({
        "figure_number": 22,
        "figure_id": "fig22_rag_vs_norag_and_ablation_study",
        "data_contract_id": 22,
        "data_contract_name": "RAG vs Without-RAG",
        "group": "E, F",
        "title": "RAG vs Without-RAG Contribution and Retrieval/Chunking Ablation Study",
        "protocol": "Paired bootstrap differences (10,000 resamples, seed=42, Holm-Bonferroni correction across ablation families).",
        "metrics": ["abs_diff", "abs_improvement", "rel_improvement", "ci_95_low", "ci_95_high", "p_value_holm"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/ablation/rag_contribution.csv",
            "evaluation/artifacts/stage04/ablation/ablation_results.csv",
            "evaluation/artifacts/stage04/retrieval/retrieval_metrics_per_query.csv",
        ],
        "caption_vi": (
            "Hình 22: (A) Chênh lệch ghép cặp Answer Correctness giữa từng cấu hình RAG và NO_RAG kèm khoảng tin cậy 95% — A4 thấp hơn "
            "NO_RAG -0,044 (p=0,238, chưa có ý nghĩa thống kê ở α=0,05, không chứng minh tương đương) nhưng giảm Contradiction xuống 0,000; "
            "(B) Nghiên cứu Ablation trên nDCG@5 cho thấy Reranker là nguồn tăng lớn nhất (+0,202 / +40,0% ở Profile A, Holm p=0,001), "
            "BM25 vượt mạnh Dense, còn Adaptive giảm nDCG@5 có ý nghĩa ở BM25 và Rerank."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 23 (Group G): Candidate Pool Sensitivity (Single-Pass) & Top-K Sensitivity
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1))
    fig.subplots_adjust(wspace=0.26, bottom=0.19, top=0.83)

    # Panel A: Candidate Pool Sensitivity (pool = 10, 30, 60, 120; single-pass 640 calls)
    for cid in ["A3", "A4", "B3", "B4"]:
        sub = pool_df[pool_df["combo_id"] == cid].sort_values("candidate_top_k")
        st = combo_style(cid)
        ax1.plot(
            sub["mean_latency_ms"].to_numpy(),
            sub["ndcg_at_5"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=2.0, markersize=6.5, label=cid,
        )
        if cid in ("A3", "B3"):
            mid_x = float(sub["mean_latency_ms"].mean())
            mid_y = float(sub["ndcg_at_5"].mean())
            dy_c = 0.020 if cid == "A3" else -0.028
            ax1.text(mid_x, mid_y + dy_c, f"{cid} (p=10..120)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#065F46")
        else:
            for _, row in sub.iterrows():
                pool_v = int(row["candidate_top_k"])
                if cid == "A4" and pool_v == 30:
                    ax1.annotate(
                        f"A4 pool=30\nnDCG@5={row['ndcg_at_5']:.4f} (99.8%)\nMean={row['mean_latency_ms']:,.0f} ms",
                        xy=(row["mean_latency_ms"], row["ndcg_at_5"]),
                        xytext=(row["mean_latency_ms"] + 260, row["ndcg_at_5"] - 0.115),
                        fontsize=7.8, fontweight="bold", color="#4C1D95",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F3FF", edgecolor="#7B2CBF"),
                        arrowprops=dict(arrowstyle="->", color="#7B2CBF", lw=1.3),
                    )
                else:
                    dx, dy = (55, 0.008) if pool_v != 10 else (55, -0.006)
                    ax1.text(row["mean_latency_ms"] + dx, row["ndcg_at_5"] + dy, f"p={pool_v}", fontsize=7.4, color="#334155")
    ax1.set_xlabel("Single-Pass Mean Retrieval Latency (ms) [Protocol 5]")
    ax1.set_ylabel("nDCG@5 (Judged@5 = 1.0)")
    ax1.set_xlim(500, 4050)
    ax1.set_ylim(0.35, 0.78)
    ax1.set_title("(A) Candidate Pool Sensitivity (pool=10, 30, 60, 120; Single-Pass)")
    ax1.legend(loc="lower right", fontsize=8.2)

    # Panel B: Top-K Sensitivity Operating Curve for A4 & B4 (K=1..60)
    x_k = np.arange(len(KS))
    a4_tk = topk_df[topk_df["combo_id"] == "A4"].sort_values("k")
    b4_tk = topk_df[topk_df["combo_id"] == "B4"].sort_values("k")
    ax2.plot(x_k, a4_tk["ndcg_at_k"], color="#7B2CBF", marker="D", linewidth=2.0, label="A4 nDCG@K")
    ax2.plot(x_k, a4_tk["precision_at_k"], color="#0072B2", marker="o", linewidth=1.8, label="A4 Precision@K")
    ax2.plot(x_k, a4_tk["evidence_recall_at_k"], color="#009E73", marker="^", linewidth=2.0, label="A4 Evidence Recall@K")
    ax2.plot(x_k, a4_tk["complete_evidence_rate_at_k"], color="#D97706", marker="s", linewidth=1.8, label="A4 Complete Evidence@K")
    ax2.plot(x_k, b4_tk["evidence_recall_at_k"], color="#009E73", linestyle="--", marker="^", linewidth=1.5, alpha=0.8, label="B4 Evidence Recall@K")

    ax2.axvspan(2, 3, color="#DBEAFE", alpha=0.45, label="Practical Chat Zone (K=5–10)")
    ax2.axvline(4, color="#B45309", linestyle=":", linewidth=1.2, label="High-Completeness K=15 (A4: 80%)")
    ax2.set_xticks(x_k, [str(k) for k in KS])
    ax2.set_xlabel("Cutoff K")
    ax2.set_ylabel("Metric Score (0–1)")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("(B) Top-K Operating Trade-Off (Precision vs Evidence Recall)")
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.43), fontsize=7.0, ncol=1)

    fig.suptitle("Figure 23: Candidate Pool Sensitivity (Single-Pass) and Top-K Operating Sensitivity")
    add_protocol_footer(
        fig,
        "Source: candidate_pool_sensitivity.csv (Protocol 5: 640 single-pass calls, separate from Main Warm Latency) & top_k_sensitivity.csv. A4 pool=30 retains 99.8% of pool=120 nDCG@5.",
    )
    saved = save_figure(fig, "fig23_candidate_pool_and_topk_sensitivity", png_dir, svg_dir)
    records.append({
        "figure_number": 23,
        "figure_id": "fig23_candidate_pool_and_topk_sensitivity",
        "data_contract_id": 23,
        "data_contract_name": "Candidate Pool Sensitivity",
        "group": "G",
        "title": "Candidate Pool Sensitivity (Single-Pass) and Top-K Operating Sensitivity",
        "protocol": "Protocol 5 — Candidate Pool Sensitivity (640 single-pass calls, pool=10,30,60,120, K=5) + Top-K sensitivity across K=1..60.",
        "metrics": ["ndcg_at_5", "evidence_recall_at_5", "mean_latency_ms", "p95_latency_ms", "precision_at_k", "complete_evidence_rate_at_k"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/sensitivity/candidate_pool_sensitivity.csv",
            "evaluation/artifacts/stage04/sensitivity/top_k_sensitivity.csv",
        ],
        "caption_vi": (
            "Hình 23: (A) Phân tích độ nhạy kích thước Candidate Pool (đo single-pass độc lập, không trộn với Main Warm Latency): "
            "A4 tại pool=30 giữ 99,8% nDCG@5 so với pool=120 (0,7039 vs 0,7053) nhưng giảm độ trễ trung bình từ 3.486 ms xuống 1.628 ms; "
            "(B) Độ nhạy Top-K cho thấy vùng K=5–10 tối ưu cho chat ngắn gọn, còn K=15 đạt 80,0% Complete Evidence ở A4 trước khi bão hòa sau K=30."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 24 (Group G): Context Budget Sensitivity (Exploratory n=10)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9))
    fig.subplots_adjust(wspace=0.27, bottom=0.19, top=0.83)

    budgets = [1024, 2048, 4096]
    bx = np.arange(len(budgets))
    metric_specs = [
        ("answer_correctness_norm", "Correctness", "#0072B2", "o"),
        ("faithfulness_norm", "Faithfulness", "#009E73", "s"),
        ("answer_completeness_norm", "Completeness", "#7B2CBF", "^"),
        ("citation_accuracy_norm", "Citation Acc.", "#D55E00", "D"),
    ]
    for ax, cid, ttl in [
        (ax1, "A4", "(A) Profile A — A4 (Recursive + Hybrid Rerank, n=10)"),
        (ax2, "B4", "(B) Profile B — B4 (Adaptive + Hybrid Rerank, n=10)"),
    ]:
        sub = bud_df[bud_df["combo_id"] == cid].sort_values("token_budget")
        for col, lbl, clr, mk in metric_specs:
            vals = sub[col].to_numpy()
            ax.plot(bx, vals, color=clr, marker=mk, linewidth=2.0, markersize=6.2, label=lbl)
            for xi, v in zip(bx, vals):
                ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4, color=clr, fontweight="bold")
        tok_labels = [
            f"{int(r['token_budget'])}\n(actual ~{r['mean_actual_context_tokens']:.0f})"
            for _, r in sub.iterrows()
        ]
        ax.set_xticks(bx, tok_labels)
        ax.set_xlabel("Context Token Budget (Actual Mean Packed Tokens)")
        ax.set_ylabel("Controlled Agent Score (0–1)")
        ax.set_ylim(0.38, 1.09)
        ax.set_title(ttl)
        ax.legend(loc="lower left", fontsize=7.8, ncol=2)

    fig.suptitle("Figure 24: Exploratory Context Budget Sensitivity (1024 / 2048 / 4096 Tokens, n=10 Queries)")
    add_protocol_footer(
        fig,
        "Source: context_budget_sensitivity.csv (exploratory 10-query subset, Controlled Agent Generation). A4 peaks at 2,048 tokens; expanding to 4,096 degrades Faithfulness (0.925 -> 0.825) & Citation Acc (0.850 -> 0.700).",
    )
    saved = save_figure(fig, "fig24_context_budget_sensitivity", png_dir, svg_dir)
    records.append({
        "figure_number": 24,
        "figure_id": "fig24_context_budget_sensitivity",
        "data_contract_id": 24,
        "data_contract_name": "Context Budget Sensitivity",
        "group": "G",
        "title": "Exploratory Context Budget Sensitivity (1024 / 2048 / 4096 Tokens, n=10 Queries)",
        "protocol": "Exploratory Controlled Agent subset (10 queries x {A4, B4} x {1024, 2048, 4096} tokens = 40 budget answers + 20 main answers).",
        "metrics": ["mean_actual_context_tokens", "answer_correctness_norm", "faithfulness_norm", "answer_completeness_norm", "citation_accuracy_norm"],
        "source_artifacts": ["evaluation/artifacts/stage04/sensitivity/context_budget_sensitivity.csv"],
        "caption_vi": (
            "Hình 24: Phân tích độ nhạy ngân sách ngữ cảnh (Context Budget: 1.024 / 2.048 / 4.096 tokens) trên mẫu thăm dò 10 câu hỏi cho A4 và B4. "
            "Mức 2.048 tokens mang lại cân bằng tốt nhất cho A4 (Correctness = 0,950, Faithfulness = 0,925); tăng lên 4.096 tokens đưa thêm ngữ cảnh "
            "nhiễu khiến Faithfulness (0,825) và Citation Accuracy (0,700) suy giảm."
        ),
        **saved,
    })

    return records
