"""Groups K, H, I, J (Figures 25-28): Pareto Trade-Off, Statistics, Robustness/Study & Error Analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.stage05 import COMBO_ORDER
from evaluation.stage05.data_loader import Stage04Bundle
from evaluation.stage05.style import add_protocol_footer, combo_style, save_figure


def generate_analysis_figures(
    bundle: Stage04Bundle,
    png_dir: Path,
    svg_dir: Path,
) -> list[dict[str, Any]]:
    """Generate Figures 25-28 (Groups K, H, I, J)."""
    pf = bundle.pareto_frontier
    ci_df = bundle.confidence_intervals
    wtl_df = bundle.win_tie_loss
    ans_df = bundle.annotation_sensitivity
    sg_df = bundle.subgroup_metrics
    rob_df = bundle.robustness_metrics
    st_df = bundle.study_summary
    ft_json = bundle.functional_tests
    ftax = bundle.failure_taxonomy
    err_rows = bundle.error_records
    rep_cases = bundle.representative_cases
    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fig 25 (Group K): Quality-Latency-Resource Pareto Frontier
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    for ax, k_val, q_col, q_lbl, ttl in [
        (ax1, 5, "ndcg_at_k", "nDCG@5", "(A) Retrieval nDCG@5 vs Warm P95 Latency (K=5, pool=120)"),
        (ax2, 15, "evidence_recall_at_k", "Evidence Recall@15", "(B) Evidence Recall@15 vs Warm P95 Latency (K=15, pool=120)"),
    ]:
        sub = pf[pf["k"] == k_val].copy()
        # Plot Pareto frontier polyline sorted by P95 latency
        front = sub[sub["is_pareto_optimal"] == True].sort_values("p95_latency_ms")  # noqa: E712
        ax.plot(
            front["p95_latency_ms"].to_numpy(),
            front[q_col].to_numpy(),
            color="#111827", linestyle="--", linewidth=1.6, zorder=2, label="Observed Pareto Frontier (A1, B4, A4)",
        )
        for _, row in sub.iterrows():
            cid = row["combo_id"]
            st = combo_style(cid)
            gpu_mb = float(row["peak_gpu_allocated_max_mb"])
            size = 95 if gpu_mb == 0 else (165 if gpu_mb < 1000 else 280)
            edge = "#111827" if bool(row["is_pareto_optimal"]) else "#64748B"
            lw = 2.0 if bool(row["is_pareto_optimal"]) else 0.8
            ax.scatter(
                row["p95_latency_ms"], row[q_col],
                s=size, color=st["color"], marker=st["marker"],
                edgecolors=edge, linewidths=lw, alpha=0.92, zorder=4,
            )
            tag = f"{cid}*" if bool(row["is_pareto_optimal"]) else cid
            offsets = {
                "A1": (95, 0.018), "A2": (95, -0.020), "A3": (95, 0.014), "A4": (-620, 0.032),
                "B1": (95, 0.012), "B2": (95, -0.020), "B3": (95, -0.022), "B4": (-640, -0.068),
            }
            dx, dy = offsets.get(cid, (80, 0.01))
            ax.text(
                row["p95_latency_ms"] + dx, row[q_col] + dy,
                f"{tag} ({row[q_col]:.3f}, {row['p95_latency_ms']:.0f}ms)",
                fontsize=7.3, fontweight="bold" if bool(row["is_pareto_optimal"]) else "normal",
                color="#111827" if bool(row["is_pareto_optimal"]) else "#475569",
            )
        ax.set_xlabel("Main Warm P95 Retrieval Latency (ms, pool=120)")
        ax.set_ylabel(q_lbl)
        ax.set_xlim(300, 4350)
        ax.set_ylim(0.24 if k_val == 5 else 0.30, 0.84 if k_val == 5 else 1.03)
        ax.set_title(ttl)
        ax.legend(loc="upper left", fontsize=7.8)
        ax.text(
            0.98, 0.03,
            "Marker size = Peak GPU VRAM:\nSmall: 0 MB | Med: 458 MB | Large: 2.65 GB\n* = Non-dominated Pareto-optimal",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#334155",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )

    fig.suptitle("Figure 25: Quality–Latency–GPU Resource Pareto Frontier (Matched Main Warm Protocol)")
    add_protocol_footer(
        fig,
        "Source: pareto_frontier.csv & quality_latency_tradeoff.csv (matched Protocol 1 Warm pool=120). Non-dominated set = {A1, B4, A4}; each occupies a distinct operating niche.",
    )
    saved = save_figure(fig, "fig25_quality_latency_pareto_tradeoff", png_dir, svg_dir)
    records.append({
        "figure_number": 25,
        "figure_id": "fig25_quality_latency_pareto_tradeoff",
        "data_contract_id": 25,
        "data_contract_name": "Quality-Latency Trade-off",
        "group": "K",
        "title": "Quality–Latency–GPU Resource Pareto Frontier (Matched Main Warm Protocol)",
        "protocol": "Comparable Protocol 1 Main Warm Latency (pool=120) vs Official Retrieval/Evidence quality at K=5 and K=15; no arbitrary composite weights.",
        "metrics": ["ndcg_at_k", "evidence_recall_at_k", "p95_latency_ms", "peak_gpu_allocated_max_mb", "is_pareto_optimal"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/tradeoff/pareto_frontier.csv",
            "evaluation/artifacts/stage04/tradeoff/quality_latency_tradeoff.csv",
        ],
        "caption_vi": (
            "Hình 25: Biên tối ưu Pareto đa tiêu chí Chất lượng – Độ trễ P95 – GPU VRAM trên cùng giao thức Main Warm (pool=120), "
            "không dùng trọng số chủ quan. Ba tổ hợp không bị trội là A1 (độ trễ thấp nhất 607,5 ms, 0 MB VRAM), B4 (trung gian, "
            "rerank nhanh hơn 41%, tổng P95 thấp hơn A4 13,0–16,8%) và A4 (dẫn đầu nDCG@5 = 0,705 và Evidence Recall@15 = 0,908)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 26 (Group H): Statistical CIs, Win-Tie-Loss, Annotation Sensitivity & Subgroups
    # ------------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.2, 9.6))
    fig.subplots_adjust(wspace=0.28, hspace=0.34, bottom=0.09, top=0.90)

    # Panel A: 95% Bootstrap CIs (10,000 resamples) for nDCG@5 & EvRecall@15
    y_ci = np.arange(len(COMBO_ORDER))
    ci_ndcg = ci_df[(ci_df["k"] == 5) & (ci_df["metric"] == "ndcg_at_k")].set_index("combo_id")
    ci_ev15 = ci_df[(ci_df["k"] == 15) & (ci_df["metric"] == "evidence_recall_at_k")].set_index("combo_id")

    for i, c in enumerate(COMBO_ORDER):
        m1, l1, h1 = float(ci_ndcg.loc[c, "mean"]), float(ci_ndcg.loc[c, "ci_95_low"]), float(ci_ndcg.loc[c, "ci_95_high"])
        m2, l2, h2 = float(ci_ev15.loc[c, "mean"]), float(ci_ev15.loc[c, "ci_95_low"]), float(ci_ev15.loc[c, "ci_95_high"])
        ax1.errorbar(m1, y_ci[i] - 0.16, xerr=[[m1 - l1], [h1 - m1]], fmt="D", color="#7B2CBF", capsize=3.2, markersize=5.5, label="nDCG@5 (95% CI)" if i == 0 else None)
        ax1.errorbar(m2, y_ci[i] + 0.16, xerr=[[m2 - l2], [h2 - m2]], fmt="o", color="#009E73", capsize=3.2, markersize=5.5, label="EvRecall@15 (95% CI)" if i == 0 else None)
    ax1.set_yticks(y_ci, list(COMBO_ORDER))
    ax1.invert_yaxis()
    ax1.set_xlim(0.18, 1.00)
    ax1.set_xlabel("Score with 95% Bootstrap CI (10,000 resamples, n=40)")
    ax1.set_title("(A) 95% Bootstrap Confidence Intervals (nDCG@5 & EvRecall@15)")
    ax1.legend(loc="lower right", fontsize=8.0)

    # Panel B: Win–Tie–Loss for Key Paired Comparisons (nDCG@5)
    wtl5 = wtl_df[(wtl_df["k"] == 5) & (wtl_df["metric"] == "ndcg_at_k")].set_index("comparison_id")
    key_comps = [
        ("R1_A4_vs_A3", "A4 vs A3 (Rerank Rec)"),
        ("R2_B4_vs_B3", "B4 vs B3 (Rerank Ada)"),
        ("H2_A3_vs_A2", "A3 vs A2 (RRF vs Dense)"),
        ("H1_A3_vs_A1", "A3 vs A1 (RRF vs BM25)"),
        ("C4_B4_vs_A4", "B4 vs A4 (Ada vs Rec)"),
        ("C1_B1_vs_A1", "B1 vs A1 (Ada vs Rec)"),
    ]
    yw = np.arange(len(key_comps))
    wins = np.array([int(wtl5.loc[k, "wins"]) for k, _ in key_comps])
    ties = np.array([int(wtl5.loc[k, "ties"]) for k, _ in key_comps])
    losses = np.array([int(wtl5.loc[k, "losses"]) for k, _ in key_comps])
    ax2.barh(yw, wins, height=0.55, color="#009E73", label="Wins (new > base)")
    ax2.barh(yw, ties, left=wins, height=0.55, color="#CBD5E1", label="Ties (|Δ| <= 1e-4)")
    ax2.barh(yw, losses, left=wins + ties, height=0.55, color="#DC2626", label="Losses (new < base)")
    for i in range(len(key_comps)):
        if wins[i] >= 3:
            ax2.text(wins[i] / 2, yw[i], str(wins[i]), va="center", ha="center", color="#FFFFFF", fontsize=8.0, fontweight="bold")
        if ties[i] >= 3:
            ax2.text(wins[i] + ties[i] / 2, yw[i], str(ties[i]), va="center", ha="center", color="#111827", fontsize=8.0, fontweight="bold")
        if losses[i] >= 3:
            ax2.text(wins[i] + ties[i] + losses[i] / 2, yw[i], str(losses[i]), va="center", ha="center", color="#FFFFFF", fontsize=8.0, fontweight="bold")
    ax2.set_yticks(yw, [lbl for _, lbl in key_comps])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 40)
    ax2.set_xlabel("Number of Queries (Total n = 40)")
    ax2.set_title("(B) Win–Tie–Loss Counts on nDCG@5")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=7.6)

    # Panel C: Annotation Sensitivity (Primary-Only vs Final Adjudicated)
    ans5 = ans_df[(ans_df["k"] == 5) & (ans_df["metric"] == "ndcg_at_k")].set_index("combo_id")
    ans15 = ans_df[(ans_df["k"] == 15) & (ans_df["metric"] == "ndcg_at_k")].set_index("combo_id")
    cx = np.arange(len(COMBO_ORDER))
    w = 0.35
    s5 = [float(ans5.loc[c, "abs_shift_final_minus_primary"]) for c in COMBO_ORDER]
    s15 = [float(ans15.loc[c, "abs_shift_final_minus_primary"]) for c in COMBO_ORDER]
    ax3.axhline(0.0, color="#111827", linewidth=1.0)
    ax3.bar(cx - w / 2, s5, width=w, color="#0072B2", label="Δ nDCG@5 (Final − Primary)")
    ax3.bar(cx + w / 2, s15, width=w, color="#D97706", label="Δ nDCG@15 (Final − Primary)")
    ax3.set_xticks(cx, list(COMBO_ORDER))
    ax3.set_ylim(-0.017, 0.007)
    ax3.set_ylabel("Absolute Score Shift (Δ nDCG)")
    ax3.set_title("(C) AI Annotation Sensitivity (Max |Δ| = 0.011; Rank Preserved)")
    ax3.legend(loc="lower right", fontsize=7.8)
    ax3.text(
        0.50, 0.94,
        "Note: AI agreement != human-verified accuracy (human_validated=false)",
        transform=ax3.transAxes, ha="center", va="top", fontsize=7.5, style="italic", color="#7C2D12",
    )

    # Panel D: Language & Difficulty Subgroups (nDCG@5 for A1, A4, B1, B4)
    sg5 = sg_df[sg_df["k"] == 5]
    subgroups = [
        ("language", "vi", "VI\n(n=24)"),
        ("language", "en", "EN\n(n=16)"),
        ("difficulty", "easy", "Easy\n(n=12)"),
        ("difficulty", "medium", "Med\n(n=18)"),
        ("difficulty", "hard", "Hard\n(n=10)"),
    ]
    sx = np.arange(len(subgroups))
    w = 0.19
    focus_combos = ["A1", "A4", "B1", "B4"]
    for idx, cid in enumerate(focus_combos):
        st = combo_style(cid)
        vals = [
            float(sg5[(sg5["combo_id"] == cid) & (sg5["subgroup_dim"] == d) & (sg5["subgroup_value"] == v)]["ndcg_at_k"].iloc[0])
            for d, v, _ in subgroups
        ]
        ax4.bar(
            sx + (idx - 1.5) * w, vals, width=w,
            color=st["color"], hatch=st["hatch"], edgecolor="#1F2937", linewidth=0.5, label=cid,
        )
    ax4.axvline(1.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax4.set_xticks(sx, [lbl for _, _, lbl in subgroups])
    ax4.set_ylabel("Mean nDCG@5")
    ax4.set_ylim(0.0, 0.90)
    ax4.set_title("(D) Language & Difficulty Subgroup Performance (nDCG@5)")
    ax4.legend(loc="upper right", ncol=4, fontsize=7.8)

    fig.suptitle("Figure 26: Statistical Uncertainty, Win–Tie–Loss, Annotation Sensitivity & Subgroups", y=0.965)
    add_protocol_footer(
        fig,
        "Source: confidence_intervals.csv, win_tie_loss.csv, annotation_sensitivity.csv & subgroup_metrics.csv (10,000 bootstrap resamples, seed=42, human_validated=false).",
        y=0.008,
    )
    saved = save_figure(fig, "fig26_statistical_ci_subgroups_and_audit", png_dir, svg_dir)
    records.append({
        "figure_number": 26,
        "figure_id": "fig26_statistical_ci_subgroups_and_audit",
        "data_contract_id": 26,
        "data_contract_name": "Statistical Confidence Intervals",
        "group": "H",
        "title": "Statistical Uncertainty, Win–Tie–Loss, Annotation Sensitivity & Subgroups",
        "protocol": "10,000-resample bootstrap (seed=42) on n=40 queries, paired Win-Tie-Loss (tol=1e-4), AI annotation adjudication sensitivity, and VI/EN + Easy/Med/Hard subgroups.",
        "metrics": ["ci_95_low", "ci_95_high", "wins", "ties", "losses", "abs_shift_final_minus_primary", "ndcg_at_k"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/statistics/confidence_intervals.csv",
            "evaluation/artifacts/stage04/statistics/win_tie_loss.csv",
            "evaluation/artifacts/stage04/statistics/annotation_sensitivity.csv",
            "evaluation/artifacts/stage04/statistics/subgroup_metrics.csv",
        ],
        "caption_vi": (
            "Hình 26: Tổng hợp phân tích thống kê (10.000 mẫu bootstrap, seed=42). (A) Khoảng tin cậy 95% cho nDCG@5 và EvRecall@15; "
            "(B) Tỷ số Thắng–Hòa–Thua ghép cặp trên 40 câu hỏi (A4 thắng A3 ở 36/40 câu; B4 thua A4 ở 28/40 câu); (C) Độ nhạy gán nhãn AI "
            "(|Δ nDCG| ≤ 0,011, giữ nguyên xếp hạng nhưng không thay thế xác nhận con người); (D) Hiệu năng theo ngôn ngữ (VI/EN) và độ khó."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 27 (Group I): Robustness Results (n=20) & Study Features (Separated)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.32, 1.0]})
    fig.subplots_adjust(wspace=0.28, bottom=0.20, top=0.82)

    cat_order = [
        ("out_of_domain", "Out-of-Domain"),
        ("unanswerable_in_corpus", "Unanswerable"),
        ("prompt_injection", "Prompt Injection*"),
        ("distractor_context", "Distractor Context"),
        ("semantic_paraphrase", "Semantic Paraphrase"),
        ("typo_noisy", "VI Unaccented / Typo"),
    ]
    ridx = rob_df.set_index("category")
    ry = np.arange(len(cat_order))
    h_rates = [float(ridx.loc[c, "correct_handling_rate"]) * 100.0 for c, _ in cat_order]
    q_rates = [float(ridx.loc[c, "mean_answer_quality_norm"]) * 100.0 for c, _ in cat_order]
    n_counts = [int(ridx.loc[c, "n_cases"]) for c, _ in cat_order]
    degs = [ridx.loc[c, "mean_ev_recall_degradation"] for c, _ in cat_order]

    h_bar = 0.35
    ax1.barh(ry - h_bar / 2, h_rates, height=h_bar, color="#0072B2", label="Correct Handling Rate (%)")
    ax1.barh(ry + h_bar / 2, q_rates, height=h_bar, color="#009E73", alpha=0.85, hatch="///", edgecolor="#064E3B", label="Answer Quality Score (%)")
    for i in range(len(cat_order)):
        deg_str = f" | ΔEvRec={float(degs[i]):+.2f}" if pd.notna(degs[i]) and abs(float(degs[i])) > 1e-6 else ""
        ax1.text(max(h_rates[i], q_rates[i]) + 2.0, ry[i], f"{h_rates[i]:.1f}% / {q_rates[i]:.1f}%{deg_str}", va="center", ha="left", fontsize=7.6)
    ax1.set_yticks(ry, [f"{lbl}\n(n={n})" for (_, lbl), n in zip(cat_order, n_counts)])
    ax1.set_ylim(6.0, -0.55)
    ax1.set_xlim(0, 152)
    ax1.set_xlabel("Percentage / Normalized Score (%)")
    ax1.set_title("(A) Robustness Evaluation (n=20 Independent Cases, A4 Pipeline)")
    ax1.legend(loc="lower right", fontsize=7.6)
    ax1.text(
        0.01, 0.02,
        "*n=3 prompt-injection cases passed; do NOT generalize to overall security.",
        transform=ax1.transAxes, fontsize=7.2, style="italic", color="#991B1B",
    )

    # Panel B: Study Features — Strictly Separating Real Backend Contracts vs Controlled Agent Quiz/Roadmap
    s_idx = st_df.set_index("feature")
    m_cols = [
        ("schema_valid_rate", "Schema\nValid"),
        ("content_correctness_norm", "Content\nCorrect."),
        ("source_grounding_norm", "Source\nGrounding"),
        ("pedagogical_quality_norm", "Pedagog.\nQuality"),
    ]
    sx = np.arange(len(m_cols))
    w = 0.34
    quiz_v = [float(s_idx.loc["quiz_generation_controlled_agent", col]) for col, _ in m_cols]
    road_v = [float(s_idx.loc["study_roadmap_controlled_agent", col]) for col, _ in m_cols]
    b1 = ax2.bar(sx - w / 2, quiz_v, width=w, color="#7B2CBF", label="Controlled Agent Quiz (n=8)")
    b2 = ax2.bar(sx + w / 2, road_v, width=w, color="#D97706", hatch="///", edgecolor="#78350F", label="Controlled Agent Roadmap (n=4)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(sx, [lbl for _, lbl in m_cols])
    ax2.set_ylabel("Normalized Score (0–1)")
    ax2.set_ylim(0.0, 1.42)
    ax2.set_title("(B) Study Features (Controlled Agent vs Real Contract Checks)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.50, 0.855), ncol=2, fontsize=7.4)
    ax2.text(
        0.50, 0.985,
        f"Real Backend Module Contracts (Separate): {ft_json['contract_checks_passed']}/{ft_json['contract_checks_total']} PASSED\n"
        "(Query planner, abstention, DAG topo-sort, citations; guardrail.py = 0B scaffold)",
        transform=ax2.transAxes, ha="center", va="top", fontsize=7.1, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF5", edgecolor="#10B981"),
    )

    fig.suptitle("Figure 27: Robustness Evaluation (n=20) and Study Features Quality vs Contract Checks")
    add_protocol_footer(
        fig,
        "Source: robustness_metrics.csv, study_features_summary.csv & functional_test_results.json. VI unaccented typo failure (0.0%, ΔEvRecall=-0.667) stems from accent_insensitive=false.",
    )
    saved = save_figure(fig, "fig27_robustness_and_study_features", png_dir, svg_dir)
    records.append({
        "figure_number": 27,
        "figure_id": "fig27_robustness_and_study_features",
        "data_contract_id": 27,
        "data_contract_name": "Robustness Results",
        "group": "I",
        "title": "Robustness Evaluation (n=20) and Study Features Quality vs Contract Checks",
        "protocol": "20 independent robustness cases (A4 real retrieval + Controlled Agent generation) and Study Features separating 11/11 real module contract checks from Controlled Agent Quiz (n=8) & Roadmap (n=4).",
        "metrics": ["correct_handling_rate", "abstention_rate", "mean_ev_recall_degradation", "mean_answer_quality_norm", "schema_valid_rate", "content_correctness_norm", "source_grounding_norm", "pedagogical_quality_norm"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/robustness/robustness_metrics.csv",
            "evaluation/artifacts/stage04/study_features/study_features_summary.csv",
            "evaluation/artifacts/stage04/study_features/functional_test_results.json",
        ],
        "caption_vi": (
            "Hình 27: (A) Đánh giá độ bền vững trên 20 ca độc lập kèm cỡ mẫu từng nhóm — OOD (4/4), Unanswerable (4/4) và Prompt Injection (3/3 — "
            "cỡ mẫu nhỏ, không suy rộng thành an toàn toàn hệ thống) đạt 100%, nhưng tiếng Việt không dấu (n=3) thất bại hoàn toàn (0%, EvRecall@5 "
            "giảm -0,67) do accent_insensitive=false; (B) Tách bạch 11/11 kiểm thử hợp đồng module backend thật khỏi điểm chất lượng Quiz (n=8) và Roadmap (n=4) bằng Controlled Agent."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 28 (Group J): Error Taxonomy Distribution & Per-Query Case-Study Map
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.subplots_adjust(wspace=0.28, bottom=0.19, top=0.83)

    nice_names = {
        "2_relevant_evidence_ranked_too_low": "2. Evidence Ranked Too Low",
        "1_relevant_evidence_not_retrieved": "1. Missing Evidence (Not Retrieved)",
        "9_missing_knowledge_unit": "9. Missing Knowledge Unit",
        "10_irrelevant_context_overload": "10. Irrelevant Context Overload",
        "7_fragmented_evidence": "7. Context Fragmentation",
        "4_semantic_mismatch": "4. Semantic Mismatch",
        "8_duplicate_context": "8. Duplicate Evidence",
    }
    ft_sorted = ftax.sort_values("occurrence_count", ascending=True)
    ey = np.arange(len(ft_sorted))
    counts = ft_sorted["occurrence_count"].to_numpy()
    labels_e = [nice_names.get(c, c) for c in ft_sorted["failure_category"]]
    bars = ax1.barh(ey, counts, height=0.58, color="#B91C1C", alpha=0.85, edgecolor="#7F1D1D", linewidth=0.5)
    for bar in bars:
        w_val = bar.get_width()
        ax1.text(w_val + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(w_val)}", va="center", ha="left", fontsize=8.4, fontweight="bold")
    ax1.set_yticks(ey, labels_e)
    ax1.set_xlabel("Observed Occurrence Count Across 40 Queries (A4 / B4)")
    ax1.set_xlim(0, 34)
    ax1.set_title("(A) Observed Failure Taxonomy (Total = 85 Tag Occurrences)")

    # Panel B: Per-Query A4 vs B4 nDCG@5 Scatter Highlighting Representative Case Studies
    a4_ndcg = np.array([float(r["ndcg_at_5"]["A4"]) for r in err_rows])
    b4_ndcg = np.array([float(r["ndcg_at_5"]["B4"]) for r in err_rows])
    qids = [r["query_id"] for r in err_rows]

    weak_ids = {c["query_id"] for c in rep_cases.get("all_methods_weak", [])}
    ada_win_ids = {c["query_id"] for c in rep_cases.get("adaptive_better_rerank", [])}
    deg_ids = {c["query_id"] for c in rep_cases.get("reranker_degrades_ndcg5", [])}

    ax2.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.1, label="Parity Line (A4 = B4)")
    for qid, x_v, y_v in zip(qids, a4_ndcg, b4_ndcg):
        if qid in weak_ids:
            clr, mk, sz, z = "#DC2626", "X", 85, 5
        elif qid in ada_win_ids:
            clr, mk, sz, z = "#D97706", "^", 75, 5
        elif qid in deg_ids:
            clr, mk, sz, z = "#7B2CBF", "s", 68, 5
        else:
            clr, mk, sz, z = "#0072B2", "o", 45, 3
        ax2.scatter(x_v, y_v, color=clr, marker=mk, s=sz, edgecolors="#111827", linewidths=0.5, alpha=0.88, zorder=z)
        if qid in weak_ids | ada_win_ids | deg_ids:
            ax2.text(x_v + 0.015, y_v - 0.025, qid, fontsize=7.4, fontweight="bold", color=clr)

    # Legend proxies
    ax2.scatter([], [], color="#0072B2", marker="o", s=45, label="Other Queries (n=40 total)")
    ax2.scatter([], [], color="#DC2626", marker="X", s=80, label="All Methods Weak (Q007, Q022, Q029)")
    ax2.scatter([], [], color="#D97706", marker="^", s=70, label="Adaptive > Recursive (Q011, Q014, Q015)")
    ax2.scatter([], [], color="#7B2CBF", marker="s", s=65, label="Reranker Degrades vs RRF (Q006–Q008)")
    ax2.set_xlabel("A4 (Recursive + Hybrid Rerank) nDCG@5")
    ax2.set_ylabel("B4 (Adaptive + Hybrid Rerank) nDCG@5")
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("(B) Per-Query A4 vs B4 nDCG@5 & Representative Case Studies")
    ax2.legend(loc="upper left", fontsize=7.3)

    fig.suptitle("Figure 28: Observed Failure Taxonomy and Per-Query Diagnostic Error Map")
    add_protocol_footer(
        fig,
        "Source: failure_taxonomy_summary.csv, error_analysis.jsonl & representative_case_studies.json. Majority of points lie below parity (A4 > B4 in 28/40 queries, B4 > A4 in 10/40).",
    )
    saved = save_figure(fig, "fig28_error_distribution_and_taxonomy", png_dir, svg_dir)
    records.append({
        "figure_number": 28,
        "figure_id": "fig28_error_distribution_and_taxonomy",
        "data_contract_id": 28,
        "data_contract_name": "Error Distribution",
        "group": "J",
        "title": "Observed Failure Taxonomy and Per-Query Diagnostic Error Map",
        "protocol": "Rule-verified failure taxonomy over 40 queries (A4 & B4 at K=5 vs K=60) linked to representative case studies.",
        "metrics": ["occurrence_count", "ndcg_at_5", "evidence_recall_at_5", "evidence_recall_at_60"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/error_analysis/failure_taxonomy_summary.csv",
            "evaluation/artifacts/stage04/error_analysis/error_analysis.jsonl",
            "evaluation/artifacts/stage04/error_analysis/representative_case_studies.json",
        ],
        "caption_vi": (
            "Hình 28: (A) Phân bố 7 dạng lỗi thực đo trên 40 câu hỏi (A4/B4): phổ biến nhất là bằng chứng có tồn tại trong Top-60 nhưng xếp "
            "ngoài Top-5 (29 lượt), không truy xuất được trong Top-60 (17) và thiếu đơn vị kiến thức (17); (B) Biểu đồ phân tán nDCG@5 giữa "
            "A4 và B4 gắn nhãn các ca điển hình (cả hai yếu: Q007/Q022/Q029; Adaptive vượt Recursive: Q011/Q014/Q015)."
        ),
        **saved,
    })

    return records
