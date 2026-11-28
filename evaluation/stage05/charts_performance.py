"""Group D (Figures 16-20): Performance, Latency Breakdown, Cold-Start & Resource Usage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluation.stage05 import COMBO_LABELS, COMBO_ORDER, KS
from evaluation.stage05.data_loader import Stage04Bundle
from evaluation.stage05.style import add_protocol_footer, combo_style, save_figure


def generate_performance_figures(
    bundle: Stage04Bundle,
    png_dir: Path,
    svg_dir: Path,
) -> list[dict[str, Any]]:
    """Generate Figures 16 to 20 (Group D: Performance & Resource Usage)."""
    lat = bundle.latency_summary
    stg = bundle.stage_latency
    cold = bundle.cold_start_latency
    prod = bundle.production_profile_latency
    res = bundle.resource_usage
    pman = bundle.performance_manifest
    records: list[dict[str, Any]] = []
    x_idx = np.arange(len(KS))
    cx = np.arange(len(COMBO_ORDER))

    # ------------------------------------------------------------------
    # Fig 16: Main Warm Mean & P50 Latency@K (14,400 calls, pool=120)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    fig.subplots_adjust(wspace=0.24, bottom=0.19, top=0.83)

    for cid in COMBO_ORDER:
        sub = lat[lat["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["mean_ms"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
        ax2.plot(
            x_idx, sub["p50_ms"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )

    for ax, ttl, ylab in [
        (ax1, "(A) Main Warm Mean Latency@K (pool=120)", "Mean Latency (ms)"),
        (ax2, "(B) Main Warm Median / P50 Latency@K (pool=120)", "P50 Latency (ms)"),
    ]:
        ax.set_xticks(x_idx, [str(k) for k in KS])
        ax.set_xlabel("Cutoff K (Top-K Returned)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 3850)
        ax.set_title(ttl)
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=7.3)

    ax1.text(
        0.02, 0.96,
        f"BM25 TTL (300s) rebuilds = {pman['warm_protocol']['bm25_rebuilds_during_measure']}\n"
        "(causes Mean > P50 tail uplift on BM25/Hybrid combos)",
        transform=ax1.transAxes, va="top", ha="left", fontsize=7.6, color="#7C2D12",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7ED", edgecolor="#FDBA74"),
    )

    fig.suptitle("Figure 16: Main Warm Benchmark Mean and P50 Retrieval Latency Across K=1..60")
    add_protocol_footer(
        fig,
        "Protocol 1 (Main Warm Latency): 14,400 calls (5 interleaved repeats, seed=42, pool=120, RTX 5060 Laptop GPU, 0 errors). Retrieval engine latency only (NOT HTTP API).",
    )
    saved = save_figure(fig, "fig16_mean_latency_at_k", png_dir, svg_dir)
    records.append({
        "figure_number": 16,
        "figure_id": "fig16_mean_latency_at_k",
        "data_contract_id": 16,
        "data_contract_name": "Mean Latency@K",
        "group": "D",
        "title": "Main Warm Benchmark Mean and P50 Retrieval Latency Across K=1..60",
        "protocol": "Protocol 1 — Main Warm Latency: 14,400 calls, 5 interleaved repeats, seed=42, candidate_top_k=120.",
        "metrics": ["mean_ms", "p50_ms"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/performance/latency_summary.csv",
            "evaluation/artifacts/stage04/performance/performance_run_manifest.json",
        ],
        "caption_vi": (
            "Hình 16: Độ trễ trung bình (Mean) và trung vị (P50) của bài đo Main Warm Latency (14.400 lượt gọi, 5 lượt lặp xen kẽ, pool=120). "
            "Độ trễ gần như phẳng theo K vì toàn bộ pool=120 được chấm điểm trước khi cắt Top-K; 115 lần tái tạo cache BM25 (TTL 300s) "
            "làm giá trị Mean nhỉnh hơn P50 ở các tổ hợp dùng BM25."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 17: Main Warm P95 Latency@K & Distribution Profile at K=5
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), gridspec_kw={"width_ratios": [1.3, 1.05]})
    fig.subplots_adjust(wspace=0.25, bottom=0.19, top=0.83)

    for cid in COMBO_ORDER:
        sub = lat[lat["combo_id"] == cid].sort_values("k")
        st = combo_style(cid)
        ax1.plot(
            x_idx, sub["p95_ms"].to_numpy(),
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], markersize=st["markersize"], label=COMBO_LABELS[cid],
        )
    ax1.set_xticks(x_idx, [str(k) for k in KS])
    ax1.set_xlabel("Cutoff K")
    ax1.set_ylabel("P95 Latency (ms)")
    ax1.set_ylim(0, 4450)
    ax1.set_title("(A) Main Warm P95 Latency@K (pool=120)")
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=7.3)

    # Panel B: P50 / Mean / P95 / P99 at K=5
    k5 = lat[lat["k"] == 5].set_index("combo_id")
    w = 0.20
    ax2.bar(cx - 1.5 * w, [k5.loc[c, "p50_ms"] for c in COMBO_ORDER], width=w, color="#0072B2", label="P50")
    ax2.bar(cx - 0.5 * w, [k5.loc[c, "mean_ms"] for c in COMBO_ORDER], width=w, color="#009E73", label="Mean")
    ax2.bar(cx + 0.5 * w, [k5.loc[c, "p95_ms"] for c in COMBO_ORDER], width=w, color="#D97706", label="P95")
    ax2.bar(cx + 1.5 * w, [k5.loc[c, "p99_ms"] for c in COMBO_ORDER], width=w, color="#B91C1C", label="P99")
    for i, c in enumerate(COMBO_ORDER):
        v = float(k5.loc[c, "p95_ms"])
        ax2.text(cx[i] + 0.5 * w, v + 80, f"{v:.0f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination (K=5, pool=120)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_ylim(0, 5950)
    ax2.set_title("(B) Latency Percentiles at K=5 (P50 / Mean / P95 / P99)")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.8)

    fig.suptitle("Figure 17: Main Warm P95 Latency@K and Tail-Latency Distribution (K=5)")
    add_protocol_footer(
        fig,
        "Protocol 1 (Main Warm Latency, pool=120, n=200/combo/K). Total warm P95 at K=5: A1=607.5 ms, B4=3,214.3 ms (-13.0% vs A4=3,693.1 ms). Retrieval only, NOT HTTP API.",
    )
    saved = save_figure(fig, "fig17_p95_latency_at_k", png_dir, svg_dir)
    records.append({
        "figure_number": 17,
        "figure_id": "fig17_p95_latency_at_k",
        "data_contract_id": 17,
        "data_contract_name": "P95 Latency@K",
        "group": "D",
        "title": "Main Warm P95 Latency@K and Tail-Latency Distribution (K=5)",
        "protocol": "Protocol 1 — Main Warm Latency: 14,400 calls, 5 interleaved repeats, candidate_top_k=120.",
        "metrics": ["p50_ms", "mean_ms", "p95_ms", "p99_ms"],
        "source_artifacts": ["evaluation/artifacts/stage04/performance/latency_summary.csv"],
        "caption_vi": (
            "Hình 17: Độ trễ P95 trên 9 mốc K và phân vị đuôi (P50/Mean/P95/P99) tại K=5 (pool=120). A1 nhanh nhất (P95 = 607,5 ms); "
            "ở tổng độ trễ warm retrieval, B4 nhanh hơn A4 khoảng 13,0% tại P95 K=5 (3.214,3 vs 3.693,1 ms) và 23,6% tại P50. "
            "Lưu ý: đây là độ trễ bộ truy xuất nội bộ, không phải độ trễ HTTP API đầu-cuối."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 18: Stage-Level Retrieval Latency Breakdown (640 calls, K=5 & K=60)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.14, bottom=0.19, top=0.82)

    stage_cols = [
        ("query_embed_mean_ms", "Query Embed", "#64748B"),
        ("bm25_search_mean_ms", "BM25 Search (incl. Qdrant/TTL)", "#0072B2"),
        ("qdrant_dense_search_mean_ms", "QdrantLocal Dense Search", "#D55E00"),
        ("rrf_fusion_mean_ms", "RRF Fusion", "#009E73"),
        ("rerank_mean_ms", "BGE Reranking (pool=120)", "#7B2CBF"),
    ]
    y_pos = np.arange(len(COMBO_ORDER))

    for ax, k_val, ttl in [
        (ax1, 5, "(A) Stage Breakdown at K=5 (pool=120, n=40/combo)"),
        (ax2, 60, "(B) Stage Breakdown at K=60 (pool=120, n=40/combo)"),
    ]:
        sub = stg[stg["k"] == k_val].set_index("combo_id")
        left = np.zeros(len(COMBO_ORDER))
        for col, lbl, clr in stage_cols:
            vals = np.array([float(sub.loc[c, col]) for c in COMBO_ORDER])
            ax.barh(y_pos, vals, left=left, height=0.58, color=clr, label=lbl)
            left += vals
        for i, c in enumerate(COMBO_ORDER):
            tot = float(sub.loc[c, "wall_clock_total_mean_ms"])
            ax.text(tot + 45, y_pos[i], f"{tot:,.0f} ms", va="center", ha="left", fontsize=8.0, fontweight="bold")
        ax.set_yticks(y_pos, [COMBO_LABELS[c] for c in COMBO_ORDER])
        ax.set_xlabel("Mean Stage Latency (ms)")
        ax.set_xlim(0, 4150)
        ax.set_title(ttl)
    ax1.invert_yaxis()

    # Callout clarifying ~41% isolated reranking speedup
    a4_rr = float(stg[(stg["combo_id"] == "A4") & (stg["k"] == 5)]["rerank_mean_ms"].iloc[0])
    b4_rr = float(stg[(stg["combo_id"] == "B4") & (stg["k"] == 5)]["rerank_mean_ms"].iloc[0])
    pct_rr = 100.0 * (1.0 - b4_rr / a4_rr)
    ax1.annotate(
        f"Isolated Rerank Stage:\nB4 ({b4_rr:,.0f} ms) is {pct_rr:.1f}% faster\nthan A4 ({a4_rr:,.0f} ms)",
        xy=(1950, 7), xytext=(2220, 4.9),
        fontsize=7.8, fontweight="bold", color="#4C1D95",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F3FF", edgecolor="#7B2CBF"),
        arrowprops=dict(arrowstyle="->", color="#7B2CBF", lw=1.2),
    )
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=5, fontsize=8.0)

    fig.suptitle("Figure 18: Stage-Level Retrieval Latency Breakdown (Isolated Reranking & QdrantLocal Overhead)", y=0.985)
    add_protocol_footer(
        fig,
        "Protocol 2 (Stage Breakdown): 640 calls (40 queries x 8 combos x {K=5, K=60}, pool=120). QdrantLocal SQLite accounts for ~461–645 ms per search leg; B3 K=5 includes a BM25 TTL rebuild.",
    )
    saved = save_figure(fig, "fig18_retrieval_latency_breakdown", png_dir, svg_dir)
    records.append({
        "figure_number": 18,
        "figure_id": "fig18_retrieval_latency_breakdown",
        "data_contract_id": 18,
        "data_contract_name": "Retrieval Latency Breakdown",
        "group": "D",
        "title": "Stage-Level Retrieval Latency Breakdown (Isolated Reranking & QdrantLocal Overhead)",
        "protocol": "Protocol 2 — Stage Breakdown: 640 calls (K=5 and K=60, candidate_top_k=120).",
        "metrics": ["query_embed_mean_ms", "bm25_search_mean_ms", "qdrant_dense_search_mean_ms", "rrf_fusion_mean_ms", "rerank_mean_ms", "wall_clock_total_mean_ms"],
        "source_artifacts": ["evaluation/artifacts/stage04/performance/stage_latency.csv"],
        "caption_vi": (
            "Hình 18: Phân rã độ trễ theo từng công đoạn (640 lượt đo, pool=120). Ở riêng giai đoạn BGE Reranking (K=5), B4 nhanh hơn A4 "
            "41,2% (1.364,8 ms vs 2.320,4 ms) nhờ chunk ngắn hơn; tuy nhiên chi phí QdrantLocal SQLite (~461–645 ms mỗi nhánh BM25/Dense) "
            "khiến mức giảm trên tổng độ trễ warm retrieval nhỏ hơn (~13–24%)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 19: Cold-Start (seconds) vs Warm (pool=120, ms) vs Production Profile (pool=10, ms)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1))
    fig.subplots_adjust(wspace=0.26, bottom=0.19, top=0.83)

    # Panel A: Cold-Start Breakdown in SECONDS (24 subprocess trials)
    cs_mean = cold.groupby("combo_id")[["model_load_ms", "bm25_build_ms", "first_retrieval_ms", "total_cold_start_ms"]].mean() / 1000.0
    ml = np.array([float(cs_mean.loc[c, "model_load_ms"]) for c in COMBO_ORDER])
    bm = np.array([float(cs_mean.loc[c, "bm25_build_ms"]) for c in COMBO_ORDER])
    fq = np.array([max(0.0, float(cs_mean.loc[c, "first_retrieval_ms"] - cs_mean.loc[c, "bm25_build_ms"])) for c in COMBO_ORDER])
    oh = np.array([max(0.0, float(cs_mean.loc[c, "total_cold_start_ms"] - cs_mean.loc[c, "model_load_ms"] - cs_mean.loc[c, "first_retrieval_ms"])) for c in COMBO_ORDER])

    ax1.bar(cx, ml, width=0.55, color="#7B2CBF", label="Model Load (s)")
    ax1.bar(cx, bm, width=0.55, bottom=ml, color="#0072B2", label="BM25 Index Build (s)")
    ax1.bar(cx, fq, width=0.55, bottom=ml + bm, color="#009E73", label="First Query Search (s)")
    ax1.bar(cx, oh, width=0.55, bottom=ml + bm + fq, color="#94A3B8", label="Subprocess / Import Overhead (s)")
    for i, c in enumerate(COMBO_ORDER):
        tot_s = float(cs_mean.loc[c, "total_cold_start_ms"])
        ax1.text(cx[i], tot_s + 0.65, f"{tot_s:.1f}s", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_xticks(cx, list(COMBO_ORDER))
    ax1.set_xlabel("Experimental Combination")
    ax1.set_ylabel("Cold-Start Time (Seconds)")
    ax1.set_ylim(0, 47.5)
    ax1.set_title("(A) Cold-Start Latency (Protocol 4: 24 Trials, Seconds)")
    ax1.legend(loc="upper left", ncol=2, fontsize=7.4)

    # Panel B: Warm K=5 (pool=120) vs Production-Parameter Profile K=5 (pool=10) in MILLISECONDS
    w = 0.27
    k5_warm = lat[lat["k"] == 5].set_index("combo_id")
    prod_idx = prod.set_index("combo_id")
    ax2.bar(cx - w, [k5_warm.loc[c, "p50_ms"] for c in COMBO_ORDER], width=w, color="#1F5A8A", label="Main Warm P50 (pool=120)")
    ax2.bar(cx, [k5_warm.loc[c, "p95_ms"] for c in COMBO_ORDER], width=w, color="#93C5FD", edgecolor="#1E3A8A", linewidth=0.6, label="Main Warm P95 (pool=120)")
    ax2.bar(cx + w, [prod_idx.loc[c, "p95_ms"] for c in COMBO_ORDER], width=w, color="#009E73", hatch="///", edgecolor="#064E3B", linewidth=0.6, label="Prod-Param Profile P95 (pool=10)")
    for i, c in enumerate(COMBO_ORDER):
        vp = float(prod_idx.loc[c, "p95_ms"])
        ax2.text(cx[i] + w, vp + 70, f"{vp:.0f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination (K=5)")
    ax2.set_ylabel("Warm Retrieval Latency (Milliseconds)")
    ax2.set_ylim(0, 4850)
    ax2.set_title("(B) Warm pool=120 vs Prod-Param Profile pool=10 (ms)")
    ax2.legend(loc="upper left", fontsize=7.6)

    fig.suptitle("Figure 19: Cold-Start Latency (Seconds) vs Warm & Production-Parameter Retrieval Latency (ms)")
    add_protocol_footer(
        fig,
        "Strict Protocol Separation — Panel A: Cold-Start (24 trials, OS cache not flushed, seconds). Panel B: Main Warm (14,400 calls, pool=120) vs Prod-Param Eval Index (1,600 calls, K=5/pool=10, NOT HTTP API).",
    )
    saved = save_figure(fig, "fig19_cold_vs_warm_and_prod_profile_latency", png_dir, svg_dir)
    records.append({
        "figure_number": 19,
        "figure_id": "fig19_cold_vs_warm_and_prod_profile_latency",
        "data_contract_id": 19,
        "data_contract_name": "Cold vs Warm Latency",
        "group": "D",
        "title": "Cold-Start Latency (Seconds) vs Warm & Production-Parameter Retrieval Latency (ms)",
        "protocol": "Protocol 4 (Cold-Start, 24 trials, seconds) strictly separated from Protocol 1 (Warm pool=120, ms) & Protocol 3 (Prod-Param K=5/pool=10, 1,600 calls, ms).",
        "metrics": ["model_load_ms", "bm25_build_ms", "first_retrieval_ms", "total_cold_start_ms", "p50_ms", "p95_ms"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/performance/cold_start_latency.csv",
            "evaluation/artifacts/stage04/performance/latency_summary.csv",
            "evaluation/artifacts/stage04/performance/production_profile_latency.csv",
        ],
        "caption_vi": (
            "Hình 19: Tách biệt rõ đơn vị và giao thức đo: (A) Độ trễ khởi động nguội (Cold-Start, 24 lần chạy tiến trình độc lập, đơn vị giây) "
            "— xây dựng BM25 Profile B tốn 15,2–16,0s so với 9,6–10,9s của Profile A; (B) Đối chiếu Main Warm (pool=120, ms) với cấu hình "
            "tham số production trên evaluation index (K=5, pool=10, 1.600 lượt gọi: A4 P95 giảm từ 3.693 ms xuống 1.231 ms)."
        ),
        **saved,
    })

    # ------------------------------------------------------------------
    # Fig 20: Resource Usage (GPU VRAM & Process RAM / RSS)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.subplots_adjust(wspace=0.27, bottom=0.19, top=0.83)

    ridx = res.set_index("combo_id")
    w = 0.36
    gpu_alloc = [float(ridx.loc[c, "peak_gpu_allocated_max_mb"]) for c in COMBO_ORDER]
    gpu_resv = [float(ridx.loc[c, "peak_gpu_reserved_max_mb"]) for c in COMBO_ORDER]

    b1 = ax1.bar(cx - w / 2, gpu_alloc, width=w, color="#7B2CBF", label="Peak GPU Allocated (MB)")
    b2 = ax1.bar(cx + w / 2, gpu_resv, width=w, color="#C4B5FD", edgecolor="#5B21B6", linewidth=0.6, label="Peak GPU Reserved (MB)")
    for i in range(len(COMBO_ORDER)):
        ax1.text(cx[i] - w / 2, gpu_alloc[i] + 55, f"{gpu_alloc[i]:.0f}", ha="center", va="bottom", fontsize=7.4)
    ax1.axhline(2686.82, color="#B91C1C", linestyle="--", linewidth=1.1, label="Warm Run Peak Alloc (2,687 MB)")
    ax1.set_xticks(cx, list(COMBO_ORDER))
    ax1.set_xlabel("Experimental Combination")
    ax1.set_ylabel("GPU VRAM (MB)")
    ax1.set_ylim(0, 3850)
    ax1.set_title("(A) Peak GPU Memory Usage (RTX 5060 8 GB, 0 OOM)")
    ax1.legend(loc="upper left", fontsize=7.6)

    # Panel B: Cold-Start Isolated Process RSS + Warm Benchmark RSS Band
    rss_vals = [float(ridx.loc[c, "process_rss_mean_mb"]) for c in COMBO_ORDER]
    bars = ax2.bar(cx, rss_vals, width=0.56, color=[combo_style(c)["color"] for c in COMBO_ORDER],
                   edgecolor="#1F2937", linewidth=0.6, label="Isolated Cold-Start Process RSS (Mean MB)")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 45, f"{h:,.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    snaps = pman["warm_protocol"]["resource_snapshots"]
    warm_rss_min = min(s["process_rss_mb"] for s in snaps)
    warm_rss_max = max(s["process_rss_mb"] for s in snaps)
    ax2.axhspan(warm_rss_min, warm_rss_max, color="#009E73", alpha=0.22, label=f"Warm Shared-Process RSS Band ({warm_rss_min:.0f}–{warm_rss_max:.0f} MB)")
    ax2.set_xticks(cx, list(COMBO_ORDER))
    ax2.set_xlabel("Experimental Combination")
    ax2.set_ylabel("Process RAM / RSS (MB)")
    ax2.set_ylim(0, 3450)
    ax2.set_title("(B) Host RAM Footprint (Cold-Start Peaks vs Warm Band)")
    ax2.legend(loc="upper left", fontsize=7.6)

    fig.suptitle("Figure 20: GPU VRAM and Host Process RAM Resource Utilization")
    add_protocol_footer(
        fig,
        "Source: resource_usage.csv & performance_run_manifest.json. Zero CUDA OOM across all 16,664 measured calls; PyTorch max_memory_allocated excludes CUDA driver context.",
    )
    saved = save_figure(fig, "fig20_resource_usage", png_dir, svg_dir)
    records.append({
        "figure_number": 20,
        "figure_id": "fig20_resource_usage",
        "data_contract_id": 20,
        "data_contract_name": "Resource Usage",
        "group": "D",
        "title": "GPU VRAM and Host Process RAM Resource Utilization",
        "protocol": "Cold-start isolated process peaks (3 trials/combo) alongside warm benchmark 5-repeat snapshots on RTX 5060 (7.96 GB VRAM) / 15.71 GB RAM.",
        "metrics": ["peak_gpu_allocated_max_mb", "peak_gpu_reserved_max_mb", "process_rss_mean_mb", "cuda_oom_count"],
        "source_artifacts": [
            "evaluation/artifacts/stage04/performance/resource_usage.csv",
            "evaluation/artifacts/stage04/performance/performance_run_manifest.json",
        ],
        "caption_vi": (
            "Hình 20: Mức tiêu thụ GPU VRAM và RAM tiến trình (RSS). BM25 thuần (A1, B1) tiêu thụ 0 MB VRAM; Dense/RRF dùng 458,5 MB VRAM; "
            "tổ hợp Reranker (A4, B4) đạt đỉnh ~2,65–2,69 GB VRAM (chỉ ~34% dung lượng RTX 5060 8 GB, 0 lỗi OOM). RAM tiến trình trong phiên "
            "warm duy trì ổn định 965–993 MB qua cả 5 lượt lặp."
        ),
        **saved,
    })

    return records
