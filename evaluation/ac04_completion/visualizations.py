"""Generate the 6 required supplementary AC-04 figures (300 DPI PNG + SVG) and run validation."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

from evaluation.ac04_completion import (
    PROTOCOL_HTTP_CONFIG_A,
    PROTOCOL_HTTP_CONFIG_B,
    PROTOCOL_LIVE_GROQ_NO_RAG,
    artifacts_dir,
)
from evaluation.stage05.data_loader import verify_inputs_and_integrity
from evaluation.stage05.style import add_protocol_footer, configure_academic_style, save_figure, sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fval(val: Any) -> float:
    if val is None or str(val).strip() in ("", "N/A", "nan", "None"):
        return float("nan")
    return float(val)


def generate_ac04_figures(root: Path) -> list[dict[str, Any]]:
    """Generate the 6 supplementary figures in Stage 05 academic style (Arial, 300 DPI PNG + SVG)."""
    style_meta = configure_academic_style()
    out_dir = artifacts_dir(root)
    png_dir = out_dir / "figures" / "png"
    svg_dir = out_dir / "figures" / "svg"

    ans_df = pd.read_csv(out_dir / "answer_quality_summary.csv", keep_default_na=False)
    lat_df = pd.read_csv(out_dir / "http_latency_summary.csv")
    rob_df = pd.read_csv(out_dir / "robustness_summary_live.csv", keep_default_na=False)
    study_df = pd.read_csv(out_dir / "study_features_summary_live.csv", keep_default_na=False)
    s4_ans = pd.read_csv(root / "evaluation/artifacts/stage04/generation/answer_metrics_summary.csv").set_index("combo_id")
    s4_lat = pd.read_csv(root / "evaluation/artifacts/stage04/performance/latency_summary.csv")
    s4_prod = pd.read_csv(root / "evaluation/artifacts/stage04/performance/production_profile_latency.csv").set_index("combo_id")
    s4_study = pd.read_csv(root / "evaluation/artifacts/stage04/study_features/study_features_summary.csv").set_index("feature")
    s4_ft = json.loads((root / "evaluation/artifacts/stage04/study_features/functional_test_results.json").read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Sup Fig 01: Live Groq Answer Quality (All 10 Metrics + Mode Breakdown)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={"width_ratios": [1.45, 0.95]})
    fig.subplots_adjust(wspace=0.26, bottom=0.20, top=0.82)

    row_specs = [
        (PROTOCOL_HTTP_CONFIG_A, "all_40_queries", "HTTP Config A (pool=10, All 40)"),
        (PROTOCOL_HTTP_CONFIG_A, "live_llm_only", "HTTP Config A (Live LLM Only)"),
        (PROTOCOL_HTTP_CONFIG_B, "all_40_queries", "HTTP Config B (pool=30, All 40)"),
        (PROTOCOL_HTTP_CONFIG_B, "live_llm_only", "HTTP Config B (Live LLM Only)"),
        (PROTOCOL_LIVE_GROQ_NO_RAG, "all_40_queries", "Direct Live Groq NO_RAG (All 40)"),
    ]
    cols_10 = [
        ("answer_correctness_norm", "Correct."),
        ("answer_relevancy_norm", "Relev."),
        ("faithfulness_norm", "Faith."),
        ("answer_completeness_norm", "Complet."),
        ("citation_accuracy_norm", "Cit.Acc."),
        ("citation_validity", "Cit.Val."),
        ("unsupported_claim_rate", "Unsup.↓"),
        ("contradiction_rate", "Contra.↓"),
        ("complexity_correctness_norm", "Complex."),
        ("code_correctness_norm", "Code"),
    ]
    mat = np.full((len(row_specs), len(cols_10)), np.nan)
    y_labels = []
    for r_i, (proto, sub_k, lbl) in enumerate(row_specs):
        m = ans_df[(ans_df["protocol"] == proto) & (ans_df["subset"] == sub_k)]
        n_val = int(m["n_total"].iloc[0]) if len(m) else 0
        y_labels.append(f"{lbl}\n(n={n_val})")
        if len(m):
            for c_i, (col, _) in enumerate(cols_10):
                mat[r_i, c_i] = _fval(m[col].iloc[0])

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax1.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isnan(v):
                ax1.text(c, r, "N/A", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#4B5563")
            else:
                clr = "#FFFFFF" if v >= 0.65 else "#111827"
                ax1.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7.6, fontweight="bold", color=clr)
    ax1.set_xticks(range(len(cols_10)), [lbl for _, lbl in cols_10], rotation=24, ha="right")
    ax1.set_yticks(range(len(row_specs)), y_labels, fontsize=8.0)
    ax1.axhline(3.5, color="#111827", linewidth=1.5)
    ax1.set_title("(A) Live Groq (openai/gpt-oss-120b) 10-Metric Quality Matrix")
    fig.colorbar(im, ax=ax1, fraction=0.032, pad=0.02, label="Score / Rate (0–1)")

    # Panel B: Generation Outcome Breakdown (Live LLM vs Extractive Fallback vs Abstained)
    protos_b = [
        (PROTOCOL_HTTP_CONFIG_A, "HTTP Config A\n(pool=10)"),
        (PROTOCOL_HTTP_CONFIG_B, "HTTP Config B\n(pool=30)"),
        (PROTOCOL_LIVE_GROQ_NO_RAG, "Direct Groq\nNO_RAG"),
    ]
    bx = np.arange(len(protos_b))
    llm_cnts, ext_cnts, abs_cnts = [], [], []
    for p, _ in protos_b:
        r = ans_df[(ans_df["protocol"] == p) & (ans_df["subset"] == "all_40_queries")].iloc[0]
        llm_cnts.append(int(r["n_live_llm"]))
        ext_cnts.append(int(r["n_extractive"]))
        abs_cnts.append(int(r["n_abstained"]))
    llm_arr, ext_arr, abs_arr = np.array(llm_cnts), np.array(ext_cnts), np.array(abs_cnts)
    ax2.bar(bx, llm_arr, width=0.52, color="#009E73", label="Live LLM Generated (mode=llm)")
    ax2.bar(bx, ext_arr, width=0.52, bottom=llm_arr, color="#D97706", hatch="///", edgecolor="#78350F", label="Extractive Fallback (mode=extractive)")
    ax2.bar(bx, abs_arr, width=0.52, bottom=llm_arr + ext_arr, color="#94A3B8", label="Abstained Before/By LLM")
    for i in range(len(bx)):
        if llm_arr[i] > 0:
            ax2.text(bx[i], llm_arr[i] / 2, str(llm_arr[i]), ha="center", va="center", color="#FFFFFF", fontweight="bold", fontsize=8.8)
        if ext_arr[i] > 0:
            ax2.text(bx[i], llm_arr[i] + ext_arr[i] / 2, str(ext_arr[i]), ha="center", va="center", color="#111827", fontweight="bold", fontsize=8.8)
        if abs_arr[i] > 0:
            ax2.text(bx[i], llm_arr[i] + ext_arr[i] + abs_arr[i] / 2, str(abs_arr[i]), ha="center", va="center", color="#111827", fontweight="bold", fontsize=8.8)
    ax2.set_xticks(bx, [lbl for _, lbl in protos_b])
    ax2.set_ylabel("Number of Queries (Total n = 40)")
    ax2.set_ylim(0, 50)
    ax2.set_title("(B) Application Generation Mode Breakdown")
    ax2.legend(loc="upper center", fontsize=7.6)

    fig.suptitle("Supplementary Figure 1: Live Groq Answer Quality and Application Generation Modes")
    add_protocol_footer(
        fig,
        "Source: answer_quality_summary.csv (openai/gpt-oss-120b, isolated eval Qdrant copy, verbatim provider context, blind pro judge, human_validated=false). N/A preserved where no context supplied.",
    )
    saved = save_figure(fig, "sup_fig01_live_groq_answer_quality", png_dir, svg_dir)
    records.append({
        "figure_number": 1,
        "figure_id": "sup_fig01_live_groq_answer_quality",
        "title": "Live Groq Answer Quality and Application Generation Modes",
        "protocol": "Isolated HTTP Config A (pool=10, top_k=5), HTTP Config B (pool=30, top_k=5), Direct Live Groq NO_RAG (openai/gpt-oss-120b).",
        "source_artifacts": ["evaluation/artifacts/ac04_completion/answer_quality_summary.csv"],
        **saved,
    })

    # ------------------------------------------------------------------
    # Sup Fig 02: Controlled Agent (Stage 04) vs Live Groq (Explicit Protocol Separation)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13.2, 5.3))
    fig.subplots_adjust(bottom=0.22, top=0.81)

    comp_metrics = [
        ("answer_correctness_norm", "Correctness"),
        ("answer_relevancy_norm", "Relevancy"),
        ("faithfulness_norm", "Faithfulness"),
        ("answer_completeness_norm", "Completeness"),
        ("citation_accuracy_norm", "Citation Acc."),
    ]
    series = [
        ("Stage 04 Controlled Agent A4\n(flash, budget=2048, n=40)", [float(s4_ans.loc["A4", c]) for c, _ in comp_metrics], "#7B2CBF", ""),
        ("Stage 04 Controlled Agent NO_RAG\n(flash, n=40)", [float(s4_ans.loc["NO_RAG", c]) if pd.notna(s4_ans.loc["NO_RAG", c]) else np.nan for c, _ in comp_metrics], "#64748B", "xx"),
        ("Live Groq HTTP Config A\n(pool=10, All 40, gpt-oss-120b)", [_fval(ans_df[(ans_df["protocol"] == PROTOCOL_HTTP_CONFIG_A) & (ans_df["subset"] == "all_40_queries")][c].iloc[0]) for c, _ in comp_metrics], "#0072B2", ""),
        ("Live Groq HTTP Config B\n(pool=30, All 40, gpt-oss-120b)", [_fval(ans_df[(ans_df["protocol"] == PROTOCOL_HTTP_CONFIG_B) & (ans_df["subset"] == "all_40_queries")][c].iloc[0]) for c, _ in comp_metrics], "#009E73", "///"),
        ("Live Groq HTTP Config B\n(Live LLM Subset Only)", [_fval(ans_df[(ans_df["protocol"] == PROTOCOL_HTTP_CONFIG_B) & (ans_df["subset"] == "live_llm_only")][c].iloc[0]) for c, _ in comp_metrics], "#10B981", ""),
        ("Direct Live Groq NO_RAG\n(gpt-oss-120b, n=40)", [_fval(ans_df[(ans_df["protocol"] == PROTOCOL_LIVE_GROQ_NO_RAG) & (ans_df["subset"] == "all_40_queries")][c].iloc[0]) for c, _ in comp_metrics], "#D55E00", "///"),
    ]
    mx = np.arange(len(comp_metrics))
    w = 0.135
    for idx, (lbl, vals, clr, ht) in enumerate(series):
        offs = mx + (idx - (len(series) - 1) / 2.0) * w
        for xi, v in zip(offs, vals):
            if np.isnan(v):
                ax.text(xi, 0.03, "N/A", ha="center", va="bottom", fontsize=6.4, rotation=90, color="#4B5563", fontweight="bold")
            else:
                ax.bar(xi, v, width=w, color=clr, hatch=ht, edgecolor="#1F2937", linewidth=0.45)
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.bar([], [], color=clr, hatch=ht, edgecolor="#1F2937", label=lbl)

    patches = [Patch(facecolor=clr, hatch=ht, edgecolor="#1F2937", label=lbl) for lbl, _, clr, ht in series]
    ax.set_xticks(mx, [lbl for _, lbl in comp_metrics])
    ax.set_ylabel("Normalized Score (0–1)")
    ax.set_ylim(0.0, 1.42)
    ax.legend(handles=patches, loc="upper center", ncol=3, fontsize=7.6)
    ax.set_title("Protocol-Separated Comparison: Stage 04 Controlled Agent vs Live Groq HTTP E2E & Direct NO_RAG")
    fig.suptitle("Supplementary Figure 2: Controlled Agent vs Live Groq Quality (Distinct Protocols — Do Not Merge)")
    add_protocol_footer(
        fig,
        "Distinct Protocols: Stage 04 Controlled Agent used direct packed context (2048 tok, flash); Live HTTP uses unmodified FastAPI intent_router + chat_retrieval + MAX_EVIDENCES=5 + openai/gpt-oss-120b.",
    )
    saved = save_figure(fig, "sup_fig02_controlled_agent_vs_live_groq", png_dir, svg_dir)
    records.append({
        "figure_number": 2,
        "figure_id": "sup_fig02_controlled_agent_vs_live_groq",
        "title": "Controlled Agent vs Live Groq Quality (Distinct Protocols)",
        "protocol": "Side-by-side protocol comparison keeping Stage 04 Controlled Agent, Live HTTP Config A/B, and Live Groq NO_RAG strictly separated.",
        "source_artifacts": [
            "evaluation/artifacts/stage04/generation/answer_metrics_summary.csv",
            "evaluation/artifacts/ac04_completion/answer_quality_summary.csv",
        ],
        **saved,
    })

    # ------------------------------------------------------------------
    # Sup Fig 03: Client-Observed HTTP End-to-End Latency
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0), gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.subplots_adjust(wspace=0.28, bottom=0.20, top=0.82)

    lidx = lat_df.set_index("experiment_group")
    chat_groups = [
        ("HTTP_Config_A_pool10", "HTTP /api/chat\nConfig A (p=10)\n(n=40)"),
        ("HTTP_Config_B_pool30", "HTTP /api/chat\nConfig B (p=30)\n(n=40)"),
        ("Direct_Live_Groq_NO_RAG", "Direct Groq\nNO_RAG\n(n=40)"),
        ("HTTP_Robustness_20", "HTTP /api/chat\nRobustness\n(n=20)"),
    ]
    gx = np.arange(len(chat_groups))
    w = 0.24
    for ax, groups_list, ttl, ymax_pad in [
        (ax1, chat_groups, "(A) Chat & Robustness HTTP Latency (ms)", 1.24),
        (ax2, [("HTTP_Quiz_8_Topics", "HTTP /api/quiz\n(n=8 topics)"), ("HTTP_Roadmap_4_Scenarios", "HTTP /api/roadmap\n(n=4 scenarios)")], "(B) Study Feature HTTP Latency (ms)", 1.28),
    ]:
        xs = np.arange(len(groups_list))
        p50s = [float(lidx.loc[k, "p50_ms"]) for k, _ in groups_list]
        means = [float(lidx.loc[k, "mean_ms"]) for k, _ in groups_list]
        p95s = [float(lidx.loc[k, "p95_ms"]) for k, _ in groups_list]
        b1 = ax.bar(xs - w, p50s, width=w, color="#0072B2", label="P50")
        b2 = ax.bar(xs, means, width=w, color="#009E73", label="Mean")
        b3 = ax.bar(xs + w, p95s, width=w, color="#D55E00", label="P95")
        for bar in list(b1) + list(b2) + list(b3):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(p95s) * 0.02, f"{h:,.0f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
        ax.set_xticks(xs, [lbl for _, lbl in groups_list])
        ax.set_ylabel("Client-Observed HTTP Latency (ms)")
        ax.set_ylim(0, max(p95s) * ymax_pad + 500)
        ax.set_title(ttl)
        ax.legend(loc="upper left", ncol=3, fontsize=7.8)

    fig.suptitle("Supplementary Figure 3: Client-Observed HTTP End-to-End Response Latency")
    add_protocol_footer(
        fig,
        "Source: http_latency_summary.csv. Complete HTTP response times measured from client against isolated FastAPI server (NOT Stage 04 retrieval-only latency); includes initial BM25 index build on Q001.",
    )
    saved = save_figure(fig, "sup_fig03_http_e2e_latency", png_dir, svg_dir)
    records.append({
        "figure_number": 3,
        "figure_id": "sup_fig03_http_e2e_latency",
        "title": "Client-Observed HTTP End-to-End Response Latency",
        "protocol": "Complete client-observed HTTP latency across /api/chat (Config A & B), /api/quiz, /api/roadmap, Robustness, and Direct Groq NO_RAG.",
        "source_artifacts": ["evaluation/artifacts/ac04_completion/http_latency_summary.csv"],
        **saved,
    })

    # ------------------------------------------------------------------
    # Sup Fig 04: Retrieval vs Generation vs Total HTTP Response Time Decomposition
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    fig.subplots_adjust(bottom=0.20, top=0.82)

    # Include Stage 04 retrieval-only reference bars clearly separated by a divider line!
    s4_a4_p10 = float(s4_prod.loc["A4", "mean_ms"])
    s4_a4_p120 = float(s4_lat[(s4_lat["combo_id"] == "A4") & (s4_lat["k"] == 5)]["mean_ms"].iloc[0])

    decomp_specs = [
        ("Stage 04 Retrieval-Only\nA4 (K=5, pool=10, 1.6k calls)", s4_a4_p10, 0.0, 0.0),
        ("Stage 04 Retrieval-Only\nA4 (K=5, pool=120, 14.4k calls)", s4_a4_p120, 0.0, 0.0),
        ("Live HTTP /api/chat\nConfig A (pool=10, n=40)", float(lidx.loc["HTTP_Config_A_pool10", "retrieval_mean_ms"]), float(lidx.loc["HTTP_Config_A_pool10", "llm_generation_mean_ms"]), float(lidx.loc["HTTP_Config_A_pool10", "http_overhead_mean_ms"])),
        ("Live HTTP /api/chat\nConfig B (pool=30, n=40)", float(lidx.loc["HTTP_Config_B_pool30", "retrieval_mean_ms"]), float(lidx.loc["HTTP_Config_B_pool30", "llm_generation_mean_ms"]), float(lidx.loc["HTTP_Config_B_pool30", "http_overhead_mean_ms"])),
        ("Direct Live Groq NO_RAG\n(n=40, No Retrieval)", 0.0, float(lidx.loc["Direct_Live_Groq_NO_RAG", "llm_generation_mean_ms"]), 0.0),
        ("Live HTTP /api/quiz\n(n=8 topics)", float(lidx.loc["HTTP_Quiz_8_Topics", "retrieval_mean_ms"]), float(lidx.loc["HTTP_Quiz_8_Topics", "llm_generation_mean_ms"]), float(lidx.loc["HTTP_Quiz_8_Topics", "http_overhead_mean_ms"])),
        ("Live HTTP /api/roadmap\n(n=4 scenarios)", float(lidx.loc["HTTP_Roadmap_4_Scenarios", "retrieval_mean_ms"]), float(lidx.loc["HTTP_Roadmap_4_Scenarios", "llm_generation_mean_ms"]), float(lidx.loc["HTTP_Roadmap_4_Scenarios", "http_overhead_mean_ms"])),
    ]
    ys = np.arange(len(decomp_specs))
    rets = np.array([x[1] for x in decomp_specs])
    llms = np.array([x[2] for x in decomp_specs])
    ovhs = np.array([x[3] for x in decomp_specs])
    tots = rets + llms + ovhs

    ax.barh(ys, rets, height=0.56, color="#0072B2", label="Retrieval / Inventory Stage (ms)")
    ax.barh(ys, llms, left=rets, height=0.56, color="#7B2CBF", label="Live Groq LLM Generation Stage (ms)")
    ax.barh(ys, ovhs, left=rets + llms, height=0.56, color="#94A3B8", label="HTTP / Middleware Overhead (ms)")
    ax.axhline(1.5, color="#B91C1C", linestyle="--", linewidth=1.3)
    for i, tot in enumerate(tots):
        ax.text(tot + max(tots) * 0.015, ys[i], f"{tot:,.0f} ms", va="center", ha="left", fontsize=8.2, fontweight="bold")
    ax.set_yticks(ys, [x[0] for x in decomp_specs], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Time (Milliseconds)")
    ax.set_xlim(0, max(tots) * 1.18)
    ax.legend(loc="lower right", fontsize=8.2)
    ax.set_title("Retrieval vs Live Groq Generation vs Total HTTP Response Time (Separated from Stage 04 Retrieval-Only)")
    fig.suptitle("Supplementary Figure 4: End-to-End Latency Decomposition (ObservabilityMiddleware Telemetry)")
    add_protocol_footer(
        fig,
        "Source: http_latency_summary.csv & Stage 04 latency CSVs. Dashed red line strictly separates Stage 04 retrieval-only benchmarks from complete Live HTTP API measurements.",
    )
    saved = save_figure(fig, "sup_fig04_retrieval_vs_generation_vs_total_time", png_dir, svg_dir)
    records.append({
        "figure_number": 4,
        "figure_id": "sup_fig04_retrieval_vs_generation_vs_total_time",
        "title": "End-to-End Latency Decomposition (Retrieval vs Generation vs Total)",
        "protocol": "Exact ObservabilityMiddleware stage decomposition (retrieval_ms, llm_ms, total HTTP ms) contrasted cleanly with Stage 04 retrieval-only references.",
        "source_artifacts": [
            "evaluation/artifacts/ac04_completion/http_latency_summary.csv",
            "evaluation/artifacts/stage04/performance/production_profile_latency.csv",
            "evaluation/artifacts/stage04/performance/latency_summary.csv",
        ],
        **saved,
    })

    # ------------------------------------------------------------------
    # Sup Fig 05: Live HTTP Quiz & Roadmap Evaluation
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1))
    fig.subplots_adjust(wspace=0.26, bottom=0.20, top=0.81)

    sidx = study_df.set_index("feature")
    m_cols = [
        ("schema_valid_rate", "Schema\nValid"),
        ("content_correctness_norm", "Content\nCorrect."),
        ("source_grounding_norm", "Source\nGrounding"),
        ("pedagogical_quality_norm", "Pedagog.\nQuality"),
    ]
    sx = np.arange(len(m_cols))
    w = 0.20
    q_live = [_fval(sidx.loc["live_http_quiz", c]) for c, _ in m_cols]
    r_live = [_fval(sidx.loc["live_http_roadmap", c]) for c, _ in m_cols]
    q_s4 = [float(s4_study.loc["quiz_generation_controlled_agent", c]) for c, _ in m_cols]
    r_s4 = [float(s4_study.loc["study_roadmap_controlled_agent", c]) for c, _ in m_cols]

    for idx, (vals, clr, ht, lbl) in enumerate([
        (q_live, "#0072B2", "", "Live HTTP Quiz (n=8)"),
        (q_s4, "#93C5FD", "///", "Stage 04 Agent Quiz (n=8)"),
        (r_live, "#D55E00", "", "Live HTTP Roadmap (n=4)"),
        (r_s4, "#FDBA74", "///", "Stage 04 Agent Roadmap (n=4)"),
    ]):
        offs = sx + (idx - 1.5) * w
        for xi, v in zip(offs, vals):
            if np.isnan(v):
                ax1.text(xi, 0.03, "N/A", ha="center", va="bottom", fontsize=6.5, rotation=90, color="#4B5563")
            else:
                ax1.bar(xi, v, width=w, color=clr, hatch=ht, edgecolor="#1F2937", linewidth=0.45)
                ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
    patches_s = [
        Patch(facecolor="#0072B2", edgecolor="#1F2937", label="Live HTTP Quiz (n=8)"),
        Patch(facecolor="#93C5FD", hatch="///", edgecolor="#1F2937", label="Stage 04 Agent Quiz (n=8)"),
        Patch(facecolor="#D55E00", edgecolor="#1F2937", label="Live HTTP Roadmap (n=4)"),
        Patch(facecolor="#FDBA74", hatch="///", edgecolor="#1F2937", label="Stage 04 Agent Roadmap (n=4)"),
    ]
    ax1.set_xticks(sx, [lbl for _, lbl in m_cols])
    ax1.set_ylabel("Normalized Score (0–1)")
    ax1.set_ylim(0, 1.38)
    ax1.legend(handles=patches_s, loc="upper center", ncol=2, fontsize=7.4)
    ax1.set_title("(A) Live HTTP vs Stage 04 Controlled Agent Study Quality")

    # Panel B: Live Quiz & Roadmap Functional Success + Raw QZ_03 422 + Stage 04 Contract Checks
    cats_b = ["Live HTTP Quiz\n(n=8 normalized)", "Live HTTP Roadmap\n(n=4 scenarios)"]
    bx2 = np.arange(2)
    http200_r = [float(sidx.loc["live_http_quiz", "http_200_count"]) / 8.0 * 100, float(sidx.loc["live_http_roadmap", "http_200_count"]) / 4.0 * 100]
    any_r = [float(sidx.loc["live_http_quiz", "any_live_llm_output_count"]) / 8.0 * 100, float(sidx.loc["live_http_roadmap", "any_live_llm_output_count"]) / 4.0 * 100]
    func_r = [float(sidx.loc["live_http_quiz", "functional_live_llm_success_count"]) / 8.0 * 100, float(sidx.loc["live_http_roadmap", "functional_live_llm_success_count"]) / 4.0 * 100]
    ax2.bar(bx2 - 0.24, http200_r, width=0.22, color="#009E73", label="HTTP 200 Transport (%)")
    ax2.bar(bx2, any_r, width=0.22, color="#0072B2", label="Any Live LLM Output (incl. partial, %)")
    ax2.bar(bx2 + 0.24, func_r, width=0.22, color="#7B2CBF", label="Full Error-Free Live Output (%)")
    for i in range(2):
        ax2.text(bx2[i] - 0.24, http200_r[i] + 2, f"{http200_r[i]:.0f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax2.text(bx2[i], any_r[i] + 2, f"{any_r[i]:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax2.text(bx2[i] + 0.24, func_r[i] + 2, f"{func_r[i]:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(bx2, cats_b)
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 175)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=1, fontsize=7.3)
    ax2.text(
        0.5, 0.98,
        f"Preserved Stage 04 Backend Contracts: {s4_ft['contract_checks_passed']}/{s4_ft['contract_checks_total']} PASSED\n"
        "Observed Raw Validation: QZ_03 ('Tree, BST & AVL') -> HTTP 422 (comma rule)",
        transform=ax2.transAxes, ha="center", va="top", fontsize=7.3, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5", edgecolor="#10B981"),
    )
    ax2.set_title("(B) Transport vs Functional Success & Contract Checks")

    fig.suptitle("Supplementary Figure 5: Live HTTP Quiz & Study Roadmap Evaluation")
    add_protocol_footer(
        fig,
        "Source: study_features_summary_live.csv & Stage 04 study summaries. Real /api/quiz & /api/roadmap via openai/gpt-oss-120b kept distinct from 11/11 contract checks and Controlled Agent.",
    )
    saved = save_figure(fig, "sup_fig05_live_quiz_and_roadmap", png_dir, svg_dir)
    records.append({
        "figure_number": 5,
        "figure_id": "sup_fig05_live_quiz_and_roadmap",
        "title": "Live HTTP Quiz & Study Roadmap Evaluation",
        "protocol": "Real HTTP /api/quiz (8 topics + raw QZ_03 422 validation) and /api/roadmap (4 scenarios) under isolated Config B (pool=30).",
        "source_artifacts": [
            "evaluation/artifacts/ac04_completion/study_features_summary_live.csv",
            "evaluation/artifacts/stage04/study_features/study_features_summary.csv",
        ],
        **saved,
    })

    # ------------------------------------------------------------------
    # Sup Fig 06: Live API Robustness (n=20) & HTTP Reliability Matrix
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.subplots_adjust(wspace=0.28, bottom=0.20, top=0.82)

    cat_specs = [
        ("out_of_domain", "Out-of-Domain"),
        ("unanswerable_in_corpus", "Unanswerable"),
        ("prompt_injection", "Query-Level\nPrompt Inj.*"),
        ("distractor_context", "Distractor Query"),
        ("semantic_paraphrase", "Semantic\nParaphrase"),
        ("typo_noisy", "VI Unaccented\n/ Typo"),
    ]
    ridx = rob_df.set_index("category")
    ry = np.arange(len(cat_specs))
    ch = [float(ridx.loc[c, "correct_handling_rate"]) * 100.0 for c, _ in cat_specs]
    aq = [_fval(ridx.loc[c, "mean_answer_quality_norm"]) * 100.0 for c, _ in cat_specs]
    ns = [int(ridx.loc[c, "n_cases"]) for c, _ in cat_specs]
    ax1.barh(ry - 0.18, ch, height=0.34, color="#0072B2", label="Correct Handling Rate (%)")
    ax1.barh(ry + 0.18, [0.0 if np.isnan(v) else v for v in aq], height=0.34, color="#009E73", hatch="///", edgecolor="#064E3B", label="Answer Quality Score (%)")
    for i in range(len(cat_specs)):
        qv = f"{aq[i]:.1f}%" if not np.isnan(aq[i]) else "N/A"
        ax1.text(max(ch[i], 0.0 if np.isnan(aq[i]) else aq[i]) + 2.0, ry[i], f"{ch[i]:.1f}% / {qv}", va="center", ha="left", fontsize=7.6)
    ax1.set_yticks(ry, [f"{lbl} (n={n})" for (_, lbl), n in zip(cat_specs, ns)])
    ax1.set_ylim(6.0, -0.55)
    ax1.set_xlim(0, 138)
    ax1.set_xlabel("Percentage / Normalized Score (%)")
    ax1.legend(loc="center right", fontsize=7.6)
    ax1.text(0.01, 0.02, "*Query-level HTTP injection (distinct from Stage 04 context-level); n=3 != full security.", transform=ax1.transAxes, fontsize=7.0, style="italic", color="#991B1B")
    ax1.set_title("(A) Live HTTP Robustness Results (n=20 Cases, Config B)")

    # Panel B: HTTP Transport vs Live LLM vs Error/Timeout Reliability Summary
    rel_groups = [
        ("HTTP_Config_A_pool10", "Chat A\n(p=10)"),
        ("HTTP_Config_B_pool30", "Chat B\n(p=30)"),
        ("HTTP_Robustness_20", "Robust.\n(n=20)"),
        ("HTTP_Quiz_8_Topics", "Quiz\n(n=8)"),
        ("HTTP_Roadmap_4_Scenarios", "Roadmap\n(n=4)"),
    ]
    rx = np.arange(len(rel_groups))
    http_ok_pct = [float(lidx.loc[k, "http_success_rate"]) * 100.0 for k, _ in rel_groups]
    llm_ok_pct = [float(lidx.loc[k, "live_llm_count"]) / max(1, int(lidx.loc[k, "measured_requests"])) * 100.0 for k, _ in rel_groups]
    err_pct = [float(lidx.loc[k, "error_rate"]) * 100.0 for k, _ in rel_groups]
    ax2.bar(rx - 0.25, http_ok_pct, width=0.24, color="#009E73", label="HTTP 200 Transport (%)")
    ax2.bar(rx, llm_ok_pct, width=0.24, color="#7B2CBF", label="Error-Free Live LLM (%)")
    ax2.bar(rx + 0.25, err_pct, width=0.24, color="#DC2626", label="HTTP/Gen Error Rate (%)")
    for i in range(len(rx)):
        ax2.text(rx[i] - 0.25, http_ok_pct[i] + 2, f"{http_ok_pct[i]:.0f}", ha="center", va="bottom", fontsize=7.0)
        ax2.text(rx[i], llm_ok_pct[i] + 2, f"{llm_ok_pct[i]:.0f}", ha="center", va="bottom", fontsize=7.0)
        if err_pct[i] > 0:
            ax2.text(rx[i] + 0.25, err_pct[i] + 2, f"{err_pct[i]:.0f}", ha="center", va="bottom", fontsize=7.0, color="#991B1B")
    ax2.set_xticks(rx, [lbl for _, lbl in rel_groups])
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 126)
    ax2.legend(loc="upper center", ncol=2, fontsize=7.4)
    ax2.set_title("(B) HTTP Transport vs Live LLM Generation vs Error Rate")

    fig.suptitle("Supplementary Figure 6: Live API Robustness and End-to-End Reliability")
    add_protocol_footer(
        fig,
        "Source: robustness_summary_live.csv & http_latency_summary.csv. Strictly separates HTTP 200 transport success from Live LLM generation vs abstention/extractive fallback.",
    )
    saved = save_figure(fig, "sup_fig06_live_api_robustness_and_reliability", png_dir, svg_dir)
    records.append({
        "figure_number": 6,
        "figure_id": "sup_fig06_live_api_robustness_and_reliability",
        "title": "Live API Robustness and End-to-End Reliability",
        "protocol": "20 robustness cases via real /api/chat (query-level prompt injection) + HTTP reliability matrix across all endpoints.",
        "source_artifacts": [
            "evaluation/artifacts/ac04_completion/robustness_summary_live.csv",
            "evaluation/artifacts/ac04_completion/http_latency_summary.csv",
        ],
        **saved,
    })

    # Write supplementary figure manifest
    norm_figs = []
    for rec in records:
        item = dict(rec)
        item["png_path"] = Path(rec["png_path"]).relative_to(root).as_posix()
        item["svg_path"] = Path(rec["svg_path"]).relative_to(root).as_posix()
        norm_figs.append(item)

    (out_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "style": style_meta,
                "total_supplementary_figures": len(norm_figs),
                "figures": norm_figs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return norm_figs


def validate_ac04_completion(root: Path, *, write_manifest: bool = False) -> dict[str, Any]:
    """Validate all 6 supplementary figures, manifests, zero secret exposure, and frozen integrity."""
    from evaluation.path_resolver import resolve_ac04_file, resolve_path

    fman = json.loads(resolve_ac04_file(root, "figure_manifest.json").read_text(encoding="utf-8"))
    figs = fman["figures"]
    stage05_integrity = verify_inputs_and_integrity(root)

    png_ok = 0
    svg_ok = 0
    for f in figs:
        pp = resolve_path(root, f["png_path"])
        sp = resolve_path(root, f["svg_path"])
        if pp.is_file():
            with Image.open(pp) as im:
                dpi = im.info.get("dpi", (0, 0))
                std = float(np.asarray(im.convert("RGB"), dtype=np.float32).std())
                if im.size[0] >= 1200 and float(dpi[0]) >= 299.0 and std > 5.0:
                    png_ok += 1
        if sp.is_file():
            try:
                tree = ET.parse(sp)
                if tree.getroot().tag.endswith("svg"):
                    svg_ok += 1
            except ET.ParseError:
                pass

    # Secret scan across ALL non-PNG AC-04 artifacts (supports both legacy and relocated locations)
    secret_pat = re.compile(r"(gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{30,})")
    secret_hits: list[str] = []
    checksums: dict[str, dict[str, Any]] = {}
    scan_paths: list[Path] = []
    legacy_dir = root / "evaluation" / "artifacts" / "ac04_completion"
    reloc_path = root / "evaluation" / "manifests" / "relocation_manifest.json"
    if legacy_dir.exists() and any(legacy_dir.rglob("*")):
        scan_paths = [p for p in sorted(legacy_dir.rglob("*")) if p.is_file()]
    elif reloc_path.is_file():
        rdata = json.loads(reloc_path.read_text(encoding="utf-8"))
        scan_paths = [
            root / e["new_path"]
            for e in rdata.get("relocated_files", [])
            if e["old_path"].startswith("evaluation/artifacts/ac04_completion/") and (root / e["new_path"]).is_file()
        ]
    for p in scan_paths:
        rel = p.relative_to(root).as_posix()
        checksums[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if p.suffix.lower() != ".png":
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if secret_pat.search(txt):
                secret_hits.append(rel)

    report_path = resolve_path(root, "STAGE_04_AC04_COMPLETION_REPORT.md")
    report_ok = False
    broken_links: list[str] = []
    if report_path.is_file():
        rtxt = report_path.read_text(encoding="utf-8")
        for t in re.findall(r"\[[^\]]*\]\(([^)]+)\)", rtxt):
            t = t.strip()
            if t.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean = t.split("#", 1)[0]
            if clean.startswith("file:///") or not (report_path.parent / clean).resolve().exists():
                broken_links.append(t)
        all_figs_mentioned = all(f["figure_id"] in rtxt for f in figs)
        report_ok = len(broken_links) == 0 and all_figs_mentioned

    # Check Stage 05 original 28 PNGs still untouched
    orig_png_dir = root / "evaluation" / "figures" / "original" / "png"
    legacy_s5_png_dir = root / "evaluation" / "artifacts" / "stage05" / "figures" / "png"
    s5_pngs = list((orig_png_dir if orig_png_dir.exists() else legacy_s5_png_dir).glob("fig*.png"))
    s5_untouched = len(s5_pngs) == 28

    summary = {
        "supplementary_figures_expected": 6,
        "supplementary_figures_found": len(figs),
        "png_passed": png_ok,
        "svg_passed": svg_ok,
        "secret_hits": secret_hits,
        "stage04_stage05_integrity_ok": stage05_integrity["all_verified"],
        "stage05_28_figures_preserved": s5_untouched,
        "report_exists": report_path.is_file(),
        "report_broken_links": broken_links,
        "report_ok": report_ok,
    }
    overall_pass = (
        len(figs) == 6
        and png_ok == 6
        and svg_ok == 6
        and len(secret_hits) == 0
        and stage05_integrity["all_verified"]
        and s5_untouched
        and report_ok
    )

    if write_manifest:
        out_manifest = resolve_ac04_file(root, "manifest.json")
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "task": "Supplementary Stage 04 AC-04 Completion (Live Groq & HTTP E2E)",
                    "overall_validation_pass": overall_pass,
                    "validation_summary": summary,
                    "artifact_checksums": checksums,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return {"overall_pass": overall_pass, "summary": summary}
