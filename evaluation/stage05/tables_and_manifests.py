"""Generate Stage 05 traceable tables, configuration recommendations, and manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.stage05 import ALL_GEN_ORDER, COMBO_LABELS, COMBO_ORDER
from evaluation.stage05.data_loader import STAGE04_AC_RECONCILIATION, Stage04Bundle
from evaluation.stage05.style import sha256_file


def generate_stage05_tables(bundle: Stage04Bundle, tables_dir: Path) -> list[dict[str, Any]]:
    """Export 10 clean, directly traceable summary CSV tables for Stage 05."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []

    def _save(df: pd.DataFrame, filename: str, sources: list[str], desc: str) -> None:
        out = tables_dir / filename
        df.to_csv(out, index=False, encoding="utf-8")
        exported.append({
            "table_id": out.stem,
            "path": f"evaluation/artifacts/stage05/tables/{filename}",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "source_artifacts": sources,
            "description": desc,
            "sha256": sha256_file(out),
        })

    # Table 01: Chunking Summary
    _save(
        bundle.chunking_metrics,
        "table_01_chunking_summary.csv",
        ["evaluation/artifacts/stage04/chunking/chunking_metrics.csv"],
        "Dual-tokenizer chunking diagnostics, coverage, redundancy, and storage comparison (Profile A vs Profile B).",
    )

    # Table 02: Retrieval Quality at K=5, K=15, K=60
    ret = bundle.retrieval_summary
    t2_rows = []
    for cid in COMBO_ORDER:
        r5 = ret[(ret["combo_id"] == cid) & (ret["k"] == 5)].iloc[0]
        r15 = ret[(ret["combo_id"] == cid) & (ret["k"] == 15)].iloc[0]
        r60 = ret[(ret["combo_id"] == cid) & (ret["k"] == 60)].iloc[0]
        t2_rows.append({
            "combo_id": cid,
            "label": COMBO_LABELS[cid],
            "p_at_5": round(float(r5["precision_at_k"]), 4),
            "pooled_r_at_5": round(float(r5["pooled_recall_at_k"]), 4),
            "mrr_at_5": round(float(r5["mrr_at_k"]), 4),
            "ndcg_at_5": round(float(r5["ndcg_at_k"]), 4),
            "map_at_5": round(float(r5["map_at_k"]), 4),
            "f1_at_5": round(float(r5["f1_at_k"]), 4),
            "hit_at_5": round(float(r5["hit_rate_at_k"]), 4),
            "ndcg_at_15": round(float(r15["ndcg_at_k"]), 4),
            "pooled_r_at_15": round(float(r15["pooled_recall_at_k"]), 4),
            "ndcg_at_60": round(float(r60["ndcg_at_k"]), 4),
            "pooled_r_at_60": round(float(r60["pooled_recall_at_k"]), 4),
            "judged_at_60": round(float(r60["judged_at_k"]), 4),
        })
    _save(
        pd.DataFrame(t2_rows),
        "table_02_retrieval_k5_k15_k60.csv",
        ["evaluation/artifacts/stage04/retrieval/retrieval_metrics_summary.csv"],
        "Retrieval quality across all 8 experimental combinations at K=5, K=15, and K=60 (pool=120, Judged@K=1.0).",
    )

    # Table 03: Evidence Coverage Summary (K=5, K=15, K=60)
    ev = bundle.evidence_summary
    t3_rows = []
    for cid in COMBO_ORDER:
        e5 = ev[(ev["combo_id"] == cid) & (ev["k"] == 5)].iloc[0]
        e15 = ev[(ev["combo_id"] == cid) & (ev["k"] == 15)].iloc[0]
        e60 = ev[(ev["combo_id"] == cid) & (ev["k"] == 60)].iloc[0]
        t3_rows.append({
            "combo_id": cid,
            "ev_recall_k5": round(float(e5["evidence_recall_at_k"]), 4),
            "ev_precision_k5": round(float(e5["evidence_precision_at_k"]), 4),
            "complete_ev_k5": round(float(e5["complete_evidence_rate_at_k"]), 4),
            "partial_ev_k5": round(float(e5["partial_evidence_coverage_at_k"]), 4),
            "ku_any_k5": round(float(e5["ku_coverage_at_k"]), 4),
            "semantic_ku_k5": round(float(e5["semantic_ku_recall_at_k"]), 4),
            "ev_redundancy_k5": round(float(e5["evidence_redundancy_at_k"]), 4),
            "ev_recall_k15": round(float(e15["evidence_recall_at_k"]), 4),
            "complete_ev_k15": round(float(e15["complete_evidence_rate_at_k"]), 4),
            "ev_recall_k60": round(float(e60["evidence_recall_at_k"]), 4),
            "complete_ev_k60": round(float(e60["complete_evidence_rate_at_k"]), 4),
        })
    _save(
        pd.DataFrame(t3_rows),
        "table_03_evidence_summary.csv",
        ["evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv"],
        "Source-backed evidence metrics and coverage hierarchy at K=5, K=15, and K=60.",
    )

    # Table 04: Performance & Resource Summary (Strictly Separated Protocols)
    lat = bundle.latency_summary
    stg = bundle.stage_latency
    prod = bundle.production_profile_latency.set_index("combo_id")
    cs = bundle.cold_start_latency.groupby("combo_id")[["model_load_ms", "bm25_build_ms", "total_cold_start_ms"]].mean()
    res = bundle.resource_usage.set_index("combo_id")
    t4_rows = []
    for cid in COMBO_ORDER:
        w5 = lat[(lat["combo_id"] == cid) & (lat["k"] == 5)].iloc[0]
        w60 = lat[(lat["combo_id"] == cid) & (lat["k"] == 60)].iloc[0]
        s5 = stg[(stg["combo_id"] == cid) & (stg["k"] == 5)].iloc[0]
        t4_rows.append({
            "combo_id": cid,
            "warm_k5_pool120_mean_ms": round(float(w5["mean_ms"]), 2),
            "warm_k5_pool120_p50_ms": round(float(w5["p50_ms"]), 2),
            "warm_k5_pool120_p95_ms": round(float(w5["p95_ms"]), 2),
            "warm_k60_pool120_p95_ms": round(float(w60["p95_ms"]), 2),
            "stage_rerank_k5_pool120_mean_ms": round(float(s5["rerank_mean_ms"]), 2),
            "prod_profile_k5_pool10_p50_ms": round(float(prod.loc[cid, "p50_ms"]), 2),
            "prod_profile_k5_pool10_p95_ms": round(float(prod.loc[cid, "p95_ms"]), 2),
            "cold_start_mean_sec": round(float(cs.loc[cid, "total_cold_start_ms"]) / 1000.0, 2),
            "cold_bm25_build_mean_sec": round(float(cs.loc[cid, "bm25_build_ms"]) / 1000.0, 2),
            "peak_gpu_alloc_mb": round(float(res.loc[cid, "peak_gpu_allocated_max_mb"]), 2),
            "cold_process_rss_mean_mb": round(float(res.loc[cid, "process_rss_mean_mb"]), 2),
        })
    _save(
        pd.DataFrame(t4_rows),
        "table_04_latency_protocols.csv",
        [
            "evaluation/artifacts/stage04/performance/latency_summary.csv",
            "evaluation/artifacts/stage04/performance/stage_latency.csv",
            "evaluation/artifacts/stage04/performance/production_profile_latency.csv",
            "evaluation/artifacts/stage04/performance/cold_start_latency.csv",
            "evaluation/artifacts/stage04/performance/resource_usage.csv",
        ],
        "Performance and resource summary across clearly separated protocols (Warm pool=120, Stage breakdown, Prod-Param pool=10, Cold-start seconds).",
    )

    # Table 05: Controlled Answer Generation Summary (N/A explicitly preserved)
    ans = bundle.answer_summary.set_index("combo_id")
    cols_gen = [
        "answer_correctness_norm", "answer_relevancy_norm", "faithfulness_norm",
        "answer_completeness_norm", "citation_accuracy_norm", "citation_validity",
        "unsupported_claim_rate", "contradiction_rate", "complexity_correctness_norm", "code_correctness_norm",
    ]
    t5_rows = []
    for cid in ALL_GEN_ORDER:
        row_out: dict[str, Any] = {"combo_id": cid, "n_answers": int(ans.loc[cid, "n_answers"])}
        for col in cols_gen:
            v = ans.loc[cid, col]
            row_out[col] = "N/A" if pd.isna(v) else round(float(v), 4)
        t5_rows.append(row_out)
    _save(
        pd.DataFrame(t5_rows),
        "table_05_generation_summary.csv",
        ["evaluation/artifacts/stage04/generation/answer_metrics_summary.csv"],
        "Controlled Agent Generation 10-metric summary across all 8 RAG combos and NO_RAG (N/A preserved).",
    )

    # Table 06: Key Ablations & Holm-Adjusted Significance (K=5 nDCG & K=15 EvRecall)
    ab = bundle.ablation_results
    sub_ab = ab[
        ((ab["k"] == 5) & (ab["metric"] == "ndcg_at_k"))
        | ((ab["k"] == 15) & (ab["metric"] == "evidence_recall_at_k"))
    ].copy()
    _save(
        sub_ab,
        "table_06_ablation_and_stats.csv",
        ["evaluation/artifacts/stage04/ablation/ablation_results.csv"],
        "Paired component ablation differences with 95% bootstrap CIs and Holm-adjusted p-values at K=5 (nDCG) and K=15 (EvRecall).",
    )

    # Table 07: Sensitivity Summary (Candidate Pool & Context Budget combined cleanly with protocol column)
    p_s = bundle.candidate_pool_sensitivity.copy()
    p_s.insert(0, "sensitivity_experiment", "candidate_pool_single_pass_n40")
    _save(
        p_s,
        "table_07_sensitivity_summary.csv",
        [
            "evaluation/artifacts/stage04/sensitivity/candidate_pool_sensitivity.csv",
            "evaluation/artifacts/stage04/sensitivity/context_budget_sensitivity.csv",
        ],
        "Single-pass Candidate Pool sensitivity (pool=10,30,60,120, K=5). Context budget sensitivity preserved in Stage 04 CSV.",
    )

    # Table 08: Robustness & Study Features
    rob = bundle.robustness_metrics.copy()
    _save(
        rob,
        "table_08_robustness_and_study.csv",
        [
            "evaluation/artifacts/stage04/robustness/robustness_metrics.csv",
            "evaluation/artifacts/stage04/study_features/study_features_summary.csv",
        ],
        "Robustness metrics across 6 categories (n=20 independent cases); Study Features separated in Stage 04 summary & functional_test_results.json.",
    )

    # Table 09: Pareto Frontier (K=5 and K=15, pool=120)
    pf_sub = bundle.pareto_frontier[bundle.pareto_frontier["k"].isin([5, 15])].copy()
    _save(
        pf_sub,
        "table_09_pareto_and_recommendations.csv",
        ["evaluation/artifacts/stage04/tradeoff/pareto_frontier.csv"],
        "Quality-Latency-Resource Pareto frontier at K=5 and K=15 under matched Main Warm Latency (pool=120).",
    )

    # Table 10: Stage 04 Acceptance Criteria Reconciliation
    _save(
        pd.DataFrame(STAGE04_AC_RECONCILIATION),
        "table_10_stage04_ac_reconciliation.csv",
        ["STAGE_04_COMPREHENSIVE_EVALUATION_REPORT.md"],
        "Explicit reconciliation between canonical Stage 04 AC-01..AC-06 definitions and checkpoint short labels (5 PASS / 1 PARTIAL preserved).",
    )

    return exported


def build_configuration_recommendations(bundle: Stage04Bundle) -> dict[str, Any]:
    """Produce structured exploratory configuration recommendations grounded in Stage 04 metrics."""
    return {
        "governance": {
            "status": "EXPLORATORY_NOT_HELD_OUT_VALIDATED",
            "human_validated": False,
            "human_approval_status": "DRAFT",
            "production_groq_e2e_status": "BLOCKED",
            "note": (
                "All recommendations below are exploratory selections derived from the 40-question Stage 04 benchmark "
                "using AI-generated pooled relevance judgments and Controlled Agent Generation. They are NOT independently "
                "validated production settings."
            ),
        },
        "operating_scenarios": {
            "quality_oriented": {
                "recommended_combo": "A4",
                "profile_id": "recursive_512_100",
                "retrieval_method": "hybrid_rerank",
                "evidence_summary": {
                    "ndcg_at_5": 0.7053,
                    "mrr_at_5": 0.9875,
                    "precision_at_5": 0.7000,
                    "evidence_recall_at_5": 0.7208,
                    "evidence_recall_at_15": 0.9083,
                    "complete_evidence_rate_at_15": 0.8000,
                    "rag_answer_correctness": 0.9250,
                    "rag_contradiction_rate": 0.0000,
                    "warm_p95_k5_pool120_ms": 3693.1,
                    "peak_gpu_allocated_mb": 2662.0,
                },
                "caveats": [
                    "Does NOT win every metric: NO_RAG Correctness (0.9688) exceeds A4; A1 leads Faithfulness (0.9250) and Citation Accuracy (0.8500); B4 leads Pooled Recall@5 (0.3509) and has 0% embedding truncation.",
                    "Highest warm retrieval latency at pool=120 (P95 = 3,693.1 ms) and 43.2% MiniLM embedding truncation.",
                ],
            },
            "low_latency_lightweight": {
                "recommended_combo": "A1",
                "profile_id": "recursive_512_100",
                "retrieval_method": "bm25",
                "evidence_summary": {
                    "ndcg_at_5": 0.5584,
                    "evidence_recall_at_5": 0.6583,
                    "evidence_recall_at_15": 0.8000,
                    "faithfulness": 0.9250,
                    "citation_accuracy": 0.8500,
                    "rag_answer_correctness": 0.8125,
                    "warm_p95_k5_pool120_ms": 607.5,
                    "prod_profile_p95_k5_pool10_ms": 447.0,
                    "peak_gpu_allocated_mb": 0.0,
                },
                "caveats": [
                    "Pareto-optimal for CPU-only / strict latency budgets; zero GPU VRAM required.",
                    "Vulnerable to lexical mismatch, paraphrases, and Vietnamese unaccented queries (accent_insensitive=false).",
                ],
            },
            "alternative_chunking": {
                "recommended_combo": "B4",
                "profile_id": "structure_adaptive",
                "retrieval_method": "hybrid_rerank",
                "evidence_summary": {
                    "embedding_truncation_rate_pct": 0.0,
                    "overlap_redundancy_pct": 1.06,
                    "evidence_redundancy_at_5": 0.0075,
                    "pooled_recall_at_5": 0.3509,
                    "isolated_rerank_mean_k5_pool120_ms": 1364.8,
                    "isolated_rerank_speedup_vs_a4_pct": 41.2,
                    "warm_p95_k5_pool120_ms": 3214.3,
                    "warm_p95_speedup_vs_a4_k5_pct": 13.0,
                    "ndcg_at_5": 0.5591,
                    "rag_answer_correctness": 0.7500,
                },
                "caveats": [
                    "Context fragmentation (median 112 chars, 2.1x chunk count) reduces nDCG@5 (-20.7%, Holm p=0.001) and EvRecall@15 (-17.4%, Holm p=0.0035) relative to A4.",
                    "Requires ~1.5x longer cold BM25 build (15.5s vs 10.7s) and 2.1x theoretical vector storage.",
                ],
            },
            "practical_balanced_exploratory": {
                "recommended_combo": "A4 (tuned candidate_top_k=30)",
                "chunking": "recursive_512_100 (chunk_size=512, overlap=100)",
                "retrieval": "BM25 + Dense + RRF (k=60) + BGE-reranker-v2-m3",
                "candidate_pool": 30,
                "top_k_range": "5-10 (15 for high-completeness analytical tasks)",
                "context_budget_tokens": 2048,
                "evidence_summary": {
                    "single_pass_pool30_ndcg_at_5": 0.7039,
                    "ndcg_retention_vs_pool120_pct": 99.80,
                    "single_pass_pool30_ev_recall_at_5": 0.7375,
                    "single_pass_pool30_mean_ms": 1628.0,
                    "single_pass_pool30_p95_ms": 1777.0,
                    "budget_2048_n10_correctness": 0.9500,
                    "budget_2048_n10_faithfulness": 0.9250,
                },
                "protocol_warning": (
                    "Pool=30 latency originates from Protocol 5 (640 single-pass calls) and budget=2048 sensitivity "
                    "originates from an exploratory n=10 subset; both must be validated on an independent held-out set "
                    "and live production stack before deployment."
                ),
            },
        },
        "future_work_not_implemented_in_stage05": [
            "Enable accent_insensitive=true (or dual diacritic-aware + folded BM25 fields) to remediate the 0.0% Vietnamese unaccented typo failure (-0.6667 EvRecall@5).",
            "Migrate from QdrantLocal SQLite (~461-645 ms/search leg) to Qdrant Server (HNSW) and evaluate persistent/async BM25 cache refresh to eliminate 300s TTL rebuild spikes.",
            "Refine structure_adaptive chunking with minimum semantic unit merging (target 80-120 tokens) to combine 0% embedding truncation with lower context fragmentation.",
            "Conduct human domain-expert ground-truth validation (moving human_validated=false -> true) and live Groq + HTTP E2E benchmarking once API credentials and production Qdrant indexing are authorized.",
        ],
        "source_stage04_recommendations": bundle.stage04_recommendations,
    }


def write_stage05_manifests(
    bundle: Stage04Bundle,
    stage05_dir: Path,
    figure_records: list[dict[str, Any]],
    table_records: list[dict[str, Any]],
    style_meta: dict[str, Any],
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write figure_manifest.json, configuration_recommendations.json, and manifest.json."""
    stage05_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).isoformat()

    # Normalize paths relative to repository root for portable manifests
    norm_figs = []
    for rec in figure_records:
        item = dict(rec)
        item["png_path"] = Path(rec["png_path"]).relative_to(bundle.root).as_posix()
        item["svg_path"] = Path(rec["svg_path"]).relative_to(bundle.root).as_posix()
        norm_figs.append(item)

    fig_manifest_path = stage05_dir / "figure_manifest.json"
    fig_manifest_payload = {
        "generated_utc": now_utc,
        "total_logical_figures": len(norm_figs),
        "total_png_files": len(norm_figs),
        "total_svg_files": len(norm_figs),
        "style": style_meta,
        "figures": norm_figs,
    }
    fig_manifest_path.write_text(json.dumps(fig_manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rec_path = stage05_dir / "configuration_recommendations.json"
    rec_payload = build_configuration_recommendations(bundle)
    rec_path.write_text(json.dumps(rec_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = stage05_dir / "manifest.json"
    manifest_payload = {
        "stage": "Stage 05 — Visualization, Comparative Analysis & Final Evaluation Conclusion",
        "generated_utc": now_utc,
        "input_integrity": bundle.integrity,
        "style": style_meta,
        "outputs": {
            "logical_figures": len(norm_figs),
            "png_count": len(norm_figs),
            "svg_count": len(norm_figs),
            "tables_count": len(table_records),
            "tables": table_records,
            "figure_manifest": "evaluation/artifacts/stage05/figure_manifest.json",
            "configuration_recommendations": "evaluation/artifacts/stage05/configuration_recommendations.json",
            "chart_validation": "evaluation/artifacts/stage05/chart_validation.json",
            "final_report": "STAGE_05_FINAL_EVALUATION_REPORT.md",
        },
        "validation_summary": validation_report.get("summary") if validation_report else None,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "figure_manifest": fig_manifest_path,
        "configuration_recommendations": rec_path,
        "manifest": manifest_path,
    }
