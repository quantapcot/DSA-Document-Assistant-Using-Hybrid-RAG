"""Stage 04 – Steps 11–15: Ablation, Sensitivity, Statistical Analysis, Error Analysis & Pareto Trade-off."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.stage04.metrics import (
    STAGE04_COMBOS,
    STAGE04_KS,
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
)

PAIRED_COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    # (comparison_id, category, combo_new, combo_base)
    ("H1_A3_vs_A1", "hybrid_contribution", "A3", "A1"),
    ("H2_A3_vs_A2", "hybrid_contribution", "A3", "A2"),
    ("H3_B3_vs_B1", "hybrid_contribution", "B3", "B1"),
    ("H4_B3_vs_B2", "hybrid_contribution", "B3", "B2"),
    ("R1_A4_vs_A3", "reranker_contribution", "A4", "A3"),
    ("R2_B4_vs_B3", "reranker_contribution", "B4", "B3"),
    ("C1_B1_vs_A1", "chunking_contribution", "B1", "A1"),
    ("C2_B2_vs_A2", "chunking_contribution", "B2", "A2"),
    ("C3_B3_vs_A3", "chunking_contribution", "B3", "A3"),
    ("C4_B4_vs_A4", "chunking_contribution", "B4", "A4"),
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _f(val: Any) -> float | None:
    if val is None or val == "" or val == "None":
        return None
    return float(val)


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (observed_mean, ci_low, ci_high) via deterministic bootstrap resampling."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs = mean(values)
    if n == 1:
        return obs, obs, obs
    rng = random.Random(seed)
    boot_means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    ]
    boot_means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * (n_resamples - 1)))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * (n_resamples - 1)))
    return obs, boot_means[lo_idx], boot_means[hi_idx]


def paired_bootstrap_test(
    new_vals: list[float],
    base_vals: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    tie_tol: float = 1e-9,
) -> dict[str, Any]:
    """Paired query-level bootstrap CI, two-sided p-value (sign-flip permutation/bootstrap), Cohen's d, W/T/L."""
    if len(new_vals) != len(base_vals) or not new_vals:
        raise ValueError("Paired lists must be nonempty and equal length")
    n = len(new_vals)
    diffs = [a - b for a, b in zip(new_vals, base_vals)]
    obs_diff = mean(diffs)
    mean_new = mean(new_vals)
    mean_base = mean(base_vals)
    rel_diff = (obs_diff / mean_base) if abs(mean_base) > 1e-12 else None

    sd = stdev(diffs) if n > 1 else 0.0
    cohens_d = (obs_diff / sd) if sd > 1e-12 else 0.0

    wins = sum(1 for d in diffs if d > tie_tol)
    losses = sum(1 for d in diffs if d < -tie_tol)
    ties = n - wins - losses

    rng = random.Random(seed)
    boot_diffs = [
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    ]
    boot_diffs.sort()
    ci_low = boot_diffs[int(math.floor(0.025 * (n_resamples - 1)))]
    ci_high = boot_diffs[int(math.ceil(0.975 * (n_resamples - 1)))]

    # Two-sided paired randomization/sign-flip test p-value (exact null distribution under H0: diff symmetric around 0)
    rng_p = random.Random(seed + 1009)
    abs_obs = abs(obs_diff)
    extreme = 0
    for _ in range(n_resamples):
        sim = sum((d if rng_p.getrandbits(1) else -d) for d in diffs) / n
        if abs(sim) >= abs_obs - 1e-15:
            extreme += 1
    p_value = (extreme + 1) / (n_resamples + 1)

    return {
        "n_queries": n,
        "mean_new": round(mean_new, 6),
        "mean_base": round(mean_base, 6),
        "abs_diff": round(obs_diff, 6),
        "rel_diff": round(rel_diff, 6) if rel_diff is not None else None,
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
        "cohens_d_paired": round(cohens_d, 4),
        "p_value_raw": round(p_value, 6),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tie_tolerance": tie_tol,
    }


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank_idx))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def compute_pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    maximize_keys: tuple[str, ...],
    minimize_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Identify non-dominated Pareto-optimal configurations (no arbitrary composite weights)."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        dominated_by: list[str] = []
        for j, o in enumerate(rows):
            if i == j:
                continue
            ge_all = all(float(o[k]) >= float(r[k]) - 1e-12 for k in maximize_keys) and all(
                float(o[k]) <= float(r[k]) + 1e-12 for k in minimize_keys
            )
            gt_any = any(float(o[k]) > float(r[k]) + 1e-12 for k in maximize_keys) or any(
                float(o[k]) < float(r[k]) - 1e-12 for k in minimize_keys
            )
            if ge_all and gt_any:
                dominated_by.append(str(o["combo_id"]))
        out.append({
            **r,
            "is_pareto_optimal": len(dominated_by) == 0,
            "dominated_by": ",".join(dominated_by),
        })
    return out


def run_comprehensive_analysis(
    *,
    root: Path,
    stage04_dir: Path,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Steps 11–15 from saved Stage 04 raw/per-query artifacts (0 extra retrieval calls)."""
    ret_pq = _read_csv(stage04_dir / "retrieval/retrieval_metrics_per_query.csv")
    ev_pq = _read_csv(stage04_dir / "evidence/evidence_metrics_per_query.csv")
    lat_sum = _read_csv(stage04_dir / "performance/latency_summary.csv")
    res_sum = _read_csv(stage04_dir / "performance/resource_usage.csv")
    gen_pq = _read_csv(stage04_dir / "generation/answer_metrics_per_query.csv")

    if not ret_pq or not ev_pq:
        raise RuntimeError("Missing retrieval or evidence per-query CSVs; run retrieval & evidence first.")

    # Index per-query retrieval + evidence metrics by (combo_id, k, query_id)
    ev_by_ckq = {(r["combo_id"], int(r["k"]), r["query_id"]): r for r in ev_pq}
    merged_by_ck: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    query_ids_sorted = sorted({r["query_id"] for r in ret_pq})
    for r in ret_pq:
        cid, k, qid = r["combo_id"], int(r["k"]), r["query_id"]
        er = ev_by_ckq[(cid, k, qid)]
        merged_by_ck[(cid, k)][qid] = {
            **r,
            "evidence_recall_at_k": er["evidence_recall_at_k"],
            "evidence_precision_at_k": er["evidence_precision_at_k"],
            "complete_evidence_rate_at_k": er["complete_evidence_rate_at_k"],
            "partial_evidence_coverage_at_k": er["partial_evidence_coverage_at_k"],
            "evidence_redundancy_at_k": er["evidence_redundancy_at_k"],
            "semantic_ku_recall_at_k": er.get("semantic_ku_recall_at_k"),
        }

    # --- STEP 11 & 13: Ablation, Descriptive Stats, Bootstrap CIs, Paired Bootstrap, Win/Tie/Loss ---
    ablation_dir = stage04_dir / "ablation"
    stats_dir = stage04_dir / "statistics"
    sens_dir = stage04_dir / "sensitivity"
    err_dir = stage04_dir / "error_analysis"
    trade_dir = stage04_dir / "tradeoff"
    for d in (ablation_dir, stats_dir, sens_dir, err_dir, trade_dir):
        d.mkdir(parents=True, exist_ok=True)

    key_metrics = (
        "precision_at_k",
        "pooled_recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "map_at_k",
        "f1_at_k",
        "hit_rate_at_k",
        "evidence_recall_at_k",
        "evidence_precision_at_k",
        "complete_evidence_rate_at_k",
    )

    stat_summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for cid, pid, method in STAGE04_COMBOS:
        for k in STAGE04_KS:
            qmap = merged_by_ck[(cid, k)]
            for m in key_metrics:
                vals = [_f(qmap[qid][m]) for qid in query_ids_sorted]
                clean = [v for v in vals if v is not None]
                stat_summary_rows.append({
                    "combo_id": cid, "profile_id": pid, "method": method, "k": k, "metric": m,
                    "n": len(clean),
                    "mean": round(mean(clean), 6),
                    "median": round(median(clean), 6),
                    "std": round(stdev(clean), 6) if len(clean) > 1 else 0.0,
                    "min": round(min(clean), 6),
                    "max": round(max(clean), 6),
                })
                if k in (1, 3, 5, 10, 15, 30, 60) and m in (
                    "precision_at_k", "pooled_recall_at_k", "mrr_at_k", "ndcg_at_k",
                    "evidence_recall_at_k", "complete_evidence_rate_at_k",
                ):
                    obs, lo, hi = bootstrap_ci(clean, n_resamples=n_bootstrap, seed=seed)
                    ci_rows.append({
                        "combo_id": cid, "profile_id": pid, "method": method, "k": k, "metric": m,
                        "n": len(clean), "mean": round(obs, 6),
                        "ci_95_low": round(lo, 6), "ci_95_high": round(hi, 6),
                        "bootstrap_resamples": n_bootstrap, "seed": seed,
                    })

    _write_csv(stats_dir / "statistical_summary.csv", stat_summary_rows)
    _write_csv(stats_dir / "confidence_intervals.csv", ci_rows)

    # Paired comparisons across K = (1, 3, 5, 10, 15, 20, 30, 45, 60) for primary metrics (with Holm per (k, metric) family)
    paired_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wtl_rows: list[dict[str, Any]] = []
    paired_metrics = ("ndcg_at_k", "precision_at_k", "pooled_recall_at_k", "mrr_at_k", "evidence_recall_at_k", "complete_evidence_rate_at_k")

    for k in STAGE04_KS:
        for m in paired_metrics:
            family_rows: list[dict[str, Any]] = []
            for comp_id, cat, c_new, c_base in PAIRED_COMPARISONS:
                v_new = [_f(merged_by_ck[(c_new, k)][q][m]) for q in query_ids_sorted]
                v_base = [_f(merged_by_ck[(c_base, k)][q][m]) for q in query_ids_sorted]
                pairs = [(a, b) for a, b in zip(v_new, v_base) if a is not None and b is not None]
                res = paired_bootstrap_test(
                    [a for a, _ in pairs], [b for _, b in pairs],
                    n_resamples=n_bootstrap if k in (5, 15) else 2000,
                    seed=seed,
                )
                row = {
                    "comparison_id": comp_id, "category": cat, "combo_new": c_new, "combo_base": c_base,
                    "k": k, "metric": m, **res,
                }
                family_rows.append(row)
            adj_ps = holm_bonferroni_adjust([r["p_value_raw"] for r in family_rows])
            for r, ap in zip(family_rows, adj_ps):
                r["p_value_holm"] = ap
                r["significant_holm_005"] = ap < 0.05
                paired_rows.append(r)
                if k in (5, 15, 60):
                    ablation_rows.append({
                        "comparison_id": r["comparison_id"], "category": r["category"],
                        "combo_new": r["combo_new"], "combo_base": r["combo_base"],
                        "k": k, "metric": m, "mean_new": r["mean_new"], "mean_base": r["mean_base"],
                        "abs_improvement": r["abs_diff"], "rel_improvement": r["rel_diff"],
                        "ci_95_low": r["ci_95_low"], "ci_95_high": r["ci_95_high"],
                        "p_value_holm": r["p_value_holm"], "significant_holm_005": r["significant_holm_005"],
                    })
                if k in (5, 15):
                    wtl_rows.append({
                        "comparison_id": r["comparison_id"], "category": r["category"],
                        "combo_new": r["combo_new"], "combo_base": r["combo_base"],
                        "k": k, "metric": m, "wins": r["wins"], "ties": r["ties"], "losses": r["losses"],
                        "tie_tolerance": r["tie_tolerance"],
                    })

    _write_csv(stats_dir / "paired_bootstrap_results.csv", paired_rows)
    _write_csv(stats_dir / "win_tie_loss.csv", wtl_rows)
    _write_csv(ablation_dir / "ablation_results.csv", ablation_rows)
    _write_csv(ablation_dir / "paired_method_comparisons.csv", [r for r in paired_rows if r["k"] in (5, 15)])

    # Step 11.4: RAG contribution vs Without-RAG baseline (when generation metrics exist)
    rag_contrib_rows: list[dict[str, Any]] = []
    if gen_pq:
        by_combo_q: dict[tuple[str, str], dict[str, Any]] = {(r["combo_id"], r["query_id"]): r for r in gen_pq}
        no_rag_q = {r["query_id"]: r for r in gen_pq if r["combo_id"] == "NO_RAG"}
        if no_rag_q:
            for cid, pid, method in STAGE04_COMBOS:
                for m in ("answer_correctness_norm", "answer_completeness_norm", "answer_relevancy_norm"):
                    pairs = [
                        (_f(by_combo_q[(cid, q)][m]), _f(no_rag_q[q][m]))
                        for q in query_ids_sorted
                        if (cid, q) in by_combo_q and q in no_rag_q
                    ]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    if pairs:
                        res = paired_bootstrap_test([a for a, _ in pairs], [b for _, b in pairs], n_resamples=n_bootstrap, seed=seed)
                        rag_contrib_rows.append({"combo_id": cid, "profile_id": pid, "method": method, "metric": m, **res})
    _write_csv(ablation_dir / "rag_contribution.csv", rag_contrib_rows)

    # Step 13.6: Subgroup analysis (VI vs EN, Difficulty, Topic, Question Type, Single vs Multi-evidence) at K=5 and K=15
    subgroup_rows: list[dict[str, Any]] = []
    for k in (5, 15):
        for cid, pid, method in STAGE04_COMBOS:
            qmap = merged_by_ck[(cid, k)]
            group_defs = [
                ("language", lambda r: r["language"]),
                ("difficulty", lambda r: r["difficulty"]),
                ("topic", lambda r: r["topic"]),
                ("question_type", lambda r: r["question_type"]),
                ("evidence_complexity", lambda r: "single_evidence" if int(r["evidence_count"]) == 1 else "multi_evidence"),
            ]
            for dim, fn in group_defs:
                buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for qid in query_ids_sorted:
                    r = qmap[qid]
                    buckets[fn(r)].append(r)
                for gval, items in sorted(buckets.items()):
                    subgroup_rows.append({
                        "combo_id": cid, "profile_id": pid, "method": method, "k": k,
                        "subgroup_dim": dim, "subgroup_value": gval, "group_n": len(items),
                        "precision_at_k": round(mean(_f(x["precision_at_k"]) for x in items), 6),
                        "pooled_recall_at_k": round(mean(_f(x["pooled_recall_at_k"]) for x in items), 6),
                        "mrr_at_k": round(mean(_f(x["mrr_at_k"]) for x in items), 6),
                        "ndcg_at_k": round(mean(_f(x["ndcg_at_k"]) for x in items), 6),
                        "evidence_recall_at_k": round(mean(_f(x["evidence_recall_at_k"]) for x in items), 6),
                        "complete_evidence_rate_at_k": round(mean(_f(x["complete_evidence_rate_at_k"]) for x in items), 6),
                    })
    _write_csv(stats_dir / "subgroup_metrics.csv", subgroup_rows)

    # Step 13.7: Annotation Sensitivity (Primary vs Final Stage 03 judgments)
    prim_path = root / "evaluation/artifacts/stage03/ai_annotations_primary.jsonl"
    rankings_path = stage04_dir / "retrieval/retrieval_rankings.jsonl"
    ann_sens_rows: list[dict[str, Any]] = []
    if prim_path.exists() and rankings_path.exists():
        prim_grades: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
        for ln in prim_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                prim_grades[(rec["query_id"], rec["profile_id"])][rec["chunk_id"]] = int(rec["ai_grade"])
        s4_rankings = [json.loads(ln) for ln in rankings_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        prim_pq: list[dict[str, Any]] = []
        for rec in s4_rankings:
            qid, pid, method, cid = rec["query_id"], rec["profile_id"], rec["method"], rec["combo_id"]
            rids = [h["chunk_id"] for h in rec["results"]]
            for pr in compute_query_retrieval_metrics(rids, prim_grades[(qid, pid)], ks=(5, 15)):
                prim_pq.append({"combo_id": cid, "profile_id": pid, "method": method, "query_id": qid, **pr})
        prim_sum = {(r["combo_id"], r["k"]): r for r in aggregate_retrieval_metrics(prim_pq)}
        fin_sum = {(r["combo_id"], int(r["k"])): r for r in _read_csv(stage04_dir / "retrieval/retrieval_metrics_summary.csv")}
        for cid, pid, method in STAGE04_COMBOS:
            for k in (5, 15):
                p_r = prim_sum[(cid, k)]
                f_r = fin_sum[(cid, k)]
                for m in ("precision_at_k", "pooled_recall_at_k", "mrr_at_k", "ndcg_at_k"):
                    pv = float(p_r[m])
                    fv = float(f_r[m])
                    ann_sens_rows.append({
                        "combo_id": cid, "profile_id": pid, "method": method, "k": k, "metric": m,
                        "primary_only_value": round(pv, 6),
                        "final_adjudicated_value": round(fv, 6),
                        "abs_shift_final_minus_primary": round(fv - pv, 6),
                    })
    _write_csv(stats_dir / "annotation_sensitivity.csv", ann_sens_rows)

    # --- STEP 12.1: Top-K Sensitivity combined table ---
    lat_by_ck = {(r["combo_id"], int(r["k"])): r for r in lat_sum}
    ret_sum = {(r["combo_id"], int(r["k"])): r for r in _read_csv(stage04_dir / "retrieval/retrieval_metrics_summary.csv")}
    ev_sum = {(r["combo_id"], int(r["k"])): r for r in _read_csv(stage04_dir / "evidence/evidence_metrics_summary.csv")}
    topk_rows: list[dict[str, Any]] = []
    for cid, pid, method in STAGE04_COMBOS:
        for k in STAGE04_KS:
            rs = ret_sum[(cid, k)]
            es = ev_sum[(cid, k)]
            ls = lat_by_ck.get((cid, k), {})
            topk_rows.append({
                "combo_id": cid, "profile_id": pid, "method": method, "k": k,
                "precision_at_k": rs["precision_at_k"],
                "pooled_recall_at_k": rs["pooled_recall_at_k"],
                "mrr_at_k": rs["mrr_at_k"],
                "ndcg_at_k": rs["ndcg_at_k"],
                "map_at_k": rs["map_at_k"],
                "f1_at_k": rs["f1_at_k"],
                "hit_rate_at_k": rs["hit_rate_at_k"],
                "evidence_recall_at_k": es["evidence_recall_at_k"],
                "evidence_precision_at_k": es["evidence_precision_at_k"],
                "complete_evidence_rate_at_k": es["complete_evidence_rate_at_k"],
                "evidence_redundancy_at_k": es["evidence_redundancy_at_k"],
                "mean_latency_ms": ls.get("mean_ms"),
                "p50_latency_ms": ls.get("p50_ms"),
                "p95_latency_ms": ls.get("p95_ms"),
            })
    _write_csv(sens_dir / "top_k_sensitivity.csv", topk_rows)

    # --- STEP 14: Error Analysis & 14-Category Failure Taxonomy ---
    chunk_diag_p = stage04_dir / "chunking/chunking_diagnostics.json"
    questions = [json.loads(l) for l in (root / "evaluation/datasets/questions_40.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    q_by_id = {q["query_id"]: q for q in questions}
    s4_rankings = [json.loads(l) for l in rankings_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rank_by_cq = {(r["combo_id"], r["query_id"]): r for r in s4_rankings}
    gt_a = {r["query_id"]: r for r in (json.loads(l) for l in (root / "evaluation/datasets/ground_truth_recursive.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    gt_b = {r["query_id"]: r for r in (json.loads(l) for l in (root / "evaluation/datasets/ground_truth_adaptive.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    err_records: list[dict[str, Any]] = []
    tax_counter: Counter[str] = Counter()
    for qid in query_ids_sorted:
        q = q_by_id[qid]
        k5 = {cid: merged_by_ck[(cid, 5)][qid] for cid, _, _ in STAGE04_COMBOS}
        k60 = {cid: merged_by_ck[(cid, 60)][qid] for cid, _, _ in STAGE04_COMBOS}
        ndcgs5 = {cid: _f(k5[cid]["ndcg_at_k"]) or 0.0 for cid, _, _ in STAGE04_COMBOS}
        evr5 = {cid: _f(k5[cid]["evidence_recall_at_k"]) or 0.0 for cid, _, _ in STAGE04_COMBOS}
        evr60 = {cid: _f(k60[cid]["evidence_recall_at_k"]) or 0.0 for cid, _, _ in STAGE04_COMBOS}

        pattern_tags: list[str] = []
        if min(ndcgs5.values()) >= 0.75 and min(evr5.values()) >= 0.99:
            pattern_tags.append("all_methods_strong")
        if max(ndcgs5.values()) < 0.50 or max(evr5.values()) < 0.50:
            pattern_tags.append("all_methods_weak")
        if ndcgs5["A4"] - ndcgs5["B4"] > 0.05:
            pattern_tags.append("recursive_better_rerank")
        elif ndcgs5["B4"] - ndcgs5["A4"] > 0.05:
            pattern_tags.append("adaptive_better_rerank")
        if ndcgs5["A1"] - ndcgs5["A2"] > 0.05 or ndcgs5["B1"] - ndcgs5["B2"] > 0.05:
            pattern_tags.append("bm25_better_than_dense")
        if ndcgs5["A2"] - ndcgs5["A1"] > 0.05 or ndcgs5["B2"] - ndcgs5["B1"] > 0.05:
            pattern_tags.append("dense_better_than_bm25")
        if ndcgs5["A3"] > max(ndcgs5["A1"], ndcgs5["A2"]) + 0.02 or ndcgs5["B3"] > max(ndcgs5["B1"], ndcgs5["B2"]) + 0.02:
            pattern_tags.append("hybrid_improves_over_singles")
        if ndcgs5["A4"] < ndcgs5["A3"] - 0.02 or ndcgs5["B4"] < ndcgs5["B3"] - 0.02:
            pattern_tags.append("reranker_degrades_ndcg5")

        failure_modes: list[dict[str, str]] = []
        for cid in ("A4", "B4"):
            if evr60[cid] < 1.0:
                failure_modes.append({"combo": cid, "category": "1_relevant_evidence_not_retrieved", "certainty": "verified"})
                failure_modes.append({"combo": cid, "category": "9_missing_knowledge_unit", "certainty": "verified"})
            elif evr5[cid] < 1.0:
                failure_modes.append({"combo": cid, "category": "2_relevant_evidence_ranked_too_low", "certainty": "verified"})
            if (_f(k5[cid]["evidence_redundancy_at_k"]) or 0.0) > 0.25:
                failure_modes.append({"combo": cid, "category": "8_duplicate_context", "certainty": "verified"})
            if (_f(k5[cid]["precision_at_k"]) or 0.0) <= 0.20 and evr5[cid] > 0.0:
                failure_modes.append({"combo": cid, "category": "10_irrelevant_context_overload", "certainty": "verified"})
        if ndcgs5["A1"] < 0.30 and ndcgs5["A2"] >= 0.50:
            failure_modes.append({"combo": "A1", "category": "3_lexical_mismatch", "certainty": "plausible"})
        if ndcgs5["A2"] < 0.30 and ndcgs5["A1"] >= 0.50:
            failure_modes.append({"combo": "A2", "category": "4_semantic_mismatch", "certainty": "plausible"})
        if int(q["required_knowledge_units"] and len(q["required_knowledge_units"])) >= 3 and (evr5["B4"] < evr5["A4"] - 0.1):
            failure_modes.append({"combo": "B4", "category": "7_fragmented_evidence", "certainty": "plausible"})

        for fm in failure_modes:
            tax_counter[fm["category"]] += 1

        err_records.append({
            "query_id": qid,
            "query": q["query"],
            "topic": q["topic"],
            "language": q["language"],
            "difficulty": q["difficulty"],
            "question_type": q["question_type"],
            "source_references": q["source_document_references"],
            "ndcg_at_5": {k: round(v, 4) for k, v in ndcgs5.items()},
            "evidence_recall_at_5": {k: round(v, 4) for k, v in evr5.items()},
            "evidence_recall_at_60": {k: round(v, 4) for k, v in evr60.items()},
            "a4_top5_chunks": [h["chunk_id"] for h in rank_by_cq[("A4", qid)]["results"][:5]],
            "a4_top5_grades": [gt_a[qid]["relevance_grades"].get(h["chunk_id"]) for h in rank_by_cq[("A4", qid)]["results"][:5]],
            "b4_top5_chunks": [h["chunk_id"] for h in rank_by_cq[("B4", qid)]["results"][:5]],
            "b4_top5_grades": [gt_b[qid]["relevance_grades"].get(h["chunk_id"]) for h in rank_by_cq[("B4", qid)]["results"][:5]],
            "pattern_tags": pattern_tags,
            "failure_modes": failure_modes,
        })

    (err_dir / "error_analysis.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in err_records) + "\n",
        encoding="utf-8",
    )
    tax_rows = [{"failure_category": k, "occurrence_count": v} for k, v in sorted(tax_counter.items())]
    _write_csv(err_dir / "failure_taxonomy_summary.csv", tax_rows)

    case_studies = {
        tag: [r for r in err_records if tag in r["pattern_tags"]][:3]
        for tag in (
            "all_methods_strong", "all_methods_weak", "recursive_better_rerank",
            "adaptive_better_rerank", "bm25_better_than_dense", "dense_better_than_bm25",
            "hybrid_improves_over_singles", "reranker_degrades_ndcg5",
        )
    }
    (err_dir / "representative_case_studies.json").write_text(json.dumps(case_studies, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- STEP 15: Quality-Latency-Resource Trade-off & Pareto Frontier ---
    res_by_combo = {r["combo_id"]: r for r in res_sum}
    gen_sum = {r["combo_id"]: r for r in _read_csv(stage04_dir / "generation/answer_metrics_summary.csv")}

    trade_rows: list[dict[str, Any]] = []
    for k in (5, 15, 60):
        for cid, pid, method in STAGE04_COMBOS:
            rs = ret_sum[(cid, k)]
            es = ev_sum[(cid, k)]
            ls = lat_by_ck.get((cid, k), {})
            ru = res_by_combo.get(cid, {})
            gs = gen_sum.get(cid, {})
            trade_rows.append({
                "combo_id": cid, "profile_id": pid, "method": method, "k": k,
                "compatibility_note": "Top60_Pool120_prefix_quality_vs_actual_K_Pool120_latency",
                "ndcg_at_k": float(rs["ndcg_at_k"]),
                "precision_at_k": float(rs["precision_at_k"]),
                "pooled_recall_at_k": float(rs["pooled_recall_at_k"]),
                "mrr_at_k": float(rs["mrr_at_k"]),
                "evidence_recall_at_k": float(es["evidence_recall_at_k"]),
                "complete_evidence_rate_at_k": float(es["complete_evidence_rate_at_k"]),
                "answer_correctness_norm": _f(gs.get("answer_correctness_norm")),
                "mean_latency_ms": _f(ls.get("mean_ms")) or 0.0,
                "p50_latency_ms": _f(ls.get("p50_ms")) or 0.0,
                "p95_latency_ms": _f(ls.get("p95_ms")) or 0.0,
                "retrieved_emb_tokens_mean": ru.get(f"retrieved_emb_tokens_k{k}_mean"),
                "peak_gpu_allocated_max_mb": ru.get("peak_gpu_allocated_max_mb"),
                "process_rss_mean_mb": ru.get("process_rss_mean_mb"),
            })
    _write_csv(trade_dir / "quality_latency_tradeoff.csv", trade_rows)

    pareto_all: list[dict[str, Any]] = []
    for k in (5, 15, 60):
        sub = [r for r in trade_rows if r["k"] == k]
        pareto_all.extend(
            compute_pareto_frontier(
                sub,
                maximize_keys=("ndcg_at_k", "evidence_recall_at_k"),
                minimize_keys=("p95_latency_ms",),
            )
        )
    _write_csv(trade_dir / "pareto_frontier.csv", pareto_all)

    k5_rows = [r for r in pareto_all if r["k"] == 5]
    k15_rows = [r for r in pareto_all if r["k"] == 15]
    recs = {
        "methodology_note": "Objective multi-criteria summary at K=5 and K=15 (candidate_top_k=120); no arbitrary composite weights. Exploratory selection on benchmark set (not held-out validated).",
        "best_retrieval_ndcg_k5": max(k5_rows, key=lambda r: r["ndcg_at_k"])["combo_id"],
        "best_evidence_recall_k5": max(k5_rows, key=lambda r: r["evidence_recall_at_k"])["combo_id"],
        "best_retrieval_ndcg_k15": max(k15_rows, key=lambda r: r["ndcg_at_k"])["combo_id"],
        "best_evidence_recall_k15": max(k15_rows, key=lambda r: r["evidence_recall_at_k"])["combo_id"],
        "lowest_p95_latency_k5": min(k5_rows, key=lambda r: r["p95_latency_ms"])["combo_id"],
        "pareto_optimal_k5": [r["combo_id"] for r in k5_rows if r["is_pareto_optimal"]],
        "pareto_optimal_k15": [r["combo_id"] for r in k15_rows if r["is_pareto_optimal"]],
    }
    if any(r["answer_correctness_norm"] is not None for r in k5_rows):
        recs["best_answer_correctness"] = max(
            [r for r in k5_rows if r["answer_correctness_norm"] is not None],
            key=lambda r: r["answer_correctness_norm"],
        )["combo_id"]
    (trade_dir / "configuration_recommendations.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"recommendations": recs, "pareto_k5": recs["pareto_optimal_k5"], "pareto_k15": recs["pareto_optimal_k15"]}
