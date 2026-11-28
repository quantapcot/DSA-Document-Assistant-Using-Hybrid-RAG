"""Blinded judging input preparation, metric aggregation, token accounting, and GATE-01..06 assessment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.ac04_completion import (
    PROTOCOL_HTTP_CONFIG_A,
    PROTOCOL_HTTP_CONFIG_B,
    PROTOCOL_LIVE_GROQ_NO_RAG,
    artifacts_dir,
)
from evaluation.ac04_completion.groq_smoke import verify_original_qdrant_untouched

_CIT_RE = re.compile(r"\[([1-9][0-9]{0,2})\]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _norm_0_4(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return round(max(0.0, min(4.0, v)) / 4.0, 4)


def _compute_citation_validity(answer_text: str, citations: list[dict[str, Any]], has_context: bool) -> float | None:
    if not has_context:
        return None
    mentioned = [int(m.group(1)) for m in _CIT_RE.finditer(answer_text or "")]
    if not mentioned:
        return 0.0 if citations else None
    max_lbl = max([int(c.get("label") or 0) for c in citations] + [len(citations), 5])
    valid = sum(1 for lbl in mentioned if 1 <= lbl <= max_lbl)
    return round(valid / len(mentioned), 4)


def prepare_judging_inputs(root: Path) -> dict[str, Any]:
    """Prepare blinded judging batches using ONLY verbatim captured context (never reconstructed)."""
    out_dir = artifacts_dir(root)
    qmap = {
        q["query_id"]: q
        for q in _read_jsonl(root / "evaluation" / "datasets" / "questions_40.jsonl")
    }

    chat_items: list[dict[str, Any]] = []
    for fname, proto in [
        ("http_chat_config_a.jsonl", PROTOCOL_HTTP_CONFIG_A),
        ("http_chat_config_b.jsonl", PROTOCOL_HTTP_CONFIG_B),
        ("live_groq_no_rag.jsonl", PROTOCOL_LIVE_GROQ_NO_RAG),
    ]:
        for row in _read_jsonl(out_dir / fname):
            qid = row["query_id"]
            q = qmap[qid]
            is_norag = proto == PROTOCOL_LIVE_GROQ_NO_RAG
            pt = row.get("provider_telemetry") or {}
            verbatim_ctx = "" if is_norag else str(pt.get("actual_context_sent") or "")
            has_ctx = (not is_norag) and bool(verbatim_ctx.strip())
            item_key = f"{proto}__{qid}"
            blind_id = "BL_" + hashlib.sha256(item_key.encode("utf-8")).hexdigest()[:12]
            chat_items.append(
                {
                    "blind_id": blind_id,
                    "item_key": item_key,
                    "protocol": proto,
                    "query_id": qid,
                    "query": q["query"],
                    "language": q.get("language"),
                    "reference_answer": q.get("reference_answer", ""),
                    "knowledge_units": q.get("required_knowledge_units", []),
                    "generation_mode": "no_rag" if is_norag else row.get("generation_mode"),
                    "has_verbatim_context": has_ctx,
                    "verbatim_context": verbatim_ctx[:14000],
                    "answer": row.get("answer", ""),
                    "citations": row.get("citations", []),
                }
            )

    chat_in_path = out_dir / "judge_inputs_chat.jsonl"
    with chat_in_path.open("w", encoding="utf-8") as f:
        for it in chat_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Robustness judging inputs
    rob_items: list[dict[str, Any]] = []
    for row in _read_jsonl(out_dir / "http_robustness.jsonl"):
        linked_qid = row.get("linked_query_id")
        ref_ans = qmap[linked_qid]["reference_answer"] if linked_qid and linked_qid in qmap else ""
        pt = row.get("provider_telemetry") or {}
        rob_items.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "expected_behavior": row["expected_behavior"],
                "injection_surface": row.get("injection_surface", "none"),
                "forbidden_token": row.get("forbidden_token"),
                "query_sent": row["query_sent"],
                "reference_answer": ref_ans,
                "generation_mode": row.get("generation_mode"),
                "has_verbatim_context": bool(pt.get("has_verbatim_context")),
                "verbatim_context": str(pt.get("actual_context_sent") or "")[:10000],
                "answer": row.get("answer", ""),
            }
        )
    rob_in_path = out_dir / "judge_inputs_robustness.jsonl"
    with rob_in_path.open("w", encoding="utf-8") as f:
        for it in rob_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Study judging inputs (exclude raw validation 422 probe from content judging, keep 8 quizzes + 4 roadmaps)
    study_items: list[dict[str, Any]] = []
    for row in _read_jsonl(out_dir / "http_quiz.jsonl"):
        if row.get("is_raw_validation_probe"):
            continue
        pt = row.get("provider_telemetry") or {}
        study_items.append(
            {
                "feature": "quiz",
                "item_id": row["case_id"],
                "topic": row["sent_topic"],
                "original_topic": row["original_topic"],
                "generation_mode": row.get("generation_mode"),
                "generated_count": row.get("generated_count", 0),
                "has_verbatim_context": bool(pt.get("has_verbatim_context")),
                "verbatim_context": str(pt.get("actual_context_sent") or "")[:12000],
                "content": row.get("questions", []),
            }
        )
    for row in _read_jsonl(out_dir / "http_roadmap.jsonl"):
        pt = row.get("provider_telemetry") or {}
        study_items.append(
            {
                "feature": "roadmap",
                "item_id": row["scenario_id"],
                "topic": row["goal"],
                "original_topic": row["goal"],
                "generation_mode": row.get("generation_mode"),
                "generated_count": row.get("items_count", 0),
                "has_verbatim_context": bool(pt.get("has_verbatim_context")),
                "verbatim_context": str(pt.get("actual_context_sent") or "")[:12000],
                "content": row.get("items", []),
            }
        )
    study_in_path = out_dir / "judge_inputs_study.jsonl"
    with study_in_path.open("w", encoding="utf-8") as f:
        for it in study_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        "chat_items": len(chat_items),
        "robustness_items": len(rob_items),
        "study_items": len(study_items),
    }


def _latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None, "std_ms": None}
    arr = np.array(values, dtype=float)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "std_ms": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
    }


def aggregate_all_ac04_metrics(root: Path) -> dict[str, Any]:
    """Aggregate all live Groq & HTTP E2E results, judgments, latency, tokens, and GATE-01..06."""
    out_dir = artifacts_dir(root)
    smoke = json.loads((out_dir / "smoke_test_report.json").read_text(encoding="utf-8"))

    cfg_a = _read_jsonl(out_dir / "http_chat_config_a.jsonl")
    cfg_b = _read_jsonl(out_dir / "http_chat_config_b.jsonl")
    no_rag = _read_jsonl(out_dir / "live_groq_no_rag.jsonl")
    edge_rows = _read_jsonl(out_dir / "http_edge_cases.jsonl")
    rob_rows = _read_jsonl(out_dir / "http_robustness.jsonl")
    quiz_all = _read_jsonl(out_dir / "http_quiz.jsonl")
    rm_rows = _read_jsonl(out_dir / "http_roadmap.jsonl")

    chat_judgments = {j["blind_id"]: j for j in _read_jsonl(out_dir / "answer_judgments_live.jsonl")}
    chat_inputs = {i["blind_id"]: i for i in _read_jsonl(out_dir / "judge_inputs_chat.jsonl")}
    rob_judgments = {j["case_id"]: j for j in _read_jsonl(out_dir / "robustness_judgments_live.jsonl")}
    study_judgments = {j["item_id"]: j for j in _read_jsonl(out_dir / "study_judgments_live.jsonl")}
    sec_audit_rows = _read_jsonl(out_dir / "secondary_audit_live.jsonl")

    # 1. Per-query judged table for Chat (Config A, Config B, Live NO_RAG)
    pq_rows: list[dict[str, Any]] = []
    raw_by_key: dict[str, dict[str, Any]] = {}
    for proto, rows in [
        (PROTOCOL_HTTP_CONFIG_A, cfg_a),
        (PROTOCOL_HTTP_CONFIG_B, cfg_b),
        (PROTOCOL_LIVE_GROQ_NO_RAG, no_rag),
    ]:
        for r in rows:
            raw_by_key[f"{proto}__{r['query_id']}"] = r

    for bid, inp in chat_inputs.items():
        j = chat_judgments.get(bid, {})
        raw = raw_by_key.get(inp["item_key"], {})
        has_ctx = bool(inp["has_verbatim_context"])
        is_norag = inp["protocol"] == PROTOCOL_LIVE_GROQ_NO_RAG
        total_kus = len(inp.get("knowledge_units", [])) or 1
        cov_kus = j.get("covered_ku_ids")
        completeness = (
            round(min(len(cov_kus), total_kus) / total_kus, 4)
            if isinstance(cov_kus, list)
            else j.get("answer_completeness_norm")
        )
        cit_valid = None if (is_norag or not has_ctx) else _compute_citation_validity(
            inp["answer"], inp.get("citations", []), has_ctx
        )
        pq_rows.append(
            {
                "blind_id": bid,
                "protocol": inp["protocol"],
                "query_id": inp["query_id"],
                "generation_mode": inp["generation_mode"],
                "has_verbatim_context": has_ctx,
                "http_transport_ok": raw.get("http_transport_ok", raw.get("http_status") == 200),
                "live_llm_generated": raw.get("live_llm_generated", False),
                "answer_correctness_norm": _norm_0_4(j.get("answer_correctness_raw")),
                "answer_relevancy_norm": _norm_0_4(j.get("answer_relevancy_raw")),
                "faithfulness_norm": _norm_0_4(j.get("faithfulness_raw")) if has_ctx else None,
                "answer_completeness_norm": completeness,
                "citation_accuracy_norm": _norm_0_4(j.get("citation_accuracy_raw")) if has_ctx else None,
                "citation_validity": cit_valid,
                "unsupported_claim_rate": j.get("unsupported_claim_rate") if has_ctx else None,
                "contradiction_rate": j.get("contradiction_rate"),
                "complexity_correctness_norm": _norm_0_4(j.get("complexity_correctness_raw")),
                "code_correctness_norm": _norm_0_4(j.get("code_correctness_raw")),
            }
        )

    pq_df = pd.DataFrame(pq_rows)
    pq_df.to_csv(out_dir / "answer_metrics_per_query_live.csv", index=False, encoding="utf-8")

    # Summary CSV (Overall per protocol + Stratified subset for live_llm_generated rows!)
    metric_cols = [
        "answer_correctness_norm", "answer_relevancy_norm", "faithfulness_norm",
        "answer_completeness_norm", "citation_accuracy_norm", "citation_validity",
        "unsupported_claim_rate", "contradiction_rate", "complexity_correctness_norm", "code_correctness_norm",
    ]

    def _agg_slice(df_slice: pd.DataFrame, proto_label: str, subset_label: str) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "protocol": proto_label,
            "subset": subset_label,
            "n_total": int(len(df_slice)),
            "n_live_llm": int(df_slice["live_llm_generated"].sum()) if len(df_slice) else 0,
            "n_extractive": int((df_slice["generation_mode"] == "extractive").sum()) if len(df_slice) else 0,
            "n_abstained": int((df_slice["generation_mode"] == "abstained").sum()) if len(df_slice) else 0,
            "n_with_verbatim_context": int(df_slice["has_verbatim_context"].sum()) if len(df_slice) else 0,
        }
        for col in metric_cols:
            vals = df_slice[col].dropna()
            rec[col] = "N/A" if len(vals) == 0 else round(float(vals.mean()), 4)
            rec[f"{col}_n"] = int(len(vals))
        return rec

    ans_sum_rows: list[dict[str, Any]] = []
    for proto in [PROTOCOL_HTTP_CONFIG_A, PROTOCOL_HTTP_CONFIG_B, PROTOCOL_LIVE_GROQ_NO_RAG]:
        sub = pq_df[pq_df["protocol"] == proto]
        ans_sum_rows.append(_agg_slice(sub, proto, "all_40_queries"))
        if proto != PROTOCOL_LIVE_GROQ_NO_RAG:
            ans_sum_rows.append(_agg_slice(sub[sub["live_llm_generated"] == True], proto, "live_llm_only"))  # noqa: E712
            ans_sum_rows.append(_agg_slice(sub[sub["has_verbatim_context"] == True], proto, "with_verbatim_context"))  # noqa: E712

    ans_sum_df = pd.DataFrame(ans_sum_rows)
    ans_sum_df.to_csv(out_dir / "answer_quality_summary.csv", index=False, encoding="utf-8")

    # 2. HTTP Latency & Reliability Summary (Condition #7 & #10 strict separation)
    quiz_eval = [r for r in quiz_all if not r.get("is_raw_validation_probe")]
    quiz_raw_422 = [r for r in quiz_all if r.get("is_raw_validation_probe")]

    def _lat_row(group_name: str, endpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        c_ms = [float(r["client_elapsed_ms"]) for r in rows if r.get("client_elapsed_ms") is not None]
        ret_ms = [float((r.get("server_telemetry") or {}).get("retrieval_ms", 0.0)) for r in rows]
        llm_ms = [float((r.get("server_telemetry") or {}).get("llm_ms_total", 0.0)) for r in rows]
        # For direct NO_RAG, llm_ms IS client_elapsed_ms and retrieval_ms = 0
        if endpoint == "direct_groq_api":
            llm_ms = c_ms
            ret_ms = [0.0 for _ in rows]
        overhead_ms = [max(0.0, c - r - l) for c, r, l in zip(c_ms, ret_ms, llm_ms)]

        http_ok = sum(1 for r in rows if r.get("http_status") == 200 and not r.get("client_error"))
        err_cnt = sum(1 for r in rows if r.get("http_status") != 200 or r.get("client_error") or r.get("generation_error"))
        to_cnt = sum(1 for r in rows if r.get("timed_out") or r.get("http_status") == 504)
        prov_calls = sum(int((r.get("provider_telemetry") or {}).get("provider_call_count", 0)) for r in rows)
        prov_retries = sum(int((r.get("server_telemetry") or {}).get("provider_retries", 0)) for r in rows) + sum(
            int(r.get("client_retries_due_to_tpm", 0)) for r in rows
        )
        rl429 = sum(int((r.get("provider_telemetry") or {}).get("provider_rate_limit_429_calls", 0)) for r in rows)
        live_llm = sum(1 for r in rows if r.get("live_llm_generated") or (r.get("generation_mode") == "llm" and not r.get("generation_error")))
        extractive = sum(1 for r in rows if r.get("generation_mode") == "extractive")
        abstained = sum(1 for r in rows if r.get("generation_mode") == "abstained")

        st = _latency_stats(c_ms)
        return {
            "experiment_group": group_name,
            "endpoint": endpoint,
            "measured_requests": len(rows),
            "provider_api_calls": prov_calls,
            "http_200_count": http_ok,
            "http_success_rate": round(http_ok / len(rows), 4) if rows else 0.0,
            "live_llm_count": live_llm,
            "extractive_fallback_count": extractive,
            "abstained_count": abstained,
            "error_or_gen_error_count": err_cnt,
            "error_rate": round(err_cnt / len(rows), 4) if rows else 0.0,
            "timeout_count": to_cnt,
            "timeout_rate": round(to_cnt / len(rows), 4) if rows else 0.0,
            "retry_count": prov_retries,
            "rate_limit_429_count": rl429,
            **st,
            "retrieval_mean_ms": round(float(np.mean(ret_ms)), 2) if ret_ms else 0.0,
            "llm_generation_mean_ms": round(float(np.mean(llm_ms)), 2) if llm_ms else 0.0,
            "http_overhead_mean_ms": round(float(np.mean(overhead_ms)), 2) if overhead_ms else 0.0,
        }

    lat_rows = [
        _lat_row("HTTP_Config_A_pool10", "/api/chat", cfg_a),
        _lat_row("HTTP_Config_B_pool30", "/api/chat", cfg_b),
        _lat_row("Direct_Live_Groq_NO_RAG", "direct_groq_api", no_rag),
        _lat_row("HTTP_Robustness_20", "/api/chat", rob_rows),
        _lat_row("HTTP_Quiz_8_Topics", "/api/quiz", quiz_eval),
        _lat_row("HTTP_Roadmap_4_Scenarios", "/api/roadmap", rm_rows),
    ]
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(out_dir / "http_latency_summary.csv", index=False, encoding="utf-8")

    # 3. Token Usage Summary (Condition #2: ACTUAL provider calls & tokens across ALL runs including smoke!)
    def _tok_row(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        p_calls = sum(int((r.get("provider_telemetry") or {}).get("provider_call_count", 0)) for r in rows)
        p_ok = sum(int((r.get("provider_telemetry") or {}).get("provider_success_calls", 0)) for r in rows)
        p_err = sum(int((r.get("provider_telemetry") or {}).get("provider_error_calls", 0)) for r in rows)
        p_429 = sum(int((r.get("provider_telemetry") or {}).get("provider_rate_limit_429_calls", 0)) for r in rows)
        ptok = sum(int((r.get("provider_telemetry") or {}).get("prompt_tokens", 0)) for r in rows)
        ctok = sum(int((r.get("provider_telemetry") or {}).get("completion_tokens", 0)) for r in rows)
        ttok = sum(int((r.get("provider_telemetry") or {}).get("total_tokens", 0)) for r in rows)
        return {
            "experiment_group": name,
            "http_or_task_requests": len(rows),
            "actual_provider_calls": p_calls,
            "provider_success_calls": p_ok,
            "provider_error_calls": p_err,
            "rate_limit_429_calls": p_429,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "total_tokens": ttok,
            "mean_tokens_per_provider_call": round(ttok / p_ok, 1) if p_ok else 0.0,
        }

    smoke_u = smoke["groq_smoke_test"].get("usage") or {}
    smoke_3q = _read_jsonl(out_dir / "http_smoke_3q.jsonl")
    tok_rows = [
        {
            "experiment_group": "Connectivity_Smoke_1q",
            "http_or_task_requests": 1,
            "actual_provider_calls": 1,
            "provider_success_calls": 1 if smoke["groq_smoke_test"].get("connectivity_ok") else 0,
            "provider_error_calls": 0 if smoke["groq_smoke_test"].get("connectivity_ok") else 1,
            "rate_limit_429_calls": 0,
            "prompt_tokens": int(smoke_u.get("prompt_tokens") or 0),
            "completion_tokens": int(smoke_u.get("completion_tokens") or 0),
            "total_tokens": int(smoke_u.get("total_tokens") or 0),
            "mean_tokens_per_provider_call": float(smoke_u.get("total_tokens") or 0),
        },
        _tok_row("HTTP_Smoke_3q", smoke_3q),
        _tok_row("HTTP_Config_A_pool10_40q", cfg_a),
        _tok_row("HTTP_Config_B_pool30_40q", cfg_b),
        _tok_row("Direct_Live_Groq_NO_RAG_40q", no_rag),
        _tok_row("HTTP_Robustness_20", rob_rows),
        _tok_row("HTTP_Quiz_8_Topics", quiz_all),
        _tok_row("HTTP_Roadmap_4_Scenarios", rm_rows),
    ]
    tok_df = pd.DataFrame(tok_rows)
    total_row = {
        "experiment_group": "TOTAL_ALL_LIVE_GROQ",
        "http_or_task_requests": int(tok_df["http_or_task_requests"].sum()),
        "actual_provider_calls": int(tok_df["actual_provider_calls"].sum()),
        "provider_success_calls": int(tok_df["provider_success_calls"].sum()),
        "provider_error_calls": int(tok_df["provider_error_calls"].sum()),
        "rate_limit_429_calls": int(tok_df["rate_limit_429_calls"].sum()),
        "prompt_tokens": int(tok_df["prompt_tokens"].sum()),
        "completion_tokens": int(tok_df["completion_tokens"].sum()),
        "total_tokens": int(tok_df["total_tokens"].sum()),
        "mean_tokens_per_provider_call": round(
            float(tok_df["total_tokens"].sum()) / max(1, int(tok_df["provider_success_calls"].sum())), 1
        ),
    }
    tok_df = pd.concat([tok_df, pd.DataFrame([total_row])], ignore_index=True)
    tok_df.to_csv(out_dir / "token_usage_summary.csv", index=False, encoding="utf-8")

    # 4. Robustness Summary (Live HTTP)
    rob_cat_rows: list[dict[str, Any]] = []
    for cat in ["out_of_domain", "unanswerable_in_corpus", "prompt_injection", "distractor_context", "semantic_paraphrase", "typo_noisy"]:
        sub_r = [r for r in rob_rows if r["category"] == cat]
        n = len(sub_r)
        judged_r = [rob_judgments.get(r["case_id"], {}) for r in sub_r]
        handled = sum(1 for j in judged_r if j.get("handled_correctly") is True)
        abstained_c = sum(1 for r in sub_r if r.get("abstained"))
        llm_c = sum(1 for r in sub_r if r.get("live_llm_generated"))
        ext_c = sum(1 for r in sub_r if r.get("extractive_fallback"))
        qual_vals = [_norm_0_4(j.get("answer_quality_0_4")) for j in judged_r]
        qual_vals = [v for v in qual_vals if v is not None]
        inj_vals = [r.get("injection_token_absent") for r in sub_r if r.get("injection_token_absent") is not None]
        rob_cat_rows.append(
            {
                "category": cat,
                "n_cases": n,
                "injection_surface": "query_level_http_input" if cat == "prompt_injection" else "none",
                "http_200_rate": round(sum(1 for r in sub_r if r.get("http_transport_ok")) / n, 4) if n else 0.0,
                "live_llm_count": llm_c,
                "extractive_count": ext_c,
                "abstained_count": abstained_c,
                "correct_handling_rate": round(handled / n, 4) if n else 0.0,
                "abstention_rate": round(abstained_c / n, 4) if n else 0.0,
                "injection_resistance_rate": round(sum(1 for x in inj_vals if x) / len(inj_vals), 4) if inj_vals else "N/A",
                "mean_answer_quality_norm": round(float(np.mean(qual_vals)), 4) if qual_vals else "N/A",
                "mean_http_ms": round(float(np.mean([r["client_elapsed_ms"] for r in sub_r])), 2) if n else 0.0,
            }
        )
    pd.DataFrame(rob_cat_rows).to_csv(out_dir / "robustness_summary_live.csv", index=False, encoding="utf-8")

    # 5. Study Features Summary (Live HTTP Quiz n=8 + Roadmap n=4)
    def _study_agg(feat: str, rows: list[dict[str, Any]], id_col: str) -> dict[str, Any]:
        n = len(rows)
        j_list = [study_judgments.get(r[id_col], {}) for r in rows]
        corr = [_norm_0_4(j.get("content_correctness_raw")) for j in j_list]
        grd = [_norm_0_4(j.get("source_grounding_raw")) for j in j_list]
        ped = [_norm_0_4(j.get("pedagogical_quality_raw")) for j in j_list]
        corr = [v for v in corr if v is not None]
        grd = [v for v in grd if v is not None]
        ped = [v for v in ped if v is not None]
        gen_ok = sum(1 for r in rows if r.get("generation_mode") == "llm" and not r.get("generation_error") and (r.get("generated_count", 0) > 0 or r.get("items_count", 0) > 0))
        any_gen = sum(1 for r in rows if r.get("generation_mode") == "llm" and (r.get("generated_count", 0) > 0 or r.get("items_count", 0) > 0))
        return {
            "feature": feat,
            "scenarios": n,
            "http_200_count": sum(1 for r in rows if r.get("http_transport_ok")),
            "any_live_llm_output_count": any_gen,
            "functional_live_llm_success_count": gen_ok,
            "schema_valid_rate": round(sum(1 for r in rows if r.get("schema_valid")) / n, 4) if n else 0.0,
            "content_correctness_norm": round(float(np.mean(corr)), 4) if corr else "N/A",
            "source_grounding_norm": round(float(np.mean(grd)), 4) if grd else "N/A",
            "pedagogical_quality_norm": round(float(np.mean(ped)), 4) if ped else "N/A",
            "mean_duplicate_rate": round(float(np.mean([r.get("duplicate_rate", 0.0) for r in rows])), 4) if n else 0.0,
            "mean_http_ms": round(float(np.mean([r["client_elapsed_ms"] for r in rows])), 2) if n else 0.0,
            "p95_http_ms": round(float(np.percentile([r["client_elapsed_ms"] for r in rows], 95)), 2) if n else 0.0,
        }

    study_df = pd.DataFrame(
        [
            _study_agg("live_http_quiz", quiz_eval, "case_id"),
            _study_agg("live_http_roadmap", rm_rows, "scenario_id"),
        ]
    )
    study_df.to_csv(out_dir / "study_features_summary_live.csv", index=False, encoding="utf-8")

    # 6. Edge Cases Summary
    edge_df = pd.DataFrame(
        [
            {
                "case_id": r["case_id"],
                "endpoint": r["endpoint"],
                "expected_status": r.get("expected_status", "client_timeout"),
                "http_status": r.get("http_status"),
                "passed": r.get("passed"),
                "client_elapsed_ms": r.get("client_elapsed_ms"),
                "generation_mode": r.get("generation_mode"),
            }
            for r in edge_rows
        ]
    )
    edge_df.to_csv(out_dir / "edge_cases_summary.csv", index=False, encoding="utf-8")

    # 7. Secondary Judge Audit Agreement
    agree_corr = []
    agree_faith = []
    for sa in sec_audit_rows:
        p = chat_judgments.get(sa["blind_id"], {})
        if p.get("answer_correctness_raw") is not None and sa.get("answer_correctness_raw") is not None:
            agree_corr.append(abs(float(p["answer_correctness_raw"]) - float(sa["answer_correctness_raw"])) <= 1.0)
        if p.get("faithfulness_raw") is not None and sa.get("faithfulness_raw") is not None:
            if float(p["faithfulness_raw"]) >= 0 and float(sa["faithfulness_raw"]) >= 0:
                agree_faith.append(abs(float(p["faithfulness_raw"]) - float(sa["faithfulness_raw"])) <= 1.0)

    audit_summary = {
        "human_validated": False,
        "human_approval_status": "DRAFT",
        "primary_judge_model": "antigravity_agent_pro",
        "secondary_audit_model": "antigravity_agent_flash",
        "stage04_synthetic_controls_retained": "3/3 PASS",
        "total_chat_judgments": len(chat_judgments),
        "total_robustness_judgments": len(rob_judgments),
        "total_study_judgments": len(study_judgments),
        "secondary_audit_sample_size": len(sec_audit_rows),
        "correctness_within_1pt_agreement_pct": round(100.0 * sum(agree_corr) / len(agree_corr), 2) if agree_corr else None,
        "faithfulness_within_1pt_agreement_pct": round(100.0 * sum(agree_faith) / len(agree_faith), 2) if agree_faith else None,
    }
    (out_dir / "answer_quality_audit_live.json").write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 8. Objective GATE-01..06 Evaluation
    orig_qdrant_untouched = verify_original_qdrant_untouched(
        root, smoke["isolated_qdrant_copy"]["source_checksums"]
    )
    g1 = bool(smoke["groq_smoke_test"].get("connectivity_ok")) and smoke["groq_smoke_test"].get("returned_model") == "openai/gpt-oss-120b"
    g2 = (
        len(cfg_a) == 40
        and len(cfg_b) == 40
        and len(no_rag) == 40
        and len(chat_judgments) == 120
        and sum(1 for r in cfg_a if r.get("live_llm_generated")) > 0
        and sum(1 for r in cfg_b if r.get("live_llm_generated")) > 0
    )
    g3 = (
        len(cfg_a) == 40
        and len(cfg_b) == 40
        and len(edge_rows) == 9
        and all(bool(r.get("passed")) for r in edge_rows)
        and len(rob_rows) == 20
    )
    g4 = (
        len(quiz_eval) == 8
        and len(quiz_raw_422) == 1
        and quiz_raw_422[0].get("http_status") == 422
        and len(rm_rows) == 4
        and len(study_judgments) == 12
    )
    g5 = (
        len(rob_judgments) == 20
        and len(sec_audit_rows) >= 12
        and orig_qdrant_untouched
        and total_row["actual_provider_calls"] > 0
    )

    gates = {
        "GATE-01": {"name": "Live Groq connectivity verified", "status": "PASS" if g1 else "FAIL"},
        "GATE-02": {"name": "Actual application Answer Generation evaluated (40q Config A + 40q Config B + 40q NO_RAG)", "status": "PASS" if g2 else "PARTIAL"},
        "GATE-03": {"name": "Actual HTTP End-to-End evaluated (Chat + Edge Cases + 504 Timeout + 20 Robustness)", "status": "PASS" if g3 else "PARTIAL"},
        "GATE-04": {"name": "Real Quiz (8 topics + raw 422) and Roadmap (4 scenarios) API evaluated", "status": "PASS" if g4 else "PARTIAL"},
        "GATE-05": {"name": "Robustness, answer quality, latency & token tracking valid and reproducible", "status": "PASS" if g5 else "PARTIAL"},
        "GATE-06": {"name": "Supplementary artifacts, 6 figures, reports, regression tests & integrity checks", "status": "PENDING_FIGURES_AND_TESTS"},
    }

    assessment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_qdrant_eval_untouched": orig_qdrant_untouched,
        "historical_stage04_ac04_status": "PARTIAL",
        "gates": gates,
        "audit_summary": audit_summary,
        "total_tokens_summary": total_row,
    }
    (out_dir / "ac04_gates_assessment.json").write_text(
        json.dumps(assessment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return assessment
