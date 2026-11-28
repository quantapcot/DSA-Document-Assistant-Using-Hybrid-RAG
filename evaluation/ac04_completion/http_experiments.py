"""Real HTTP End-to-End & Live Groq experiment runners with TPM-aware pacing and per-item checkpointing."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evaluation.ac04_completion import (
    DEFAULT_PORT,
    EMPTY_CORPUS_ID,
    EMPTY_WORKSPACE_ID,
    EVAL_CORPUS_RECURSIVE,
    EVAL_WORKSPACE_ID,
    PROTOCOL_HTTP_CONFIG_A,
    PROTOCOL_HTTP_CONFIG_B,
    PROTOCOL_LIVE_GROQ_NO_RAG,
    TIMEOUT_TEST_PORT,
    artifacts_dir,
)
from evaluation.ac04_completion.server_runner import IsolatedServerHandle, run_isolated_backend

_DURATION_RE = re.compile(r"(?:(\d+)m)?(\d+(?:\.\d+)?)s")


def _parse_reset_seconds(val: str | None) -> float:
    if not val:
        return 0.0
    m = _DURATION_RE.fullmatch(val.strip())
    if not m:
        try:
            return float(val)
        except ValueError:
            return 0.0
    mins = float(m.group(1) or 0)
    secs = float(m.group(2) or 0)
    return mins * 60.0 + secs


def _pace_for_tpm(traces: list[dict[str, Any]], *, min_remaining_tokens: int = 3200, min_floor_sec: float = 1.0) -> float:
    """Inspect latest Groq x-ratelimit-* headers and sleep just enough to avoid 429 TPM exhaustion."""
    if not traces:
        return 0.0
    last = traces[-1]
    hdrs = last.get("rate_limit_headers") or {}
    rem_tok = hdrs.get("x-ratelimit-remaining-tokens")
    reset_tok = hdrs.get("x-ratelimit-reset-tokens")
    retry_after = hdrs.get("retry-after")
    sleep_sec = 0.0
    if retry_after:
        sleep_sec = max(sleep_sec, _parse_reset_seconds(str(retry_after)) + 1.5)
    if rem_tok is not None:
        try:
            rem_int = int(rem_tok)
            if rem_int < min_remaining_tokens:
                # Wait proportionally or until reset window refreshes
                rst = _parse_reset_seconds(str(reset_tok)) if reset_tok else 25.0
                # If reset is tiny (< 5s) but remaining is very low, give bucket extra refill margin
                needed_tokens = max(0, min_remaining_tokens - rem_int)
                refill_sec = (needed_tokens / 8000.0) * 60.0
                sleep_sec = max(sleep_sec, rst + 1.5, refill_sec + 1.0)
        except ValueError:
            pass
    sleep_sec = max(sleep_sec, min_floor_sec)
    if sleep_sec > 0:
        time.sleep(min(sleep_sec, 70.0))
    return round(min(sleep_sec, 70.0), 2)


def _extract_verbatim_context(user_prompt: str) -> str:
    """Extract the exact ## CONTEXT block sent in user_prompt without fabricating anything."""
    if not user_prompt:
        return ""
    marker = "## CONTEXT"
    end_marker = "\n---\n\n## CÂU HỎI"
    idx = user_prompt.find(marker)
    if idx == -1:
        return user_prompt
    end_idx = user_prompt.find(end_marker, idx)
    if end_idx != -1:
        return user_prompt[idx:end_idx].strip()
    return user_prompt[idx:].strip()


def _summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {
            "provider_call_count": 0,
            "provider_success_calls": 0,
            "provider_error_calls": 0,
            "provider_rate_limit_429_calls": 0,
            "model_returned": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "provider_elapsed_ms_total": 0.0,
            "finish_reasons": [],
            "provider_http_statuses": [],
            "provider_errors": [],
            "last_rate_limit_headers": {},
            "actual_system_prompt": "",
            "actual_user_prompt": "",
            "actual_context_sent": "",
            "has_verbatim_context": False,
        }
    ok_traces = [t for t in traces if t.get("http_status") == 200]
    err_traces = [t for t in traces if t.get("error_type") is not None]
    rl_429 = [t for t in traces if t.get("error_status_code") == 429]
    primary = ok_traces[-1] if ok_traces else traces[-1]
    user_prompts = [t.get("user_prompt", "") for t in traces if t.get("user_prompt")]
    # Preserve every unique verbatim context sent across calls (important for multi-call quiz/roadmap)
    contexts = [_extract_verbatim_context(up) for up in user_prompts]
    joined_ctx = "\n\n=== NEXT PROVIDER CALL CONTEXT ===\n\n".join(c for c in contexts if c)
    return {
        "provider_call_count": len(traces),
        "provider_success_calls": len(ok_traces),
        "provider_error_calls": len(err_traces),
        "provider_rate_limit_429_calls": len(rl_429),
        "model_returned": primary.get("model_returned") or primary.get("model_requested"),
        "prompt_tokens": sum(int(t["prompt_tokens"]) for t in traces if t.get("prompt_tokens") is not None),
        "completion_tokens": sum(int(t["completion_tokens"]) for t in traces if t.get("completion_tokens") is not None),
        "total_tokens": sum(int(t["total_tokens"]) for t in traces if t.get("total_tokens") is not None),
        "provider_elapsed_ms_total": round(sum(float(t.get("provider_elapsed_ms") or 0.0) for t in traces), 2),
        "finish_reasons": [t.get("finish_reason") for t in traces if t.get("finish_reason") is not None],
        "provider_http_statuses": [t.get("http_status") or t.get("error_status_code") for t in traces],
        "provider_errors": [
            {"type": t.get("error_type"), "status": t.get("error_status_code"), "msg": t.get("error_message")}
            for t in err_traces
        ],
        "last_rate_limit_headers": traces[-1].get("rate_limit_headers") or {},
        "actual_system_prompt": primary.get("system_prompt", ""),
        "actual_user_prompt": primary.get("user_prompt", ""),
        "actual_context_sent": joined_ctx,
        "has_verbatim_context": bool(joined_ctx.strip()),
    }


def _load_jsonl_done_ids(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                row = json.loads(s)
                done[str(row[id_key])] = row
    return done


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_questions_40(root: Path) -> list[dict[str, Any]]:
    rows = []
    with (root / "evaluation" / "datasets" / "questions_40.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _call_http_endpoint(
    client: httpx.Client,
    handle: IsolatedServerHandle,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_sec: float = 120.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Call an isolated FastAPI endpoint and pair exact middleware telemetry + provider traces."""
    trace_before = handle.read_trace_count()
    log_before = handle.read_study_log_count()
    started = time.perf_counter()
    http_status = None
    x_proc_ms = None
    body_json: Any = None
    client_error: str | None = None
    timed_out = False

    try:
        resp = client.post(f"{handle.base_url}{endpoint}", json=payload, timeout=timeout_sec)
        client_ms = (time.perf_counter() - started) * 1000.0
        http_status = resp.status_code
        if "x-process-time-ms" in resp.headers:
            try:
                x_proc_ms = float(resp.headers["x-process-time-ms"])
            except ValueError:
                x_proc_ms = None
        try:
            body_json = resp.json()
        except Exception:
            body_json = {"raw_text": resp.text[:500]}
    except httpx.TimeoutException as exc:
        client_ms = (time.perf_counter() - started) * 1000.0
        timed_out = True
        client_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        client_ms = (time.perf_counter() - started) * 1000.0
        client_error = f"{type(exc).__name__}: {exc}"

    # Brief pause so server middleware finally-block flushes to log file
    time.sleep(0.08)
    new_traces = handle.read_trace_since(trace_before)
    new_logs = handle.read_study_logs_since(log_before)
    srv_log = new_logs[-1] if new_logs else {}
    trace_summary = _summarize_traces(new_traces)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "http_status": http_status,
        "http_transport_ok": http_status == 200 and client_error is None,
        "timed_out": timed_out,
        "client_error": client_error,
        "client_elapsed_ms": round(client_ms, 2),
        "x_process_time_ms": x_proc_ms,
        "server_telemetry": srv_log,
        "provider_telemetry": trace_summary,
        "response": body_json,
    }
    return result, new_traces


def run_http_chat_batch(
    root: Path,
    *,
    tag: str,
    protocol: str,
    candidate_top_k: int,
    out_filename: str,
    query_ids: list[str] | None = None,
    min_remaining_tokens: int = 3400,
) -> list[dict[str, Any]]:
    """Run real HTTP POST /api/chat over frozen 40 questions (or smoke subset) with checkpoint/resume."""
    questions = _load_questions_40(root)
    if query_ids is not None:
        qset = set(query_ids)
        questions = [q for q in questions if q["query_id"] in qset]

    out_path = artifacts_dir(root) / out_filename
    done = _load_jsonl_done_ids(out_path, "query_id")
    pending = [q for q in questions if q["query_id"] not in done]

    if not pending:
        print(f"[{tag}] All {len(questions)} queries already checkpointed in {out_filename}.")
        return list(done.values())

    print(f"[{tag}] Launching isolated backend (candidate_top_k={candidate_top_k}) for {len(pending)}/{len(questions)} pending queries...")
    with run_isolated_backend(root, tag=tag, candidate_top_k=candidate_top_k, port=DEFAULT_PORT) as handle:
        with httpx.Client() as client:
            for idx, q in enumerate(pending, 1):
                qid = q["query_id"]
                payload = {
                    "query": q["query"],
                    "workspace_id": EVAL_WORKSPACE_ID,
                    "corpus_id": EVAL_CORPUS_RECURSIVE,
                }
                res, new_traces = _call_http_endpoint(client, handle, "/api/chat", payload, timeout_sec=110.0)
                resp_body = res["response"] if isinstance(res["response"], dict) else {}
                gen_mode = resp_body.get("generation_mode")
                gen_err = resp_body.get("generation_error")

                # If rate_limit hit despite pacing, wait for TPM window and retry ONCE cleanly before checkpointing
                if gen_err == "rate_limit" or (res["provider_telemetry"]["provider_rate_limit_429_calls"] > 0 and gen_mode != "llm"):
                    print(f"  [{tag}] {qid} hit Groq 429 rate_limit — waiting 45s and retrying once...")
                    time.sleep(45.0)
                    res2, new_traces2 = _call_http_endpoint(client, handle, "/api/chat", payload, timeout_sec=110.0)
                    # Accumulate provider call counts transparently!
                    res2["client_retries_due_to_tpm"] = 1
                    res2["prior_rate_limited_attempt_provider_calls"] = res["provider_telemetry"]["provider_call_count"]
                    res = res2
                    new_traces = new_traces2
                    resp_body = res["response"] if isinstance(res["response"], dict) else {}
                    gen_mode = resp_body.get("generation_mode")
                    gen_err = resp_body.get("generation_error")

                row = {
                    "query_id": qid,
                    "protocol": protocol,
                    "config_tag": tag,
                    "candidate_top_k": candidate_top_k,
                    "final_top_k": 5,
                    "context_budget_applied": False,
                    "context_budget_note": "Unmodified production /api/chat uses MAX_EVIDENCES=5 without token budget packer",
                    "query": q["query"],
                    "language": q.get("language"),
                    "difficulty": q.get("difficulty"),
                    "topic": q.get("topic"),
                    "http_status": res["http_status"],
                    "http_transport_ok": res["http_transport_ok"],
                    "timed_out": res["timed_out"],
                    "client_error": res["client_error"],
                    "client_elapsed_ms": res["client_elapsed_ms"],
                    "x_process_time_ms": res["x_process_time_ms"],
                    "server_telemetry": res["server_telemetry"],
                    "provider_telemetry": res["provider_telemetry"],
                    "generation_mode": gen_mode,
                    "generation_error": gen_err,
                    "live_llm_generated": (
                        res["http_transport_ok"]
                        and gen_mode == "llm"
                        and not gen_err
                        and res["provider_telemetry"]["provider_success_calls"] > 0
                    ),
                    "extractive_fallback": gen_mode == "extractive",
                    "abstained": gen_mode == "abstained",
                    "confidence_score": resp_body.get("confidence_score"),
                    "answer": resp_body.get("answer", ""),
                    "citations": resp_body.get("citations", []),
                    "client_retries_due_to_tpm": res.get("client_retries_due_to_tpm", 0),
                }
                _append_jsonl(out_path, row)
                done[qid] = row

                slept = _pace_for_tpm(new_traces, min_remaining_tokens=min_remaining_tokens, min_floor_sec=0.5)
                print(
                    f"  [{tag}] {idx}/{len(pending)} {qid}: HTTP={row['http_status']} mode={gen_mode} "
                    f"err={gen_err} client={row['client_elapsed_ms']:.0f}ms "
                    f"ret={row['server_telemetry'].get('retrieval_ms', 0):.0f}ms "
                    f"llm={row['server_telemetry'].get('llm_ms_total', 0):.0f}ms "
                    f"tok={row['provider_telemetry']['total_tokens']} pace={slept}s"
                )
    return list(done.values())


def run_live_groq_no_rag(root: Path, *, min_remaining_tokens: int = 2400) -> list[dict[str, Any]]:
    """Run matched direct Live Groq NO_RAG baseline on all 40 frozen questions using openai/gpt-oss-120b."""
    backend_dir = str(root / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import openai
    from app.core.config import settings

    questions = _load_questions_40(root)
    out_path = artifacts_dir(root) / "live_groq_no_rag.jsonl"
    done = _load_jsonl_done_ids(out_path, "query_id")
    pending = [q for q in questions if q["query_id"] not in done]

    if not pending:
        print(f"[no_rag] All {len(questions)} NO_RAG queries already checkpointed.")
        return list(done.values())

    model = settings.GROQ_MODEL
    api_key = settings.GROQ_API_KEY.get_secret_value().strip()
    options: dict[str, Any] = (
        {"extra_body": {"reasoning_effort": "low"}} if model.startswith("openai/gpt-oss-") else {}
    )
    sys_prompt = (
        "Bạn là Trợ lý Giảng dạy môn Cấu trúc Dữ liệu & Giải thuật (DSA) tại Trường Đại học. "
        "Hãy trả lời chính xác, rõ ràng, có cấu trúc (dùng ký hiệu Big-O khi đề cập độ phức tạp và code block nếu cần). "
        "Trả lời cùng ngôn ngữ với câu hỏi của sinh viên (tiếng Việt hoặc tiếng Anh). Không bịa đặt nguồn tài liệu hay nhãn trích dẫn [1]."
    )

    print(f"[no_rag] Running direct Live Groq NO_RAG baseline ({model}) for {len(pending)}/{len(questions)} queries...")
    with openai.OpenAI(
        api_key=api_key, base_url=settings.GROQ_BASE_URL, timeout=45.0, max_retries=0
    ) as client:
        for idx, q in enumerate(pending, 1):
            qid = q["query_id"]
            attempt_traces: list[dict[str, Any]] = []
            answer_text = ""
            http_status = None
            err_msg = None
            returned_model = None
            finish_reason = None

            for attempt in range(3):
                started = time.perf_counter()
                try:
                    raw = client.chat.completions.with_raw_response.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": q["query"]},
                        ],
                        temperature=0.3,
                        max_completion_tokens=1024,
                        **options,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    parsed = raw.parse()
                    hdrs = {
                        k.lower(): v
                        for k, v in raw.headers.items()
                        if "ratelimit" in k.lower() or "retry-after" in k.lower()
                    }
                    usage = getattr(parsed, "usage", None)
                    choice = parsed.choices[0] if parsed.choices else None
                    answer_text = (choice.message.content or "").strip() if choice else ""
                    http_status = raw.status_code
                    returned_model = getattr(parsed, "model", model)
                    finish_reason = getattr(choice, "finish_reason", None) if choice else None
                    attempt_traces.append(
                        {
                            "http_status": raw.status_code,
                            "model_returned": returned_model,
                            "finish_reason": finish_reason,
                            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                            "provider_elapsed_ms": round(elapsed_ms, 2),
                            "rate_limit_headers": hdrs,
                            "error_type": None,
                            "error_status_code": None,
                        }
                    )
                    err_msg = None
                    break
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    st_code = getattr(exc, "status_code", None)
                    resp_obj = getattr(exc, "response", None)
                    hdrs = (
                        {
                            k.lower(): v
                            for k, v in resp_obj.headers.items()
                            if "ratelimit" in k.lower() or "retry-after" in k.lower()
                        }
                        if resp_obj is not None and getattr(resp_obj, "headers", None)
                        else {}
                    )
                    attempt_traces.append(
                        {
                            "http_status": None,
                            "error_type": type(exc).__name__,
                            "error_status_code": st_code,
                            "error_message": str(exc)[:250],
                            "provider_elapsed_ms": round(elapsed_ms, 2),
                            "rate_limit_headers": hdrs,
                        }
                    )
                    err_msg = f"{type(exc).__name__}({st_code})"
                    if st_code == 429 and attempt < 2:
                        time.sleep(30.0)
                        continue
                    break

            tsum = _summarize_traces(attempt_traces)
            row = {
                "query_id": qid,
                "protocol": PROTOCOL_LIVE_GROQ_NO_RAG,
                "model_requested": model,
                "model_returned": returned_model,
                "temperature": 0.3,
                "max_completion_tokens": 1024,
                "reasoning_effort": "low",
                "query": q["query"],
                "language": q.get("language"),
                "difficulty": q.get("difficulty"),
                "http_status": http_status,
                "live_llm_generated": http_status == 200 and bool(answer_text),
                "finish_reason": finish_reason,
                "error": err_msg,
                "client_elapsed_ms": tsum["provider_elapsed_ms_total"],
                "provider_telemetry": tsum,
                "answer": answer_text,
            }
            _append_jsonl(out_path, row)
            done[qid] = row
            slept = _pace_for_tpm(attempt_traces, min_remaining_tokens=min_remaining_tokens, min_floor_sec=0.4)
            print(
                f"  [no_rag] {idx}/{len(pending)} {qid}: HTTP={http_status} "
                f"ms={row['client_elapsed_ms']:.0f} tok={tsum['total_tokens']} pace={slept}s"
            )
    return list(done.values())


def run_edge_robustness_quiz_roadmap(root: Path) -> dict[str, Any]:
    """Run HTTP edge cases, 20 robustness cases, 8 Quiz topics, and 4 Roadmap scenarios under Config B (pool=30)."""
    out_dir = artifacts_dir(root)
    edge_path = out_dir / "http_edge_cases.jsonl"
    rob_path = out_dir / "http_robustness.jsonl"
    quiz_path = out_dir / "http_quiz.jsonl"
    rm_path = out_dir / "http_roadmap.jsonl"

    # Load existing Stage 04 robustness questions & study scenarios
    rob_qs = [
        json.loads(line)
        for line in (root / "evaluation" / "artifacts" / "stage04" / "robustness" / "robustness_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    study_prep = json.loads(
        (root / "evaluation" / "artifacts" / "stage04" / "study_features" / "study_prepared.json")
        .read_text(encoding="utf-8")
    )
    quiz_topics = study_prep["quiz_topics"]
    roadmap_scenarios = study_prep["roadmap_scenarios"]

    done_edge = _load_jsonl_done_ids(edge_path, "case_id")
    done_rob = _load_jsonl_done_ids(rob_path, "case_id")
    done_quiz = _load_jsonl_done_ids(quiz_path, "case_id")
    done_rm = _load_jsonl_done_ids(rm_path, "scenario_id")

    need_main_server = (
        len(done_edge) < 8
        or len(done_rob) < len(rob_qs)
        or len(done_quiz) < len(quiz_topics) + 1  # +1 for raw QZ_03 comma 422 check
        or len(done_rm) < len(roadmap_scenarios)
    )

    if need_main_server:
        with run_isolated_backend(root, tag="config_b_study_rob", candidate_top_k=30, port=DEFAULT_PORT) as handle:
            with httpx.Client() as client:
                # 1. Edge cases (empty workspace, invalid payloads, client timeout)
                edge_specs = [
                    ("EDGE_HEALTH", "GET", "/health", None, 200),
                    ("EDGE_READY", "GET", "/ready", None, 200),
                    ("EDGE_EMPTY_CHAT", "POST", "/api/chat", {"query": "Binary Search là gì?", "workspace_id": EMPTY_WORKSPACE_ID, "corpus_id": EMPTY_CORPUS_ID}, 200),
                    ("EDGE_EMPTY_QUIZ", "POST", "/api/quiz", {"topic": "Binary Search", "num_questions": 3, "workspace_id": EMPTY_WORKSPACE_ID, "corpus_id": EMPTY_CORPUS_ID}, 200),
                    ("EDGE_EMPTY_ROADMAP", "POST", "/api/roadmap", {"topic": "Binary Search", "workspace_id": EMPTY_WORKSPACE_ID, "corpus_id": EMPTY_CORPUS_ID}, 200),
                    ("EDGE_INVALID_EMPTY_QUERY", "POST", "/api/chat", {"query": "   ", "workspace_id": EVAL_WORKSPACE_ID, "corpus_id": EVAL_CORPUS_RECURSIVE}, 400),
                    ("EDGE_INVALID_ROADMAP_EXTRA", "POST", "/api/roadmap", {"topic": "Stack", "unexpected_field": 1, "workspace_id": EVAL_WORKSPACE_ID, "corpus_id": EVAL_CORPUS_RECURSIVE}, 422),
                    ("EDGE_CLIENT_TIMEOUT", "POST_TIMEOUT", "/api/chat", {"query": "So sánh BFS và DFS", "workspace_id": EVAL_WORKSPACE_ID, "corpus_id": EVAL_CORPUS_RECURSIVE}, None),
                ]
                for cid, verb, ep, body, exp_status in edge_specs:
                    if cid in done_edge:
                        continue
                    if verb == "GET":
                        t0 = time.perf_counter()
                        r = client.get(f"{handle.base_url}{ep}", timeout=10.0)
                        row = {
                            "case_id": cid, "endpoint": ep, "expected_status": exp_status,
                            "http_status": r.status_code, "client_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "passed": r.status_code == exp_status, "response": r.json(),
                        }
                    elif verb == "POST_TIMEOUT":
                        res, _ = _call_http_endpoint(client, handle, ep, body, timeout_sec=0.001)
                        row = {
                            "case_id": cid, "endpoint": ep, "expected_behavior": "client_timeout",
                            "http_status": res["http_status"], "timed_out": res["timed_out"],
                            "client_elapsed_ms": res["client_elapsed_ms"], "passed": bool(res["timed_out"]),
                            "client_error": res["client_error"],
                        }
                        time.sleep(2.0)  # allow abandoned server work to settle
                    else:
                        res, _ = _call_http_endpoint(client, handle, ep, body, timeout_sec=30.0)
                        resp_d = res["response"] if isinstance(res["response"], dict) else {}
                        passed = res["http_status"] == exp_status
                        if cid.startswith("EDGE_EMPTY_"):
                            passed = passed and resp_d.get("generation_mode") == "abstained"
                        row = {
                            "case_id": cid, "endpoint": ep, "expected_status": exp_status,
                            "http_status": res["http_status"], "client_elapsed_ms": res["client_elapsed_ms"],
                            "generation_mode": resp_d.get("generation_mode"),
                            "passed": passed, "response": resp_d,
                            "provider_telemetry": res["provider_telemetry"],
                        }
                    _append_jsonl(edge_path, row)
                    done_edge[cid] = row
                    print(f"  [edge] {cid}: status={row.get('http_status')} passed={row['passed']}")

                # 2. 20 HTTP Robustness cases (explicitly noting query-level prompt injection vs Stage 04 context-level)
                for idx, rq in enumerate(rob_qs, 1):
                    cid = rq["case_id"]
                    if cid in done_rob:
                        continue
                    q_text = rq["query"]
                    injection_surface = "none"
                    if rq["category"] == "prompt_injection" and rq.get("injected_text"):
                        q_text = f"{rq['query']}\n{rq['injected_text']}"
                        injection_surface = "query_level_http_input"
                    payload = {
                        "query": q_text,
                        "workspace_id": EVAL_WORKSPACE_ID,
                        "corpus_id": EVAL_CORPUS_RECURSIVE,
                    }
                    res, new_traces = _call_http_endpoint(client, handle, "/api/chat", payload, timeout_sec=110.0)
                    resp_body = res["response"] if isinstance(res["response"], dict) else {}
                    if resp_body.get("generation_error") == "rate_limit":
                        time.sleep(45.0)
                        res, new_traces = _call_http_endpoint(client, handle, "/api/chat", payload, timeout_sec=110.0)
                        resp_body = res["response"] if isinstance(res["response"], dict) else {}

                    ans = resp_body.get("answer", "")
                    forb = rq.get("forbidden_token")
                    inj_resisted = (forb not in ans) if forb else None
                    row = {
                        "case_id": cid,
                        "category": rq["category"],
                        "language": rq["language"],
                        "linked_query_id": rq.get("linked_query_id"),
                        "expected_behavior": rq["expected_behavior"],
                        "injection_surface": injection_surface,
                        "query_sent": q_text,
                        "forbidden_token": forb,
                        "injection_token_absent": inj_resisted,
                        "http_status": res["http_status"],
                        "http_transport_ok": res["http_transport_ok"],
                        "client_elapsed_ms": res["client_elapsed_ms"],
                        "x_process_time_ms": res["x_process_time_ms"],
                        "server_telemetry": res["server_telemetry"],
                        "provider_telemetry": res["provider_telemetry"],
                        "generation_mode": resp_body.get("generation_mode"),
                        "generation_error": resp_body.get("generation_error"),
                        "abstained": resp_body.get("generation_mode") == "abstained",
                        "extractive_fallback": resp_body.get("generation_mode") == "extractive",
                        "live_llm_generated": resp_body.get("generation_mode") == "llm" and not resp_body.get("generation_error"),
                        "confidence_score": resp_body.get("confidence_score"),
                        "answer": ans,
                        "citations": resp_body.get("citations", []),
                    }
                    _append_jsonl(rob_path, row)
                    done_rob[cid] = row
                    slept = _pace_for_tpm(new_traces, min_remaining_tokens=3400, min_floor_sec=0.5)
                    print(
                        f"  [rob] {idx}/{len(rob_qs)} {cid} ({rq['category']}): HTTP={row['http_status']} "
                        f"mode={row['generation_mode']} tok={row['provider_telemetry']['total_tokens']} pace={slept}s"
                    )

                # 3. Real HTTP /api/quiz (preserve raw QZ_03 422 validation failure + normalized single-topic run!)
                quiz_queue: list[tuple[str, str, str, bool]] = []
                for qt in quiz_topics:
                    tid, raw_topic = qt["topic_id"], qt["topic"]
                    if "," in raw_topic or ";" in raw_topic:
                        # Record authentic raw 422 first, then normalized single-topic!
                        quiz_queue.append((f"{tid}_RAW_VALIDATION", raw_topic, raw_topic, False))
                        norm_topic = raw_topic.replace(",", " and").replace(";", " and")
                        norm_topic = " ".join(norm_topic.split())
                        quiz_queue.append((tid, raw_topic, norm_topic, True))
                    else:
                        quiz_queue.append((tid, raw_topic, raw_topic, False))

                for q_idx, (cid, raw_topic, sent_topic, was_norm) in enumerate(quiz_queue, 1):
                    if cid in done_quiz:
                        continue
                    payload = {
                        "topic": sent_topic,
                        "num_questions": 5,
                        "workspace_id": EVAL_WORKSPACE_ID,
                        "corpus_id": EVAL_CORPUS_RECURSIVE,
                    }
                    res, new_traces = _call_http_endpoint(client, handle, "/api/quiz", payload, timeout_sec=245.0)
                    resp_body = res["response"] if isinstance(res["response"], dict) else {}
                    if resp_body.get("generation_error") == "rate_limit":
                        print(f"  [quiz] {cid} hit 429 — waiting 55s and retrying once...")
                        time.sleep(55.0)
                        res, new_traces = _call_http_endpoint(client, handle, "/api/quiz", payload, timeout_sec=245.0)
                        resp_body = res["response"] if isinstance(res["response"], dict) else {}

                    qs_out = resp_body.get("questions", []) if isinstance(resp_body, dict) else []
                    schema_ok = (
                        res["http_status"] == 200
                        and isinstance(qs_out, list)
                        and all(
                            isinstance(item.get("question"), str)
                            and isinstance(item.get("options"), list)
                            and len(item["options"]) == 4
                            and isinstance(item.get("correct_index"), int)
                            and 0 <= item["correct_index"] <= 3
                            for item in qs_out
                        )
                    )
                    q_texts = [str(item.get("question", "")).strip().lower() for item in qs_out]
                    dup_rate = (1.0 - len(set(q_texts)) / len(q_texts)) if q_texts else 0.0
                    row = {
                        "case_id": cid,
                        "topic_id": cid.replace("_RAW_VALIDATION", ""),
                        "original_topic": raw_topic,
                        "sent_topic": sent_topic,
                        "input_normalized": was_norm,
                        "is_raw_validation_probe": cid.endswith("_RAW_VALIDATION"),
                        "http_status": res["http_status"],
                        "http_transport_ok": res["http_transport_ok"],
                        "client_elapsed_ms": res["client_elapsed_ms"],
                        "x_process_time_ms": res["x_process_time_ms"],
                        "server_telemetry": res["server_telemetry"],
                        "provider_telemetry": res["provider_telemetry"],
                        "generation_mode": resp_body.get("generation_mode") if isinstance(resp_body, dict) else None,
                        "generation_error": resp_body.get("generation_error") if isinstance(resp_body, dict) else None,
                        "requested_count": resp_body.get("requested_count", 5) if isinstance(resp_body, dict) else 5,
                        "generated_count": resp_body.get("generated_count", 0) if isinstance(resp_body, dict) else 0,
                        "schema_valid": schema_ok,
                        "duplicate_rate": round(dup_rate, 4),
                        "questions": qs_out,
                        "citations": resp_body.get("citations", []) if isinstance(resp_body, dict) else [],
                        "raw_response_detail": resp_body.get("detail") if res["http_status"] != 200 else None,
                    }
                    _append_jsonl(quiz_path, row)
                    done_quiz[cid] = row
                    slept = _pace_for_tpm(new_traces, min_remaining_tokens=5600, min_floor_sec=1.0)
                    print(
                        f"  [quiz] {q_idx}/{len(quiz_queue)} {cid}: HTTP={row['http_status']} "
                        f"mode={row['generation_mode']} count={row['generated_count']}/5 "
                        f"err={row['generation_error']} tok={row['provider_telemetry']['total_tokens']} pace={slept}s"
                    )

                # 4. Real HTTP /api/roadmap (4 Stage 04 scenarios)
                for r_idx, rm in enumerate(roadmap_scenarios, 1):
                    sid = rm["scenario_id"]
                    if sid in done_rm:
                        continue
                    payload = {
                        "topic": rm["goal"],
                        "workspace_id": EVAL_WORKSPACE_ID,
                        "corpus_id": EVAL_CORPUS_RECURSIVE,
                    }
                    res, new_traces = _call_http_endpoint(client, handle, "/api/roadmap", payload, timeout_sec=245.0)
                    resp_body = res["response"] if isinstance(res["response"], dict) else {}
                    if resp_body.get("generation_error") == "rate_limit":
                        print(f"  [roadmap] {sid} hit 429 — waiting 55s and retrying once...")
                        time.sleep(55.0)
                        res, new_traces = _call_http_endpoint(client, handle, "/api/roadmap", payload, timeout_sec=245.0)
                        resp_body = res["response"] if isinstance(res["response"], dict) else {}

                    items_out = resp_body.get("items", []) if isinstance(resp_body, dict) else []
                    schema_ok = (
                        res["http_status"] == 200
                        and isinstance(items_out, list)
                        and all(
                            isinstance(it.get("title"), str)
                            and isinstance(it.get("objective"), str)
                            and isinstance(it.get("key_knowledge"), list)
                            and isinstance(it.get("support_quote"), str)
                            for it in items_out
                        )
                    )
                    titles = [str(it.get("title", "")).strip().lower() for it in items_out]
                    dup_rate = (1.0 - len(set(titles)) / len(titles)) if titles else 0.0
                    row = {
                        "scenario_id": sid,
                        "goal": rm["goal"],
                        "http_status": res["http_status"],
                        "http_transport_ok": res["http_transport_ok"],
                        "client_elapsed_ms": res["client_elapsed_ms"],
                        "x_process_time_ms": res["x_process_time_ms"],
                        "server_telemetry": res["server_telemetry"],
                        "provider_telemetry": res["provider_telemetry"],
                        "generation_mode": resp_body.get("generation_mode") if isinstance(resp_body, dict) else None,
                        "generation_error": resp_body.get("generation_error") if isinstance(resp_body, dict) else None,
                        "items_count": len(items_out),
                        "schema_valid": schema_ok,
                        "duplicate_rate": round(dup_rate, 4),
                        "items": items_out,
                        "uncovered_topics": resp_body.get("uncovered_topics", []) if isinstance(resp_body, dict) else [],
                        "warnings": resp_body.get("warnings", []) if isinstance(resp_body, dict) else [],
                        "citations": resp_body.get("citations", []) if isinstance(resp_body, dict) else [],
                    }
                    _append_jsonl(rm_path, row)
                    done_rm[sid] = row
                    slept = _pace_for_tpm(new_traces, min_remaining_tokens=5600, min_floor_sec=1.0)
                    print(
                        f"  [roadmap] {r_idx}/{len(roadmap_scenarios)} {sid}: HTTP={row['http_status']} "
                        f"mode={row['generation_mode']} items={row['items_count']} "
                        f"err={row['generation_error']} tok={row['provider_telemetry']['total_tokens']} pace={slept}s"
                    )

    # 5. Server-side 504 timeout handling probe in a dedicated short-lived server (CHAT_REQUEST_TIMEOUT=0.05)
    if "EDGE_SERVER_504_TIMEOUT" not in done_edge:
        with run_isolated_backend(
            root,
            tag="timeout_probe",
            candidate_top_k=10,
            port=TIMEOUT_TEST_PORT,
            extra_env={"CHAT_REQUEST_TIMEOUT": "0.05"},
        ) as handle_to:
            with httpx.Client() as client:
                res, _ = _call_http_endpoint(
                    client,
                    handle_to,
                    "/api/chat",
                    {"query": "So sánh thuật toán QuickSort và MergeSort", "workspace_id": EVAL_WORKSPACE_ID, "corpus_id": EVAL_CORPUS_RECURSIVE},
                    timeout_sec=30.0,
                )
                row = {
                    "case_id": "EDGE_SERVER_504_TIMEOUT",
                    "endpoint": "/api/chat",
                    "expected_status": 504,
                    "http_status": res["http_status"],
                    "client_elapsed_ms": res["client_elapsed_ms"],
                    "passed": res["http_status"] == 504,
                    "response": res["response"],
                }
                _append_jsonl(edge_path, row)
                done_edge["EDGE_SERVER_504_TIMEOUT"] = row
                print(f"  [edge] EDGE_SERVER_504_TIMEOUT: status={row['http_status']} passed={row['passed']}")

    return {
        "edge_cases": len(done_edge),
        "robustness": len(done_rob),
        "quiz": len(done_quiz),
        "roadmap": len(done_rm),
    }
