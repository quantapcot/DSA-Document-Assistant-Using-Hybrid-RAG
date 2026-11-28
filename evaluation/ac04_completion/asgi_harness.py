"""Isolated evaluation uvicorn launcher with read-only OpenAI/Groq SDK call telemetry.

Runs the unmodified production FastAPI app (backend/app/main.py) without editing a single
byte of backend code, while observing exact Groq provider calls (verbatim messages/context,
exact token usage, finish_reason, HTTP status, rate-limit headers, retries, and errors)
in-process and writing sanitized JSONL trace records (never logging API keys).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TRACE_LOCK = threading.Lock()
_CALL_SEQ = 0


def _install_readonly_provider_observer(trace_path: Path) -> None:
    """Wrap openai.resources.chat.completions.Completions.create read-only for telemetry."""
    import openai.resources.chat.completions as chat_mod

    orig_create = chat_mod.Completions.create

    def observed_create(self, *args, **kwargs):
        global _CALL_SEQ
        started = time.perf_counter()
        messages = kwargs.get("messages", [])
        sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        usr_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        rec: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_requested": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "response_format_type": (
                kwargs["response_format"].get("type")
                if isinstance(kwargs.get("response_format"), dict)
                else None
            ),
            "extra_body": kwargs.get("extra_body"),
            "system_prompt": sys_msg,
            "user_prompt": usr_msg,
            "http_status": None,
            "model_returned": None,
            "finish_reason": None,
            "assistant_content": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "rate_limit_headers": {},
            "provider_elapsed_ms": None,
            "error_type": None,
            "error_status_code": None,
            "error_message": None,
        }

        try:
            # Use with_raw_response on the exact same resource to capture HTTP headers + parsed body
            raw_resp = self.with_raw_response.create(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            parsed = raw_resp.parse()
            headers = {
                k.lower(): v
                for k, v in raw_resp.headers.items()
                if "ratelimit" in k.lower() or "retry-after" in k.lower() or "x-groq" in k.lower()
            }
            choice = parsed.choices[0] if getattr(parsed, "choices", None) else None
            usage = getattr(parsed, "usage", None)
            rec.update(
                {
                    "http_status": raw_resp.status_code,
                    "model_returned": getattr(parsed, "model", None),
                    "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                    "assistant_content": getattr(choice.message, "content", "") if choice else "",
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                    "rate_limit_headers": headers,
                    "provider_elapsed_ms": round(elapsed_ms, 2),
                }
            )
            return parsed
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            resp_obj = getattr(exc, "response", None)
            headers = {}
            if resp_obj is not None and getattr(resp_obj, "headers", None):
                headers = {
                    k.lower(): v
                    for k, v in resp_obj.headers.items()
                    if "ratelimit" in k.lower() or "retry-after" in k.lower()
                }
            rec.update(
                {
                    "provider_elapsed_ms": round(elapsed_ms, 2),
                    "error_type": type(exc).__name__,
                    "error_status_code": getattr(exc, "status_code", None),
                    "error_message": str(exc)[:300],
                    "rate_limit_headers": headers,
                }
            )
            raise
        finally:
            with _TRACE_LOCK:
                _CALL_SEQ += 1
                rec["call_seq"] = _CALL_SEQ
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Guard against recursion when calling self.with_raw_response.create (which uses _client directly, not Completions.create)
    # Verify: with_raw_response wraps Completions.create ONLY if bound; in openai>=1.x, Async/Sync raw wrapper calls orig!
    # Therefore call orig_create directly on raw wrapper OR use orig_create + httpx response hook!
    _ = orig_create


def _install_safe_provider_observer(trace_path: Path) -> None:
    """Safe, recursion-free read-only observer using openai.resources.chat.completions.Completions.create + httpx hook."""
    import openai
    import openai.resources.chat.completions as chat_mod
    from openai._response import to_raw_response_wrapper

    orig_create = chat_mod.Completions.create

    def observed_create(self, *args, **kwargs):
        global _CALL_SEQ
        started = time.perf_counter()
        messages = kwargs.get("messages", [])
        sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        usr_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        rec: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_requested": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "response_format_type": (
                kwargs["response_format"].get("type")
                if isinstance(kwargs.get("response_format"), dict)
                else None
            ),
            "extra_body": kwargs.get("extra_body"),
            "system_prompt": sys_msg,
            "user_prompt": usr_msg,
            "http_status": None,
            "model_returned": None,
            "finish_reason": None,
            "assistant_content": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "rate_limit_headers": {},
            "provider_elapsed_ms": None,
            "error_type": None,
            "error_status_code": None,
            "error_message": None,
        }
        try:
            raw_fn = to_raw_response_wrapper(orig_create)
            raw_resp = raw_fn(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            parsed = raw_resp.parse()
            headers = {
                k.lower(): v
                for k, v in raw_resp.headers.items()
                if "ratelimit" in k.lower() or "retry-after" in k.lower() or "x-groq" in k.lower()
            }
            choice = parsed.choices[0] if getattr(parsed, "choices", None) else None
            usage = getattr(parsed, "usage", None)
            rec.update(
                {
                    "http_status": raw_resp.status_code,
                    "model_returned": getattr(parsed, "model", None),
                    "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                    "assistant_content": getattr(choice.message, "content", "") if choice else "",
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                    "rate_limit_headers": headers,
                    "provider_elapsed_ms": round(elapsed_ms, 2),
                }
            )
            return parsed
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            resp_obj = getattr(exc, "response", None)
            headers = {}
            if resp_obj is not None and getattr(resp_obj, "headers", None):
                headers = {
                    k.lower(): v
                    for k, v in resp_obj.headers.items()
                    if "ratelimit" in k.lower() or "retry-after" in k.lower()
                }
            rec.update(
                {
                    "provider_elapsed_ms": round(elapsed_ms, 2),
                    "error_type": type(exc).__name__,
                    "error_status_code": getattr(exc, "status_code", None),
                    "error_message": str(exc)[:300],
                    "rate_limit_headers": headers,
                }
            )
            raise
        finally:
            with _TRACE_LOCK:
                _CALL_SEQ += 1
                rec["call_seq"] = _CALL_SEQ
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    chat_mod.Completions.create = observed_create
    _ = openai


def main() -> None:
    port = int(os.environ.get("AC04_SERVER_PORT", "8015"))
    trace_file = Path(os.environ["AC04_PROVIDER_TRACE_PATH"])
    _install_safe_provider_observer(trace_file)

    import uvicorn
    from app.main import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
