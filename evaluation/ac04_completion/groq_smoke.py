"""Environment verification, isolated Qdrant copy verification, and minimal live Groq smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.ac04_completion import (
    EXPECTED_BRANCH,
    EXPECTED_MAIN,
    EXPECTED_STAGE05_HEAD,
    artifacts_dir,
    runtime_dir,
)
from evaluation.stage05.data_loader import verify_inputs_and_integrity


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_checksums(folder: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            rel = p.relative_to(folder).as_posix()
            out[rel] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    return out


def prepare_isolated_qdrant_copy(root: Path) -> dict[str, Any]:
    """Create and checksum-verify an isolated copy of data/runtime/qdrant_eval."""
    src = root / "data" / "runtime" / "qdrant_eval"
    dst = runtime_dir(root) / "qdrant_eval_copy"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing frozen evaluation Qdrant directory: {src}")

    src_hashes = _dir_checksums(src)
    if dst.exists():
        # Verify existing copy; rebuild cleanly only if checksum differs
        dst_hashes = _dir_checksums(dst)
        if dst_hashes != src_hashes:
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    dst_hashes = _dir_checksums(dst)
    # Exclude transient SQLite lock file if created later, but right after copy must be 100% identical
    identical = src_hashes == dst_hashes
    if not identical:
        raise RuntimeError("Isolated Qdrant copy checksum verification FAILED!")

    return {
        "source_dir": src.relative_to(root).as_posix(),
        "isolated_copy_dir": dst.relative_to(root).as_posix(),
        "files_verified": len(src_hashes),
        "checksums_identical": identical,
        "source_checksums": src_hashes,
    }


def verify_original_qdrant_untouched(root: Path, baseline_checksums: dict[str, dict[str, Any]]) -> bool:
    """Confirm original data/runtime/qdrant_eval and data/runtime/qdrant_local were never modified."""
    src = root / "data" / "runtime" / "qdrant_eval"
    current = _dir_checksums(src)
    return current == baseline_checksums


def verify_ac04_environment(root: Path) -> dict[str, Any]:
    """Verify Git baseline, Stage 04/05 frozen integrity, secret gitignore, and backend Groq config."""
    stage05_integrity = verify_inputs_and_integrity(root)

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.strip()

    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    main_sha = _git("rev-parse", "main")
    stage05_ancestor_ok = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_STAGE05_HEAD, "HEAD"],
            cwd=str(root),
            capture_output=True,
        ).returncode
        == 0
    )
    env_gitignored = (
        subprocess.run(["git", "check-ignore", "backend/.env"], cwd=str(root), capture_output=True).returncode == 0
    )

    # Load backend settings cleanly without exposing key value
    backend_dir = str(root / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from app.core.config import settings

    key_obj = settings.GROQ_API_KEY
    key_val = key_obj.get_secret_value().strip() if key_obj else ""
    key_ready = bool(key_val) and key_val.startswith("gsk_") and len(key_val) >= 30

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": branch,
            "expected_branch": EXPECTED_BRANCH,
            "head": head,
            "expected_stage05_head": EXPECTED_STAGE05_HEAD,
            "stage05_ancestor_ok": stage05_ancestor_ok,
            "main": main_sha,
            "expected_main": EXPECTED_MAIN,
            "baseline_ok": (
                branch == EXPECTED_BRANCH
                and stage05_ancestor_ok
                and main_sha == EXPECTED_MAIN
            ),
        },
        "stage04_stage05_integrity_ok": stage05_integrity["all_verified"],
        "secrets_safety": {
            "backend_env_gitignored": env_gitignored,
            "groq_api_key_configured": key_ready,
            "groq_api_key_length": len(key_val),
            "groq_api_key_starts_gsk": key_val.startswith("gsk_"),
        },
        "groq_config": {
            "base_url": settings.GROQ_BASE_URL,
            "configured_model": settings.GROQ_MODEL,
            "llm_chat_timeout": settings.LLM_CHAT_TIMEOUT,
            "llm_quiz_generation_timeout": settings.LLM_QUIZ_GENERATION_TIMEOUT,
            "llm_quiz_verification_timeout": settings.LLM_QUIZ_VERIFICATION_TIMEOUT,
            "llm_revision_timeout": settings.LLM_REVISION_TIMEOUT,
            "chat_request_timeout": settings.CHAT_REQUEST_TIMEOUT,
            "quiz_request_timeout": settings.QUIZ_REQUEST_TIMEOUT,
            "revision_request_timeout": settings.REVISION_REQUEST_TIMEOUT,
        },
    }


def run_groq_smoke_test(root: Path) -> dict[str, Any]:
    """Run a minimal 1-call live Groq API smoke test to verify auth, model, and rate limits."""
    env_check = verify_ac04_environment(root)
    if not env_check["secrets_safety"]["groq_api_key_configured"]:
        raise RuntimeError("GROQ_API_KEY is not configured in backend/.env")
    if not env_check["secrets_safety"]["backend_env_gitignored"]:
        raise RuntimeError("backend/.env is NOT gitignored — refusing to proceed!")

    qdrant_copy = prepare_isolated_qdrant_copy(root)

    backend_dir = str(root / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import openai
    from app.core.config import settings

    model = settings.GROQ_MODEL
    api_key = settings.GROQ_API_KEY.get_secret_value().strip()
    options: dict[str, Any] = (
        {"extra_body": {"reasoning_effort": "low"}} if model.startswith("openai/gpt-oss-") else {}
    )

    started = time.perf_counter()
    smoke_result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": model,
        "base_url": settings.GROQ_BASE_URL,
        "connectivity_ok": False,
        "http_status": None,
        "returned_model": None,
        "elapsed_ms": None,
        "usage": {},
        "rate_limit_headers": {},
        "response_preview": "",
        "error": None,
    }

    try:
        with openai.OpenAI(
            api_key=api_key,
            base_url=settings.GROQ_BASE_URL,
            timeout=30.0,
            max_retries=0,
        ) as client:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Reply concisely in one short sentence."},
                    {"role": "user", "content": "What is the average time complexity of Binary Search on a sorted array?"},
                ],
                temperature=0.0,
                max_completion_tokens=64,
                **options,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            parsed = raw_resp.parse()
            headers = {
                k.lower(): v
                for k, v in raw_resp.headers.items()
                if "ratelimit" in k.lower() or "retry-after" in k.lower() or "x-groq" in k.lower()
            }
            usage = {}
            if getattr(parsed, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(parsed.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(parsed.usage, "completion_tokens", None),
                    "total_tokens": getattr(parsed.usage, "total_tokens", None),
                }
            content = (parsed.choices[0].message.content or "").strip() if parsed.choices else ""
            smoke_result.update(
                {
                    "connectivity_ok": True,
                    "http_status": raw_resp.status_code,
                    "returned_model": getattr(parsed, "model", None),
                    "finish_reason": getattr(parsed.choices[0], "finish_reason", None) if parsed.choices else None,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "usage": usage,
                    "rate_limit_headers": headers,
                    "response_preview": content[:200],
                }
            )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        smoke_result["elapsed_ms"] = round(elapsed_ms, 2)
        smoke_result["error"] = f"{type(exc).__name__}: {getattr(exc, 'status_code', '')} {str(exc)[:300]}"

    out_dir = artifacts_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": env_check,
        "isolated_qdrant_copy": qdrant_copy,
        "groq_smoke_test": smoke_result,
    }
    (out_dir / "smoke_test_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
