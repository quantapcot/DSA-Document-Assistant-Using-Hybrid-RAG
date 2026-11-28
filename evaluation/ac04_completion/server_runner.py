"""Lifecycle manager for the isolated evaluation FastAPI server and telemetry pairing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from evaluation.ac04_completion import DEFAULT_PORT, EVAL_COLLECTION, runtime_dir

_STUDY_LOG_RE = re.compile(
    r"study endpoint=(?P<endpoint>\S+)\s+elapsed_ms=(?P<elapsed_ms>[0-9.]+)\s+"
    r"retrieval_calls=(?P<retrieval_calls>\d+)\s+inventory_reads=(?P<inventory_reads>\d+)\s+"
    r"retrieval_ms=(?P<retrieval_ms>[0-9.]+)\s+provider_calls=(?P<provider_calls>\d+)\s+"
    r"provider_retries=(?P<provider_retries>\d+)\s+content_retries=(?P<content_retries>\d+)\s+"
    r"llm_ms=(?P<llm_ms>\{.*\})"
)


def write_isolated_retrieval_yaml(
    root: Path,
    *,
    tag: str,
    candidate_top_k: int,
    final_top_k: int = 5,
) -> Path:
    """Write a temporary retrieval YAML pointing strictly to the isolated qdrant_eval_copy."""
    rdir = runtime_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    qdrant_copy = (rdir / "qdrant_eval_copy").resolve().as_posix()
    yaml_path = rdir / f"retrieval_{tag}.yaml"
    content = f"""# Isolated AC-04 Evaluation Config ({tag}) — DO NOT USE FOR PRODUCTION
embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
reranker_model: BAAI/bge-reranker-v2-m3
collection: {EVAL_COLLECTION}
qdrant_mode: local
qdrant_path: "{qdrant_copy}"
candidate_top_k: {candidate_top_k}
final_top_k: {final_top_k}
rrf_k: 60
bm25_k1: 1.5
bm25_b: 0.75
accent_insensitive: false
batch_size: 64
embedding_device: auto
reranker_device: auto
reranker_batch_size: 4
reranker_max_length: 512
bm25_cache_ttl: 300
bm25_cache_max_entries: 16
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


@dataclass
class IsolatedServerHandle:
    base_url: str
    tag: str
    log_path: Path
    trace_path: Path
    process: subprocess.Popen[str]

    def read_trace_count(self) -> int:
        if not self.trace_path.is_file():
            return 0
        with self.trace_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def read_trace_since(self, start_idx: int) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.trace_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                s = line.strip()
                if s and idx >= start_idx:
                    rows.append(json.loads(s))
        return rows

    def read_study_log_count(self) -> int:
        if not self.log_path.is_file():
            return 0
        txt = self.log_path.read_text(encoding="utf-8", errors="ignore")
        return len(_STUDY_LOG_RE.findall(txt))

    def read_study_logs_since(self, start_idx: int) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        txt = self.log_path.read_text(encoding="utf-8", errors="ignore")
        matches = list(_STUDY_LOG_RE.finditer(txt))
        out: list[dict[str, Any]] = []
        for m in matches[start_idx:]:
            llm_raw = m.group("llm_ms").replace("'", '"')
            try:
                llm_dict = json.loads(llm_raw)
            except json.JSONDecodeError:
                llm_dict = {}
            out.append(
                {
                    "endpoint": m.group("endpoint"),
                    "server_elapsed_ms": float(m.group("elapsed_ms")),
                    "retrieval_calls": int(m.group("retrieval_calls")),
                    "inventory_reads": int(m.group("inventory_reads")),
                    "retrieval_ms": float(m.group("retrieval_ms")),
                    "provider_calls": int(m.group("provider_calls")),
                    "provider_retries": int(m.group("provider_retries")),
                    "content_retries": int(m.group("content_retries")),
                    "llm_ms_by_task": llm_dict,
                    "llm_ms_total": round(sum(float(v) for v in llm_dict.values()), 2),
                }
            )
        return out


@contextmanager
def run_isolated_backend(
    root: Path,
    *,
    tag: str,
    candidate_top_k: int,
    port: int = DEFAULT_PORT,
    extra_env: dict[str, str] | None = None,
    startup_timeout_sec: float = 120.0,
) -> Iterator[IsolatedServerHandle]:
    """Launch unmodified FastAPI backend pointed at isolated eval Qdrant copy; guarantee cleanup."""
    rdir = runtime_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    yaml_path = write_isolated_retrieval_yaml(root, tag=tag, candidate_top_k=candidate_top_k)
    log_path = rdir / f"server_{tag}.log"
    trace_path = rdir / f"provider_trace_{tag}.jsonl"
    sessions_dir = rdir / f"sessions_{tag}"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Start fresh server log per launch (provider trace appends so resume preserves history)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "backend"),
            "PYTHONUTF8": "1",
            "RETRIEVAL_CONFIG_PATH": str(yaml_path),
            "SESSION_RUNTIME_DIR": str(sessions_dir),
            "AC04_SERVER_PORT": str(port),
            "AC04_PROVIDER_TRACE_PATH": str(trace_path),
        }
    )
    if extra_env:
        env.update(extra_env)

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(root / "evaluation" / "ac04_completion" / "asgi_harness.py")],
        cwd=str(root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    handle = IsolatedServerHandle(
        base_url=base_url, tag=tag, log_path=log_path, trace_path=trace_path, process=proc
    )

    try:
        deadline = time.monotonic() + startup_timeout_sec
        ready = False
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    log_file.flush()
                    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:]
                    raise RuntimeError(f"Isolated backend exited early (code={proc.returncode}):\n{tail}")
                try:
                    r = client.get(f"{base_url}/ready")
                    if r.status_code == 200 and r.json().get("status") == "ready":
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.8)
        if not ready:
            raise TimeoutError(f"Isolated backend did not become ready within {startup_timeout_sec}s")
        yield handle
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()
        # Small pause for Windows SQLite file lock release
        time.sleep(1.0)
