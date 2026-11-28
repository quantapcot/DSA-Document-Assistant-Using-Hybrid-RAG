"""Supplementary Stage 04 AC-04 completion: Live Groq API & isolated HTTP End-to-End evaluation."""

from __future__ import annotations

from pathlib import Path

EXPECTED_BRANCH = "feat/final-readiness"
EXPECTED_STAGE05_HEAD = "139957c0d6974df273c86ac2029be1c8fb7d158e"
EXPECTED_MAIN = "b84f09450f213b03ad57ef8b388f482ee0f0eaa2"

EVAL_WORKSPACE_ID = "evaluation"
EVAL_CORPUS_RECURSIVE = "eval-recursive-512-v1"
EVAL_COLLECTION = "eval_dsa_chunks_v1"

EMPTY_WORKSPACE_ID = "eval-empty-workspace"
EMPTY_CORPUS_ID = "eval-empty-corpus"

DEFAULT_PORT = 8015
TIMEOUT_TEST_PORT = 8016

# Protocols kept strictly separate across all artifacts, tables, and charts
PROTOCOL_HTTP_CONFIG_A = "http_e2e_config_a_prod_default_pool10"
PROTOCOL_HTTP_CONFIG_B = "http_e2e_config_b_exploratory_pool30"
PROTOCOL_LIVE_GROQ_NO_RAG = "direct_live_groq_no_rag"
PROTOCOL_STAGE04_CONTROLLED = "stage04_controlled_agent_a4_budget2048"


def artifacts_dir(root: Path) -> Path:
    return root / "evaluation" / "artifacts" / "ac04_completion"


def runtime_dir(root: Path) -> Path:
    return root / "data" / "runtime" / "ac04_eval_runtime"
