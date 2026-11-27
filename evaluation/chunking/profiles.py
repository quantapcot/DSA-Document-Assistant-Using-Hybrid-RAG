"""Official Evaluation chunking profile configurations (Profile A vs Profile B)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation.corpus.provenance import (
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    compute_config_hash,
)

PROFILE_A_ID = "recursive_512_100"
PROFILE_B_ID = "structure_adaptive"

RECURSIVE_CONFIG: dict[str, Any] = {
    "profile_id": PROFILE_A_ID,
    "unit": "characters",
    "chunk_size": 512,
    "chunk_overlap": 100,
    "min_chunk_size": 100,
    "min_words": 15,
    "non_alpha_max_ratio": 0.8,
    "separators": ["\n\n", "\n", ". ", " ", ""],
}

CONTENT_CLASSIFICATION_PRECEDENCE: tuple[str, ...] = (
    "table",
    "heading",
    "code_block",
    "pseudocode",
    "complexity_analysis",
    "algorithm_steps",
    "definition",
    "formula",
    "algorithm_explanation",
    "narrative",
)


@dataclass(frozen=True)
class ContentTypeBudget:
    soft_min: int
    soft_target_low: int
    soft_target_high: int
    soft_max: int
    allow_short_standalone: bool
    overlap_max_tokens: int


ADAPTIVE_CONTENT_BUDGETS: dict[str, ContentTypeBudget] = {
    "heading": ContentTypeBudget(2, 2, 24, 24, False, 0),
    "definition": ContentTypeBudget(12, 30, 60, 60, True, 0),
    "narrative": ContentTypeBudget(20, 50, 90, 90, True, 18),
    "algorithm_explanation": ContentTypeBudget(20, 70, 110, 110, True, 20),
    "algorithm_steps": ContentTypeBudget(20, 70, 110, 110, True, 22),
    "code_block": ContentTypeBudget(15, 50, 115, 115, True, 20),
    "pseudocode": ContentTypeBudget(15, 50, 115, 115, True, 20),
    "formula": ContentTypeBudget(8, 20, 80, 80, True, 0),
    "complexity_analysis": ContentTypeBudget(12, 30, 80, 80, True, 0),
    "table": ContentTypeBudget(15, 30, 100, 100, True, 35),
}

ADAPTIVE_CONFIG_V1: dict[str, Any] = {
    "profile_id": PROFILE_B_ID,
    "config_version": "ADAPTIVE_CONFIG_V1",
    "tokenizer_model": EMBEDDING_MODEL_ID,
    "tokenizer_revision": EMBEDDING_MODEL_REVISION,
    "hard_max_total_embed_tokens": EMBEDDING_MAX_SEQ_LENGTH,
    "proposed_content_budget": EMBEDDING_MAX_SEQ_LENGTH - 2,
    "heading_prefix_max_tokens": 24,
    "table_header_max_tokens": 40,
    "min_prose_keep_tokens": 3,
    "merge_cross_page": False,
    "precedence": list(CONTENT_CLASSIFICATION_PRECEDENCE),
    "budgets": {k: asdict(v) for k, v in ADAPTIVE_CONTENT_BUDGETS.items()},
}


def get_profile_config(profile_id: str) -> dict[str, Any]:
    if profile_id == PROFILE_A_ID:
        return dict(RECURSIVE_CONFIG)
    if profile_id == PROFILE_B_ID:
        return dict(ADAPTIVE_CONFIG_V1)
    raise ValueError(f"Unknown chunking profile: {profile_id!r}")


def get_profile_config_hash(profile_id: str) -> str:
    return compute_config_hash(get_profile_config(profile_id))
