"""Shared canonical source representation, provenance contracts, coverage, and manifests."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "benchmark_manifest_v1"
RAW_AGGREGATE_SPEC = "sha256_v1_relpath_colon_bytes_colon_hex_lf"
SOURCE_IDENTITY_SCHEMA_VERSION = "source_id_v1"
CHUNK_IDENTITY_SCHEMA_VERSION = "chunk_id_v1"
CANONICAL_COORDINATE_VERSION = "canonical_nfc_page_v1"

EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_MAX_SEQ_LENGTH = 128

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANKER_MAX_LENGTH = 512


@dataclass(frozen=True)
class SourceSpan:
    """Half-open Unicode NFC character interval [char_start, char_end) in canonical page text."""

    page: int
    block_id: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "block_id": self.block_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True)
class OverlapSegment:
    """Context overlap carried from a preceding chunk; excluded from original coverage union."""

    from_chunk_id: str | None
    page: int
    block_id: str | None
    char_start: int | None
    char_end: int | None
    text_len: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticSegment:
    """Repeated/derived context (heading prefix, table header, code signature, or linearized row)."""

    kind: str  # "heading_prefix" | "table_header" | "code_signature" | "linearized_table_row"
    source_block_id: str | None
    text_len: int
    truncated: bool = False
    source_backed: bool = False
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FallbackInfo:
    """Records whether deterministic splitting triggered fallback or forced token-span split."""

    is_forced_split: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalBlock:
    """One extracted block on a CanonicalPage with exact coordinates in canonical_text."""

    block_id: str
    raw_kind: str  # "paragraph" | "heading" | "table" | "code" | "list_item" | "text_block"
    text: str
    char_start: int
    char_end: int
    style_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_monospace: bool = False
    indent_level: float = 0.0
    table_header: str | None = None
    row_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class CanonicalPage:
    """Canonical cleaned page/virtual-section representation shared by Profiles A and B."""

    document_id: str
    page: int
    file_type: str
    canonical_text: str
    language: str
    lang_hint: str
    blocks: list[CanonicalBlock] = field(default_factory=list)
    dropped_duplicate_records: int = 0
    extraction_notes: list[str] = field(default_factory=list)


@dataclass
class CanonicalDocument:
    """Canonical source document parsed identically for Profile A and Profile B."""

    document_id: str
    filename: str
    source_path: str
    source_revision: str
    file_type: str
    file_bytes: int
    lang_hint: str
    pages: list[CanonicalPage] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_document_id(normalized_rel_path: str) -> str:
    """Create collision-free document_id from relative path including file extension."""
    norm = normalized_rel_path.strip().replace("\\", "/")
    if not norm:
        raise ValueError("normalized_rel_path must be nonempty")
    return sha256_text(norm)[:16]


def compute_content_hash(text: str) -> str:
    return sha256_text(text)[:16]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(config))[:12]


def make_chunk_id(
    *,
    profile_id: str,
    chunking_config_hash: str,
    document_id: str,
    source_revision: str,
    page: int,
    chunk_index: int,
    primary_span: tuple[int, int] | None,
    content_hash: str,
) -> str:
    """Create deterministic 24-hex chunk_id isolated by profile and configuration."""
    span_label = f"{primary_span[0]}-{primary_span[1]}" if primary_span else "unmapped"
    payload = (
        f"{profile_id}|{chunking_config_hash}|{document_id}|{source_revision}"
        f"|p{page}|i{chunk_index}|{span_label}|{content_hash}"
    )
    return sha256_text(payload)[:24]


def interval_union_length(intervals: list[tuple[int, int]]) -> int:
    """Compute exact merged character length of [start, end) intervals without double-counting."""
    valid = sorted((max(0, s), max(0, e)) for s, e in intervals if e > s)
    if not valid:
        return 0
    total = 0
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def build_chunk_record(
    *,
    doc: CanonicalDocument,
    page: int,
    section: str | None,
    text: str,
    chunk_index: int,
    profile_id: str,
    chunking_config_hash: str,
    content_type: str,
    language: str,
    source_spans: list[SourceSpan],
    overlap_segments: list[OverlapSegment],
    synthetic_segments: list[SyntheticSegment],
    fallback: FallbackInfo,
    emb_token_count: int,
    reranker_token_count: int | None = None,
    canonical_source_version: str = CANONICAL_COORDINATE_VERSION,
    body_text_for_hash: str | None = None,
) -> dict[str, Any]:
    """Assemble a DocumentChunk-compatible dict with full evaluation provenance metadata."""
    body_for_hash = body_text_for_hash if body_text_for_hash is not None else text
    content_hash = compute_content_hash(body_for_hash)
    primary_span = (
        (source_spans[0].char_start, source_spans[-1].char_end)
        if source_spans
        else None
    )
    chunk_id = make_chunk_id(
        profile_id=profile_id,
        chunking_config_hash=chunking_config_hash,
        document_id=doc.document_id,
        source_revision=doc.source_revision,
        page=page,
        chunk_index=chunk_index,
        primary_span=primary_span,
        content_hash=content_hash,
    )
    return {
        "chunk_id": chunk_id,
        "document_id": doc.document_id,
        "filename": doc.filename,
        "page": page,
        "section": section,
        "text": text,
        "metadata": {
            "language": language,
            "lang_hint": doc.lang_hint,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "file_type": doc.file_type,
            "source_path": doc.source_path,
            "source_revision": doc.source_revision,
            "profile_id": profile_id,
            "chunking_config_hash": chunking_config_hash,
            "content_hash": content_hash,
            "content_type": content_type,
            "canonical_source_version": canonical_source_version,
            "source_spans": [s.to_dict() for s in source_spans],
            "overlap_segments": [o.to_dict() for o in overlap_segments],
            "synthetic_segments": [syn.to_dict() for syn in synthetic_segments],
            "fallback": fallback.to_dict(),
            "emb_token_count": emb_token_count,
            **({"reranker_token_count": reranker_token_count} if reranker_token_count is not None else {}),
        },
    }


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


def validate_chunk_provenance(chunk: dict[str, Any], canonical_page: CanonicalPage) -> list[str]:
    """Validate DocumentChunk compatibility, bounds, and actual canonical text correspondence."""
    errors: list[str] = []
    for key in ("chunk_id", "document_id", "filename", "text"):
        val = chunk.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"invalid top-level {key}")
    if type(chunk.get("page")) is not int or chunk["page"] <= 0:
        errors.append("invalid page")
    if chunk.get("document_id") != canonical_page.document_id:
        errors.append("document_id mismatch with canonical_page")
    if chunk.get("page") != canonical_page.page:
        errors.append("page mismatch with canonical_page")

    meta = chunk.get("metadata")
    if not isinstance(meta, dict):
        return [*errors, "missing metadata dict"]
    for req in ("source_path", "source_revision", "profile_id", "chunking_config_hash", "content_hash", "source_spans"):
        if req not in meta:
            errors.append(f"missing metadata.{req}")

    canonical_text = canonical_page.canonical_text
    page_len = len(canonical_text)
    chunk_text = chunk.get("text", "")
    chunk_norm = _norm_ws(chunk_text)
    fb_reason = (meta.get("fallback") or {}).get("reason")

    spans = meta.get("source_spans", [])
    overlaps = meta.get("overlap_segments", [])
    if not isinstance(spans, list):
        errors.append("source_spans must be a list")
    elif not spans and not overlaps:
        errors.append("empty source_spans and overlap_segments")
    else:
        for sp in spans:
            cs, ce = sp.get("char_start"), sp.get("char_end")
            if sp.get("page") != canonical_page.page:
                errors.append(f"span page mismatch: {sp}")
                continue
            if type(cs) is not int or type(ce) is not int or cs < 0 or ce <= cs or ce > page_len:
                errors.append(f"out-of-bounds span {sp} for page_len={page_len}")
                continue
            src_sub = canonical_text[cs:ce]
            if not src_sub.strip():
                errors.append(f"blank source_span {sp}")
                continue
            if fb_reason == "oversized_table_row_linearized":
                cells = [c.strip() for c in src_sub.strip("| ").split("|") if c.strip()]
                if cells and not any(_norm_ws(cell) in chunk_norm or chunk_norm in _norm_ws(cell) for cell in cells):
                    errors.append(f"linearized table row span mismatch {sp}")
            elif src_sub not in chunk_text and _norm_ws(src_sub) not in chunk_norm:
                errors.append(f"source_span text mismatch {sp}")

    for ov in overlaps:
        cs, ce = ov.get("char_start"), ov.get("char_end")
        if cs is not None and ce is not None:
            if type(cs) is not int or type(ce) is not int or cs < 0 or ce <= cs or ce > page_len:
                errors.append(f"out-of-bounds overlap_segment {ov}")
                continue
            ov_sub = canonical_text[cs:ce]
            if ov_sub.strip() and ov_sub not in chunk_text and _norm_ws(ov_sub) not in chunk_norm:
                errors.append(f"overlap_segment text mismatch {ov}")

    for syn in meta.get("synthetic_segments", []):
        if syn.get("source_backed"):
            cs, ce = syn.get("char_start"), syn.get("char_end")
            if type(cs) is not int or type(ce) is not int or cs < 0 or ce <= cs or ce > page_len:
                errors.append(f"out-of-bounds source_backed synthetic_segment {syn}")
                continue
            syn_src = canonical_text[cs:ce]
            if not syn.get("truncated"):
                if syn_src not in chunk_text and _norm_ws(syn_src) not in chunk_norm:
                    errors.append(f"source_backed synthetic_segment text mismatch {syn}")

    return errors


def build_raw_inventory(raw_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Build deterministic raw inventory and versioned aggregate SHA-256."""
    files = sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}),
        key=lambda p: p.relative_to(raw_dir).as_posix(),
    )
    inventory: list[dict[str, Any]] = []
    for path in files:
        data = path.read_bytes()
        rel_path = path.relative_to(raw_dir).as_posix()
        inventory.append({
            "rel_path": rel_path,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        })
    raw_aggregate = sha256_text(
        "\n".join(f"{r['rel_path']}:{r['bytes']}:{r['sha256']}" for r in inventory)
    )
    return inventory, raw_aggregate


def build_benchmark_manifest(
    *,
    raw_inventory: list[dict[str, Any]],
    raw_aggregate_sha256: str,
    preprocessor_info: dict[str, Any],
    profile_id: str,
    chunking_config: dict[str, Any],
    chunking_config_hash: str,
    chunks: list[dict[str, Any]],
    corpus_jsonl_relpath: str,
    corpus_sha256: str,
    qdrant_scope: dict[str, str],
) -> dict[str, Any]:
    """Build a deterministic benchmark_manifest_v1 without wall-clock timestamps."""
    chunk_ids_digest = sha256_text("\n".join(c["chunk_id"] for c in chunks))
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "raw_file_count": len(raw_inventory),
        "raw_total_bytes": sum(r["bytes"] for r in raw_inventory),
        "raw_aggregate_sha256_spec": RAW_AGGREGATE_SPEC,
        "raw_aggregate_sha256": raw_aggregate_sha256,
        "raw_inventory": raw_inventory,
        "preprocessor": preprocessor_info,
        "identity_schema_versions": {
            "source_identity": SOURCE_IDENTITY_SCHEMA_VERSION,
            "chunk_identity": CHUNK_IDENTITY_SCHEMA_VERSION,
            "canonical_coordinate": CANONICAL_COORDINATE_VERSION,
        },
        "tokenizer": {
            "embedding_model": EMBEDDING_MODEL_ID,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
            "embedding_max_seq_length": EMBEDDING_MAX_SEQ_LENGTH,
            "reranker_model": RERANKER_MODEL_ID,
            "reranker_revision": RERANKER_MODEL_REVISION,
            "reranker_max_length": RERANKER_MAX_LENGTH,
        },
        "profile": {
            "profile_id": profile_id,
            "chunking_config": chunking_config,
            "chunking_config_hash": chunking_config_hash,
            "chunk_count": len(chunks),
            "chunk_ids_sha256": chunk_ids_digest,
            "corpus_jsonl_path": corpus_jsonl_relpath,
            "corpus_sha256": corpus_sha256,
            "qdrant_scope": qdrant_scope,
        },
    }
