"""Profile A (recursive_512_100) adapter reusing production chunker with exact per-char origin tracking."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_processing.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    _SEPARATORS,
    chunk_text,
)
from app.document_processing.text_cleaner import is_valid_chunk_text

from evaluation.chunking.profiles import PROFILE_A_ID, RECURSIVE_CONFIG, get_profile_config_hash
from evaluation.chunking.structure_adaptive import total_emb_tokens
from evaluation.corpus.provenance import (
    CanonicalDocument,
    CanonicalPage,
    FallbackInfo,
    OverlapSegment,
    SourceSpan,
    build_chunk_record,
)

# Each character carries (canonical_index_or_-1, is_overlap_flag)
_Origin = tuple[int, bool]


@dataclass(frozen=True)
class _TrackedString:
    text: str
    origins: tuple[_Origin, ...]


def _strip_tracked(ts: _TrackedString) -> _TrackedString:
    s = ts.text
    stripped = s.strip()
    if not stripped:
        return _TrackedString("", ())
    lead = len(s) - len(s.lstrip())
    trail = len(s) - len(s.rstrip())
    end = len(s) - trail
    return _TrackedString(stripped, ts.origins[lead:end])


def _split_tracked_str(ts: _TrackedString, sep: str) -> list[_TrackedString]:
    """Split *ts* identically to str.split(sep) (or list(str) when sep=='') preserving per-char origins."""
    s = ts.text
    orig = ts.origins
    if not sep:
        return [_TrackedString(s[i], (orig[i],)) for i in range(len(s))]
    parts: list[_TrackedString] = []
    pos = 0
    sep_len = len(sep)
    while True:
        idx = s.find(sep, pos)
        if idx == -1:
            parts.append(_TrackedString(s[pos:], orig[pos:]))
            break
        parts.append(_TrackedString(s[pos:idx], orig[pos:idx]))
        pos = idx + sep_len
    return parts


def _split_by_separator_tracked(
    ts: _TrackedString,
    sep: str,
    chunk_size: int,
    overlap: int,
    consumed_seps: list[tuple[int, int]] | None = None,
) -> list[_TrackedString]:
    """Exact mirror of app.document_processing.chunker._split_by_separator with per-char origins."""
    parts = _split_tracked_str(ts, sep)
    # Reconstruct separator origins directly from gaps between consecutive split slices in ts
    sep_origins_list: list[tuple[_Origin, ...]] = []
    if sep:
        cursor = 0
        sep_len = len(sep)
        for p in parts[:-1]:
            cursor += len(p.text)
            sep_origins_list.append(ts.origins[cursor : cursor + sep_len])
            cursor += sep_len
    else:
        sep_origins_list = [() for _ in range(max(0, len(parts) - 1))]

    chunks: list[_TrackedString] = []
    cur_text = ""
    cur_orig: list[_Origin] = []

    for idx, part in enumerate(parts):
        if cur_text:
            sep_orig = sep_origins_list[idx - 1]
            cand_len = len(cur_text) + len(sep) + len(part.text)
            if cand_len <= chunk_size:
                cur_text = cur_text + sep + part.text
                cur_orig.extend(sep_orig)
                cur_orig.extend(part.origins)
            else:
                if consumed_seps is not None and sep.strip():
                    for ch_s, (pos, is_ov) in zip(sep, sep_orig):
                        if not is_ov and pos >= 0 and not ch_s.isspace():
                            consumed_seps.append((pos, pos + 1))
                chunks.append(_strip_tracked(_TrackedString(cur_text, tuple(cur_orig))))
                cur_text = part.text
                cur_orig = list(part.origins)
        else:
            cur_text = part.text
            cur_orig = list(part.origins)

    if cur_text:
        chunks.append(_strip_tracked(_TrackedString(cur_text, tuple(cur_orig))))

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[_TrackedString] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        tail_text = prev.text[-overlap:]
        tail_orig = tuple((pos, True) for pos, _ in prev.origins[-overlap:])
        combined_text = tail_text + " " + cur.text
        combined_orig = (*tail_orig, (-1, True), *cur.origins)
        overlapped.append(_strip_tracked(_TrackedString(combined_text, combined_orig)))
    return overlapped


def _recursive_split_tracked(
    ts: _TrackedString,
    separators: list[str],
    chunk_size: int,
    overlap: int,
    consumed_seps: list[tuple[int, int]] | None = None,
) -> list[_TrackedString]:
    """Exact mirror of app.document_processing.chunker._recursive_split with per-char origins."""
    if not ts.text:
        return []
    if len(ts.text) <= chunk_size:
        stripped = _strip_tracked(ts)
        return [stripped] if stripped.text else []

    for sep_idx, sep in enumerate(separators):
        if sep == "" or sep in ts.text:
            chunks = _split_by_separator_tracked(ts, sep, chunk_size, overlap, consumed_seps=consumed_seps)
            remaining_seps = separators[sep_idx + 1:]
            result: list[_TrackedString] = []
            for ch in chunks:
                if len(ch.text) > chunk_size and remaining_seps:
                    result.extend(
                        _recursive_split_tracked(
                            ch, remaining_seps, chunk_size, overlap, consumed_seps=consumed_seps
                        )
                    )
                elif ch.text.strip():
                    result.append(_strip_tracked(ch))
            return result

    step = chunk_size - overlap
    return [
        _strip_tracked(_TrackedString(ts.text[i : i + chunk_size], ts.origins[i : i + chunk_size]))
        for i in range(0, len(ts.text), step)
    ]


def _extract_contiguous_runs(origins: tuple[_Origin, ...], *, want_overlap: bool) -> list[tuple[int, int]]:
    """Collapse per-character canonical indices into exact half-open [start, end) intervals."""
    runs: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None

    for pos, is_ov in origins:
        if pos < 0 or is_ov != want_overlap:
            if cur_start is not None and cur_end is not None:
                runs.append((cur_start, cur_end))
                cur_start = cur_end = None
            continue
        if cur_start is None:
            cur_start = pos
            cur_end = pos + 1
        elif pos == cur_end:
            cur_end = pos + 1
        else:
            runs.append((cur_start, cur_end))
            cur_start = pos
            cur_end = pos + 1

    if cur_start is not None and cur_end is not None:
        runs.append((cur_start, cur_end))
    return runs


def _blocks_for_intervals(page: CanonicalPage, intervals: list[tuple[int, int]]) -> list[SourceSpan]:
    """Map exact [start, end) intervals to CanonicalBlock SourceSpans without expanding bounds."""
    spans: list[SourceSpan] = []
    text = page.canonical_text
    for start, end in intervals:
        if end <= start:
            continue
        matched_any = False
        for blk in page.blocks:
            os_ = max(start, blk.char_start)
            oe_ = min(end, blk.char_end)
            if oe_ > os_ and text[os_:oe_].strip():
                # Trim leading/trailing whitespace so every span points to exact non-blank text
                sub = text[os_:oe_]
                lead = len(sub) - len(sub.lstrip())
                trail = len(sub) - len(sub.rstrip())
                spans.append(SourceSpan(page=page.page, block_id=blk.block_id, char_start=os_ + lead, char_end=oe_ - trail))
                matched_any = True
        if not matched_any and text[start:end].strip():
            sub = text[start:end]
            lead = len(sub) - len(sub.lstrip())
            trail = len(sub) - len(sub.rstrip())
            spans.append(SourceSpan(
                page=page.page,
                block_id=f"{page.document_id}:p{page.page}:raw",
                char_start=start + lead,
                char_end=end - trail,
            ))
    return spans


def chunk_document_recursive(
    doc: CanonicalDocument,
    emb_tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run existing Recursive Character Chunking (512/100) on a shared CanonicalDocument.

    Preserves 100% of production chunk_text() boundaries, text, order, section, and
    is_valid_chunk_text(min_words=15) filtering while providing character-exact provenance.
    """
    cfg_hash = get_profile_config_hash(PROFILE_A_ID)
    chunk_records: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    chunk_idx = 0

    for page in doc.pages:
        canonical = page.canonical_text
        if not canonical or not canonical.strip():
            continue

        # 1. Exact production chunk_text() boundaries
        prod_chunks = chunk_text(
            canonical,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
        # 2. Exact per-character origin mirror (zero .find() heuristics, zero coarse parent fallback)
        initial_ts = _TrackedString(canonical, tuple((i, False) for i in range(len(canonical))))
        consumed_seps: list[tuple[int, int]] = []
        tracked_all = _recursive_split_tracked(
            initial_ts, _SEPARATORS, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, consumed_seps=consumed_seps
        )
        tracked_kept = [t for t in tracked_all if len(t.text) >= MIN_CHUNK_SIZE]
        tracked_short_dropped = [t for t in tracked_all if len(t.text) < MIN_CHUNK_SIZE]

        for s, e in consumed_seps:
            dropped_records.append({
                "document_id": doc.document_id,
                "page": page.page,
                "char_start": s,
                "char_end": e,
                "reason": "recursive_separator_consumed",
            })

        for dt in tracked_short_dropped:
            for s, e in _extract_contiguous_runs(dt.origins, want_overlap=False):
                if canonical[s:e].strip():
                    dropped_records.append({
                        "document_id": doc.document_id,
                        "page": page.page,
                        "char_start": s,
                        "char_end": e,
                        "reason": "MIN_CHUNK_SIZE_lt_100",
                    })

        # Strict parity invariant between tracked mirror and production chunk_text()
        if len(tracked_kept) != len(prod_chunks) or any(
            t.text != p["text"] for t, p in zip(tracked_kept, prod_chunks)
        ):
            raise RuntimeError(
                f"Recursive provenance mirror diverged from production chunk_text on "
                f"{doc.source_path} page {page.page}"
            )

        for seg, ch in zip(tracked_kept, prod_chunks):
            text = ch["text"]
            section = ch["section"]

            orig_runs = _extract_contiguous_runs(seg.origins, want_overlap=False)
            ov_runs = _extract_contiguous_runs(seg.origins, want_overlap=True)

            valid_words = len(text.split()) >= RECURSIVE_CONFIG["min_words"]
            valid_full = is_valid_chunk_text(text, min_words=RECURSIVE_CONFIG["min_words"])
            if not valid_full:
                reason = "min_words_lt_15" if not valid_words else "non_alpha_gt_80pct"
                for s, e in orig_runs:
                    if canonical[s:e].strip():
                        dropped_records.append({
                            "document_id": doc.document_id,
                            "page": page.page,
                            "char_start": s,
                            "char_end": e,
                            "reason": reason,
                        })
                continue

            source_spans = _blocks_for_intervals(page, orig_runs)
            overlap_segs: list[OverlapSegment] = []
            for ov_s, ov_e in ov_runs:
                if ov_e > ov_s and canonical[ov_s:ov_e].strip():
                    sub = canonical[ov_s:ov_e]
                    lead = len(sub) - len(sub.lstrip())
                    trail = len(sub) - len(sub.rstrip())
                    s_trim, e_trim = ov_s + lead, ov_e - trail
                    ov_blk = _blocks_for_intervals(page, [(s_trim, e_trim)])
                    overlap_segs.append(OverlapSegment(
                        from_chunk_id=chunk_records[-1]["chunk_id"] if chunk_records else None,
                        page=page.page,
                        block_id=ov_blk[0].block_id if ov_blk else None,
                        char_start=s_trim,
                        char_end=e_trim,
                        text_len=e_trim - s_trim,
                    ))

            emb_toks = total_emb_tokens(emb_tokenizer, text)
            body_for_hash = "".join(canonical[sp.char_start:sp.char_end] for sp in source_spans) if source_spans else text
            fallback = FallbackInfo(
                is_forced_split=False,
                reason="overlap_only_recursive_fragment" if (not source_spans and overlap_segs) else None,
            )

            record = build_chunk_record(
                doc=doc,
                page=page.page,
                section=section,
                text=text,
                chunk_index=chunk_idx,
                profile_id=PROFILE_A_ID,
                chunking_config_hash=cfg_hash,
                content_type="recursive_segment",
                language=page.language,
                source_spans=source_spans,
                overlap_segments=overlap_segs,
                synthetic_segments=[],
                fallback=fallback,
                emb_token_count=emb_toks,
                body_text_for_hash=body_for_hash,
            )
            chunk_records.append(record)
            chunk_idx += 1

    return chunk_records, dropped_records
