"""Profile B: Rule-based, deterministic Structure-aware Adaptive Chunking (ADAPTIVE_CONFIG_V1)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluation.chunking.profiles import (
    ADAPTIVE_CONFIG_V1,
    ADAPTIVE_CONTENT_BUDGETS,
    PROFILE_B_ID,
    get_profile_config_hash,
)
from evaluation.corpus.provenance import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    FallbackInfo,
    OverlapSegment,
    SourceSpan,
    SyntheticSegment,
    build_chunk_record,
)

# ---------------------------------------------------------------------------
# Rule-based DSA structure detectors (deterministic)
# ---------------------------------------------------------------------------

_HEADING_NUM_RE = re.compile(
    r"^(?:#{1,6}\s+.+|(?:\d+\.)+\d*\s+\S.+|(?:Chương|Chapter|Phần|Bài|Section|Topic)\s+\d+.*)$",
    re.IGNORECASE,
)
_DEF_MARKER_RE = re.compile(
    r"^(?:Định nghĩa|Khái niệm|Definition|Concept)\b\s*[:\-–]"
    r"|^.{2,60}\s+(?:là một|là cấu trúc dữ liệu|được định nghĩa là|is a data structure|is an algorithm|is defined as)\b",
    re.IGNORECASE,
)
_ALGO_EXPL_RE = re.compile(
    r"^(?:Ý tưởng|Nguyên lý|Nguyên tắc|Hoạt động|Giải thích|Idea|Intuition|Approach|How it works)\b",
    re.IGNORECASE,
)
_STEP_LINE_RE = re.compile(r"^\s*(?:Bước\s*\d+\s*[:.]|Step\s*\d+\s*[:.]|\d+[\.\)]\s+\S|[-*•]\s+\S)", re.IGNORECASE)
_CODE_LINE_RE = re.compile(
    r"(?:^\s*(?:def |class |struct |void |int |long |bool |char |float |double |template\s*<|#include\b|return\b|for\s*\(|while\s*\(|if\s*\(|else\s*\{?|elif\b))"
    r"|(?:;\s*$)|(?:\{\s*$)|(?:^\s*\}\s*$)|(?:->)",
)
_PSEUDO_LINE_RE = re.compile(
    r"\b(?:Algorithm|Procedure|Function|Input|Output|Repeat|Until|Then|EndFor|EndWhile|EndIf)\b|←|:=",
    re.IGNORECASE,
)
_COMPLEXITY_RE = re.compile(
    r"(?:Độ phức tạp|Time complexity|Space complexity|Auxiliary space|Worst[\s-]case|Average[\s-]case|Best[\s-]case)\b"
    r"|(?:[OΘΩ]\s*\([^)]+\))",
    re.IGNORECASE,
)
_INDEXABLE_TERM_RE = re.compile(r"[^\W_]", re.UNICODE)
_FORMULA_RE = re.compile(
    r"[∑∏∀∃≤≥≠≈∈∪⌊⌋⌈⌉÷±×∞→⇒⇔]"
    r"|T\s*\([^)]+\)\s*="
    r"|(?:[^\W\d_]|[\U0001D400-\U0001D7FF])(?:\s*\([^)]{0,24}\)|\s*\[[^\]]{0,24}\])?\s*=\s*[\w\[\(\{\-–−\U0001D400-\U0001D7FF]",
    re.UNICODE,
)
_PROSE_TERMINATOR_RE = re.compile(r"[.?!]\s*$")

# Per-character canonical origin: (block_id, canonical_char_idx) or None for synthetic joiner chars
_CharOrigin = tuple[str, int] | None


@dataclass
class _HeadingPrefixInfo:
    text: str
    truncated: bool
    spans: list[SourceSpan]
    original_text: str


@dataclass
class _TypedUnit:
    block_id: str
    content_type: str
    text: str
    spans: list[SourceSpan]
    origins: list[_CharOrigin]
    table_header: str | None = None
    row_spans: list[tuple[int, int]] | None = None
    has_pipe_table_signal: bool = False


@dataclass
class _BodyPiece:
    text: str
    spans: list[SourceSpan]
    origins: list[_CharOrigin] | None = None
    is_forced_split: bool = False
    fallback_reason: str | None = None
    synthetic_override: str | None = None
    table_header_span: tuple[str, int, int] | None = None
    table_header_truncated: bool = False


def _make_span_origins(span: SourceSpan, text_len: int) -> list[_CharOrigin]:
    """Create exact per-character origin map for a single canonical span."""
    if text_len <= 0:
        return []
    span_len = span.char_end - span.char_start
    if span_len == text_len:
        return [(span.block_id, span.char_start + i) for i in range(text_len)]
    return [
        (span.block_id, span.char_start + i) if i < span_len else None
        for i in range(text_len)
    ]


def _origins_to_spans(page_no: int, origins: list[_CharOrigin], text: str) -> list[SourceSpan]:
    """Collapse per-character canonical origins into exact non-blank contiguous SourceSpans."""
    spans: list[SourceSpan] = []
    n = min(len(origins), len(text))
    run_blk: str | None = None
    run_src_start = 0
    run_src_end = 0
    run_txt_start = 0

    def flush(txt_end: int) -> None:
        if run_blk is None or run_src_end <= run_src_start:
            return
        sub = text[run_txt_start:txt_end]
        if not sub.strip():
            return
        lead = len(sub) - len(sub.lstrip())
        trail = len(sub) - len(sub.rstrip())
        s = run_src_start + lead
        e = run_src_end - trail
        if e > s:
            spans.append(SourceSpan(page=page_no, block_id=run_blk, char_start=s, char_end=e))

    for i in range(n):
        orig = origins[i]
        if orig is None:
            flush(i)
            run_blk = None
            continue
        blk_id, pos = orig
        if run_blk == blk_id and pos == run_src_end:
            run_src_end = pos + 1
        else:
            flush(i)
            run_blk = blk_id
            run_src_start = pos
            run_src_end = pos + 1
            run_txt_start = i

    flush(n)
    return spans


def _join_tracked_parts(
    parts: list[tuple[str, list[_CharOrigin]]],
    sep: str,
) -> tuple[str, list[_CharOrigin]]:
    """Join tracked (text, origins) segments with *sep*, marking separator chars with None origins."""
    if not parts:
        return "", []
    texts: list[str] = []
    origins: list[_CharOrigin] = []
    sep_orig: list[_CharOrigin] = [None] * len(sep)
    for idx, (t, o) in enumerate(parts):
        if idx > 0 and sep:
            origins.extend(sep_orig)
        texts.append(t)
        origins.extend(o)
    return sep.join(texts), origins


def load_embedding_tokenizer():
    """Load the cached MiniLM embedding tokenizer offline (128-token hard budget)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        ADAPTIVE_CONFIG_V1["tokenizer_model"],
        local_files_only=True,
    )


def total_emb_tokens(tokenizer: Any, text: str) -> int:
    """Measure exact encoder input token count including model special tokens."""
    if not text:
        return 2
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def _clamp_text_tokens(tokenizer: Any, text: str, max_content_tokens: int) -> tuple[str, bool]:
    """Clamp a synthetic prefix cleanly within *max_content_tokens* without calling encode > budget."""
    if not text or max_content_tokens <= 0:
        return "", bool(text)
    tok_count = len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
    if tok_count <= max_content_tokens:
        return text, False

    words = text.split()
    kept: list[str] = []
    for w in words:
        cand = " ".join([*kept, w])
        if len(tokenizer(cand, add_special_tokens=False, truncation=False)["input_ids"]) <= max_content_tokens:
            kept.append(w)
        else:
            break
    if kept:
        return " ".join(kept), True
    # Single oversized token/word fallback
    hi = len(text)
    lo = 1
    best = text[:1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(tokenizer(text[:mid], add_special_tokens=False, truncation=False)["input_ids"]) <= max_content_tokens:
            best = text[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


def _split_mixed_block(blk: CanonicalBlock, page_no: int) -> list[tuple[str, str, int, int]]:
    """Split a block along line boundaries if it mixes code/pseudocode and complexity prose."""
    if blk.raw_kind in ("table", "heading", "code"):
        return [(blk.raw_kind, blk.text, blk.char_start, blk.char_end)]

    lines = blk.text.splitlines(keepends=True)
    if len(lines) < 2:
        return [(blk.raw_kind, blk.text, blk.char_start, blk.char_end)]

    def line_mode(ln: str) -> str:
        s = ln.strip()
        if not s:
            return "blank"
        is_code_ln = bool(_CODE_LINE_RE.search(ln) or ln.startswith(("    ", "\t")))
        is_pseudo_ln = bool(_PSEUDO_LINE_RE.search(ln))
        if is_code_ln or is_pseudo_ln:
            if s.startswith(("#", "//")):
                return "code"
            return "code"
        if _COMPLEXITY_RE.search(s):
            return "complexity"
        return "prose"

    groups: list[tuple[str, str, int, int]] = []
    cur_mode: str | None = None
    cur_start = blk.char_start
    cur_buf: list[str] = []
    pos = blk.char_start

    for ln in lines:
        mode = line_mode(ln)
        if mode == "blank":
            if cur_buf:
                cur_buf.append(ln)
            else:
                cur_start += len(ln)
            pos += len(ln)
            continue

        if cur_mode is None:
            cur_mode = mode
            cur_buf.append(ln)
        elif mode == cur_mode or (cur_mode == "prose" and mode == "complexity"):
            cur_buf.append(ln)
        elif cur_mode == "code" and mode in ("complexity", "prose"):
            sub = "".join(cur_buf).rstrip("\n")
            if sub.strip():
                lead = len(sub) - len(sub.lstrip("\n"))
                groups.append(("code_or_pseudo", sub.lstrip("\n"), cur_start + lead, cur_start + len(sub)))
            cur_mode = mode
            cur_start = pos
            cur_buf = [ln]
        elif cur_mode in ("prose", "complexity") and mode == "code":
            sub = "".join(cur_buf).rstrip("\n")
            if sub.strip():
                lead = len(sub) - len(sub.lstrip("\n"))
                groups.append(("text", sub.lstrip("\n"), cur_start + lead, cur_start + len(sub)))
            cur_mode = mode
            cur_start = pos
            cur_buf = [ln]
        else:
            cur_buf.append(ln)
        pos += len(ln)

    if cur_buf:
        sub = "".join(cur_buf).rstrip("\n")
        if sub.strip():
            lead = len(sub) - len(sub.lstrip("\n"))
            groups.append((cur_mode or blk.raw_kind, sub.lstrip("\n"), cur_start + lead, cur_start + len(sub)))

    if not groups:
        return [(blk.raw_kind, blk.text, blk.char_start, blk.char_end)]
    return groups


def classify_text_unit(
    text: str,
    *,
    raw_kind: str,
    is_bold: bool = False,
    is_monospace: bool = False,
) -> str:
    """Apply strict precedence rules: table > code > pseudo > heading > complexity > steps > def > formula > algo_expl > narrative."""
    s = text.strip()
    if raw_kind == "table":
        return "table"

    lines = [ln for ln in s.splitlines() if ln.strip()]
    word_count = len(s.split())

    # 2. Code block (checked before generic bold/size heading so slide code comments/lines are not labeled heading)
    code_hits = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
    indented_hits = sum(1 for ln in lines if ln.startswith(("  ", "\t")))
    if raw_kind == "code" or is_monospace:
        return "code_block"
    if (len(lines) >= 2 and (code_hits >= 2 or (code_hits >= 1 and indented_hits >= 1))) or (
        len(lines) == 1 and s.startswith(("def ", "class ", "struct ", "void ", "int "))
    ):
        return "code_block"

    # 3. Pseudocode
    pseudo_hits = sum(1 for ln in lines if _PSEUDO_LINE_RE.search(ln))
    if (len(lines) >= 2 and pseudo_hits >= 2) or (pseudo_hits >= 1 and indented_hits >= 1):
        return "pseudocode"

    # 4. Heading (explicit style/numbered/bold short non-bullet, non-code title)
    if raw_kind == "heading":
        return "heading"
    if (
        len(lines) <= 2
        and word_count <= 12
        and not _PROSE_TERMINATOR_RE.search(s)
        and not s.startswith(("-", "*", "•", "# "))
        and not _STEP_LINE_RE.match(s)
        and not _FORMULA_RE.search(s)
    ):
        if _HEADING_NUM_RE.match(s) or is_bold:
            return "heading"

    # 5. Complexity analysis
    if _COMPLEXITY_RE.search(s) and word_count <= 85:
        if s.lower().startswith(("độ phức tạp", "time complexity", "space complexity")) or len(re.findall(r"[OΘΩ]\s*\([^)]+\)", s)) >= 1:
            return "complexity_analysis"

    # 6. Algorithm steps
    step_hits = sum(1 for ln in lines if _STEP_LINE_RE.match(ln))
    if step_hits >= 2 or raw_kind == "list_item" or (len(lines) == 1 and _STEP_LINE_RE.match(s)):
        return "algorithm_steps"

    # 7. Definition
    if _DEF_MARKER_RE.search(s):
        return "definition"

    # 8. Formula (includes compact equations / index assignments like i=0, k=i+1, Z(S)=[...])
    formula_hits = len(_FORMULA_RE.findall(s))
    if formula_hits >= 2 or (formula_hits >= 1 and word_count <= 35):
        return "formula"

    # 9. Algorithm explanation
    if _ALGO_EXPL_RE.match(s):
        return "algorithm_explanation"

    # 10. Default narrative
    return "narrative"


def _split_preserving_origins(
    text: str,
    origins: list[_CharOrigin],
    pattern: re.Pattern[str],
    *,
    preserve_indent: bool = False,
) -> list[tuple[str, list[_CharOrigin]]]:
    """Split *text* at regex matches while preserving exact per-character origins."""
    parts: list[tuple[str, list[_CharOrigin]]] = []
    last = 0
    strip_chars = "\r\n" if preserve_indent else None

    def append_frag(start_idx: int, end_idx: int) -> None:
        frag = text[start_idx:end_idx]
        cleaned = frag.strip(strip_chars) if strip_chars is not None else frag.strip()
        if not cleaned.strip():
            return
        lead = len(frag) - len(frag.lstrip(strip_chars) if strip_chars is not None else frag.lstrip())
        trail = len(frag) - len(frag.rstrip(strip_chars) if strip_chars is not None else frag.rstrip())
        s = start_idx + lead
        e = end_idx - trail
        parts.append((text[s:e], origins[s:e]))

    for m in pattern.finditer(text):
        end = m.end()
        if end > last:
            append_frag(last, end)
            last = end
    if last < len(text):
        append_frag(last, len(text))
    return parts if parts else [(text, origins)]


def _split_line_level_origins(
    text: str,
    origins: list[_CharOrigin],
    *,
    by_function: bool = False,
) -> list[tuple[str, list[_CharOrigin]]]:
    """Split multiline code/pseudocode/steps at function/step boundaries preserving per-char origins."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return [(text, origins)]

    groups: list[tuple[str, list[_CharOrigin]]] = []
    cur_start = 0
    pos = 0

    func_re = re.compile(r"^(?:def |class |void |int |bool |struct |Algorithm |Procedure |Function )", re.IGNORECASE)
    step_re = re.compile(r"^(?:Bước\s*\d+|Step\s*\d+|\d+[\.\)]\s+)", re.IGNORECASE)

    for idx, ln in enumerate(lines):
        split_here = False
        if idx > 0 and pos > cur_start:
            if by_function and func_re.match(ln):
                split_here = True
            elif not by_function and step_re.match(ln.lstrip()):
                split_here = True
        if split_here:
            joined = text[cur_start:pos].rstrip("\n")
            if joined.strip():
                groups.append((joined, origins[cur_start : cur_start + len(joined)]))
            cur_start = pos
        pos += len(ln)

    if pos > cur_start:
        joined = text[cur_start:pos].rstrip("\n")
        if joined.strip():
            groups.append((joined, origins[cur_start : cur_start + len(joined)]))
    return groups if groups else [(text, origins)]


def _assemble_text(heading_prefix: str | None, table_header: str | None, overlap: str, body: str) -> str:
    parts = [p for p in (heading_prefix, table_header, overlap, body) if p]
    return "\n".join(parts)


def _token_span_window_split(
    unit_text: str,
    origins: list[_CharOrigin] | None,
    page_no: int,
    tokenizer: Any,
    heading_prefix: str | None,
    table_header: str | None,
    *,
    fallback_spans: list[SourceSpan] | None = None,
) -> list[_BodyPiece]:
    """Deterministic guaranteed-progress fallback splitter enforcing final_text <= 128 tokens."""
    hard_limit = ADAPTIVE_CONFIG_V1["hard_max_total_embed_tokens"]
    pieces: list[_BodyPiece] = []
    pos = 0
    n = len(unit_text)

    while pos < n:
        while pos < n and unit_text[pos].isspace():
            pos += 1
        if pos >= n:
            break

        lo = pos + 1
        hi = n
        best_end = pos + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cand_body = unit_text[pos:mid].rstrip()
            if not cand_body:
                lo = mid + 1
                continue
            cand_full = _assemble_text(heading_prefix, table_header, "", cand_body)
            if total_emb_tokens(tokenizer, cand_full) <= hard_limit:
                best_end = mid
                lo = mid + 1
            else:
                hi = mid - 1

        snap_end = best_end
        if snap_end < n and not unit_text[snap_end].isspace():
            ws = unit_text.rfind(" ", pos + 1, best_end)
            nl = unit_text.rfind("\n", pos + 1, best_end)
            cut = max(ws, nl)
            if cut > pos + max(8, (best_end - pos) // 3):
                snap_end = cut

        raw_slice = unit_text[pos:snap_end]
        chunk_slice = raw_slice.strip()
        if not chunk_slice:
            snap_end = max(pos + 1, best_end)
            raw_slice = unit_text[pos:snap_end]
            chunk_slice = raw_slice.strip() or unit_text[pos : pos + 1]

        while total_emb_tokens(tokenizer, _assemble_text(heading_prefix, table_header, "", chunk_slice)) > hard_limit and len(chunk_slice) > 1:
            snap_end -= 1
            raw_slice = unit_text[pos:snap_end]
            chunk_slice = raw_slice.strip()
        if snap_end <= pos:
            snap_end = pos + 1
            raw_slice = unit_text[pos:snap_end]
            chunk_slice = raw_slice

        lead = len(raw_slice) - len(raw_slice.lstrip())
        s_idx = pos + lead
        e_idx = s_idx + len(chunk_slice)

        if origins is not None:
            piece_orig = origins[s_idx:e_idx]
            piece_spans = _origins_to_spans(page_no, piece_orig, chunk_slice)
        else:
            piece_orig = None
            piece_spans = list(fallback_spans or [])

        pieces.append(_BodyPiece(
            text=chunk_slice,
            spans=piece_spans,
            origins=piece_orig,
            is_forced_split=True,
            fallback_reason="oversized_unit_token_span",
        ))
        pos = snap_end

    return pieces


def _split_table_unit(
    unit: _TypedUnit,
    page_no: int,
    tokenizer: Any,
    heading_prefix: str | None,
) -> list[_BodyPiece]:
    """Split table along complete rows, repeating source-backed table header and linearizing oversized rows."""
    hard_limit = ADAPTIVE_CONFIG_V1["hard_max_total_embed_tokens"]
    soft_max = ADAPTIVE_CONTENT_BUDGETS["table"].soft_max
    lines = unit.text.splitlines()
    if len(lines) < 3 or not unit.table_header:
        return _split_non_table_unit(unit, page_no, "table", tokenizer, heading_prefix, None)

    raw_header = "\n".join(lines[:2])
    hdr_toks = len(tokenizer(raw_header, add_special_tokens=False, truncation=False)["input_ids"])
    header_used = raw_header
    hdr_truncated = False
    if hdr_toks > ADAPTIVE_CONFIG_V1["table_header_max_tokens"]:
        header_used, hdr_truncated = _clamp_text_tokens(
            tokenizer, raw_header, ADAPTIVE_CONFIG_V1["table_header_max_tokens"]
        )

    base_span = unit.spans[0]
    header_span: tuple[str, int, int] | None = None
    if unit.row_spans and len(unit.row_spans) >= 2:
        hs = unit.row_spans[0][0]
        he = unit.row_spans[1][1]
        if he > hs:
            header_span = (base_span.block_id, hs, he)

    col_names = [c.strip() for c in lines[0].strip("| ").split("|")]
    data_rows = lines[2:]
    row_offsets = (unit.row_spans[2:] if unit.row_spans and len(unit.row_spans) >= len(lines) else [])

    pieces: list[_BodyPiece] = []
    cur_rows: list[str] = []
    cur_spans: list[SourceSpan] = []

    def row_span_at(r_idx: int) -> SourceSpan:
        if r_idx < len(row_offsets):
            rs, re_ = row_offsets[r_idx]
            return SourceSpan(base_span.page, base_span.block_id, rs, re_)
        return base_span

    def fits_with(body_text: str, hdr: str | None = header_used) -> bool:
        assembled = _assemble_text(heading_prefix, hdr, "", body_text)
        return total_emb_tokens(tokenizer, assembled) <= hard_limit

    for r_idx, row_line in enumerate(data_rows):
        r_span = row_span_at(r_idx)
        cand_rows = [*cur_rows, row_line]
        cand_body = "\n".join(cand_rows)
        cand_toks = total_emb_tokens(tokenizer, _assemble_text(heading_prefix, header_used, "", cand_body))
        if cur_rows and (cand_toks > soft_max + 2 or not fits_with(cand_body)):
            pieces.append(_BodyPiece(
                text="\n".join(cur_rows),
                spans=cur_spans,
                synthetic_override=header_used,
                table_header_span=header_span,
                table_header_truncated=hdr_truncated,
                fallback_reason="table_header_clamped" if hdr_truncated else None,
            ))
            cur_rows = []
            cur_spans = []

        if fits_with(row_line):
            cur_rows.append(row_line)
            cur_spans.append(r_span)
        else:
            if cur_rows:
                pieces.append(_BodyPiece(
                    text="\n".join(cur_rows),
                    spans=cur_spans,
                    synthetic_override=header_used,
                    table_header_span=header_span,
                    table_header_truncated=hdr_truncated,
                ))
                cur_rows = []
                cur_spans = []
            # Single oversized table row -> linearize as "Col: Val" without dropping cells
            cells = [c.strip() for c in row_line.strip("| ").split("|")]
            pairs = [
                f"{col_names[i] if i < len(col_names) else f'Col{i+1}'}: {val}"
                for i, val in enumerate(cells) if val
            ]
            linearized = "; ".join(pairs) if pairs else row_line
            forced_parts = _token_span_window_split(
                linearized,
                None,
                page_no,
                tokenizer,
                heading_prefix,
                None,
                fallback_spans=[r_span],
            )
            for fp in forced_parts:
                fp.fallback_reason = "oversized_table_row_linearized"
                fp.synthetic_override = ""
                fp.table_header_span = header_span
                fp.table_header_truncated = True
            pieces.extend(forced_parts)

    if cur_rows:
        pieces.append(_BodyPiece(
            text="\n".join(cur_rows),
            spans=cur_spans,
            synthetic_override=header_used,
            table_header_span=header_span,
            table_header_truncated=hdr_truncated,
            fallback_reason="table_header_clamped" if hdr_truncated else None,
        ))
    return pieces


def _split_non_table_unit(
    unit: _TypedUnit,
    page_no: int,
    content_type: str,
    tokenizer: Any,
    heading_prefix: str | None,
    table_header: str | None,
) -> list[_BodyPiece]:
    """Apply content-specific boundary splitting with exact per-character origin propagation."""
    hard_limit = ADAPTIVE_CONFIG_V1["hard_max_total_embed_tokens"]
    soft_max = ADAPTIVE_CONTENT_BUDGETS[content_type].soft_max

    def fits_soft_and_hard(txt: str) -> bool:
        toks = total_emb_tokens(tokenizer, _assemble_text(heading_prefix, table_header, "", txt))
        return toks <= min(hard_limit, soft_max + 2)

    if fits_soft_and_hard(unit.text):
        spans = _origins_to_spans(page_no, unit.origins, unit.text) or unit.spans
        return [_BodyPiece(text=unit.text, spans=spans, origins=unit.origins)]

    # Generate boundary candidates ordered from coarsest to finest with exact origins
    if content_type in ("code_block", "pseudocode"):
        step1 = _split_line_level_origins(unit.text, unit.origins, by_function=True)
        expanded: list[tuple[str, list[_CharOrigin]]] = []
        for txt, orig in step1:
            if fits_soft_and_hard(txt):
                expanded.append((txt, orig))
            else:
                expanded.extend(_split_preserving_origins(txt, orig, re.compile(r"\n+"), preserve_indent=True))
        raw_parts = expanded
    elif content_type == "algorithm_steps":
        step1 = _split_line_level_origins(unit.text, unit.origins, by_function=False)
        raw_parts = []
        for txt, orig in step1:
            if fits_soft_and_hard(txt):
                raw_parts.append((txt, orig))
            else:
                raw_parts.extend(_split_preserving_origins(txt, orig, re.compile(r"(?<=[.?!;])\s+|\n+")))
    else:
        raw_parts = _split_preserving_origins(unit.text, unit.origins, re.compile(r"\n{2,}"))
        refined: list[tuple[str, list[_CharOrigin]]] = []
        for txt, orig in raw_parts:
            if fits_soft_and_hard(txt):
                refined.append((txt, orig))
            else:
                sents = _split_preserving_origins(txt, orig, re.compile(r"(?<=[.?!])\s+|\n+"))
                for stxt, sorig in sents:
                    if fits_soft_and_hard(stxt):
                        refined.append((stxt, sorig))
                    else:
                        refined.extend(_split_preserving_origins(stxt, sorig, re.compile(r"(?<=[;:,])\s+")))
        raw_parts = refined

    pieces: list[_BodyPiece] = []
    cur_parts: list[tuple[str, list[_CharOrigin]]] = []
    join_sep = "\n" if content_type in ("code_block", "pseudocode", "algorithm_steps") else " "

    def flush_cur() -> None:
        if not cur_parts:
            return
        j_txt, j_orig = _join_tracked_parts(cur_parts, join_sep)
        j_spans = _origins_to_spans(page_no, j_orig, j_txt)
        pieces.append(_BodyPiece(text=j_txt, spans=j_spans, origins=j_orig))
        cur_parts.clear()

    for ptxt, porig in raw_parts:
        candidate = join_sep.join([*(t for t, _ in cur_parts), ptxt]) if cur_parts else ptxt
        if cur_parts and not fits_soft_and_hard(candidate):
            flush_cur()

        if total_emb_tokens(tokenizer, _assemble_text(heading_prefix, table_header, "", ptxt)) <= hard_limit:
            cur_parts.append((ptxt, porig))
        else:
            flush_cur()
            pieces.extend(_token_span_window_split(ptxt, porig, page_no, tokenizer, heading_prefix, table_header))

    flush_cur()
    return pieces


def _extract_overlap_tail_with_spans(
    piece: _BodyPiece,
    page_no: int,
    tokenizer: Any,
    max_overlap_tokens: int,
) -> tuple[str, list[SourceSpan]]:
    """Select an overlap suffix on clean word boundaries with exact canonical SourceSpans."""
    text = piece.text
    if not text or max_overlap_tokens <= 0:
        return "", []
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return "", []

    # Try last sentence first if short enough
    stripped = text.strip()
    sent_parts = re.split(r"(?<=[.?!])\s+", stripped)
    chosen_start: int | None = None
    chosen_end: int | None = None
    if len(sent_parts) >= 2 and sent_parts[-1]:
        cand = sent_parts[-1]
        if len(tokenizer(cand, add_special_tokens=False, truncation=False)["input_ids"]) <= max_overlap_tokens:
            idx = text.rfind(cand)
            if idx >= 0:
                chosen_start = idx
                chosen_end = idx + len(cand)

    if chosen_start is None or chosen_end is None:
        end_pos = matches[-1].end()
        best_start: int | None = None
        for m in reversed(matches):
            cand = text[m.start() : end_pos]
            if len(tokenizer(cand, add_special_tokens=False, truncation=False)["input_ids"]) <= max_overlap_tokens:
                best_start = m.start()
            else:
                break
        if best_start is None:
            return "", []
        chosen_start, chosen_end = best_start, end_pos

    ov_text = text[chosen_start:chosen_end]
    if piece.origins is not None:
        ov_spans = _origins_to_spans(page_no, piece.origins[chosen_start:chosen_end], ov_text)
    elif piece.spans:
        last_sp = piece.spans[-1]
        ov_end = last_sp.char_end
        ov_start = max(last_sp.char_start, ov_end - len(ov_text))
        ov_spans = [SourceSpan(page_no, last_sp.block_id, ov_start, ov_end)]
    else:
        ov_spans = []
    return ov_text, ov_spans


def _code_signature_header(
    piece: _BodyPiece,
    page_no: int,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[str, bool, list[SourceSpan]]:
    """Extract leading signature/Algorithm line and its exact canonical source span."""
    pos = 0
    for ln in piece.text.splitlines(keepends=True):
        st = ln.strip()
        if st:
            clamped, truncated = _clamp_text_tokens(tokenizer, st, max_tokens)
            lead = len(ln) - len(ln.lstrip())
            s = pos + lead
            e = s + len(st)
            spans = _origins_to_spans(page_no, piece.origins[s:e], st) if piece.origins is not None else []
            return clamped, truncated, spans
        pos += len(ln)
    return "", False, []


def chunk_document_adaptive(
    doc: CanonicalDocument,
    emb_tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute deterministic Structure-aware Adaptive Chunking (ADAPTIVE_CONFIG_V1)."""
    cfg_hash = get_profile_config_hash(PROFILE_B_ID)
    hard_limit = ADAPTIVE_CONFIG_V1["hard_max_total_embed_tokens"]
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    chunk_idx = 0

    for page in doc.pages:
        if not page.canonical_text.strip():
            continue

        # 1. Mixed-content splitting + classification with exact per-character origins
        units: list[_TypedUnit] = []
        for blk in page.blocks:
            sub_items = _split_mixed_block(blk, page.page)
            for sub_idx, (kind_hint, sub_txt, sub_s, sub_e) in enumerate(sub_items):
                if len(sub_txt.split()) < 1:
                    continue
                ctype = classify_text_unit(
                    sub_txt,
                    raw_kind="code" if kind_hint == "code_or_pseudo" else ("table" if blk.raw_kind == "table" else blk.raw_kind),
                    is_bold=blk.is_bold,
                    is_monospace=blk.is_monospace,
                )
                sub_id = blk.block_id if len(sub_items) == 1 else f"{blk.block_id}:s{sub_idx}"
                sp = SourceSpan(page.page, blk.block_id, sub_s, sub_e)
                units.append(_TypedUnit(
                    block_id=sub_id,
                    content_type=ctype,
                    text=sub_txt,
                    spans=[sp],
                    origins=_make_span_origins(sp, len(sub_txt)),
                    table_header=blk.table_header,
                    row_spans=blk.row_spans,
                ))

        # 2. Compatible block grouping within the same (document_id, page, section)
        grouped: list[tuple[str | None, _HeadingPrefixInfo | None, str, list[_TypedUnit]]] = []
        current_section: str | None = None
        pending_heading: _TypedUnit | None = None

        for unit in units:
            if unit.content_type == "heading":
                sec_name = unit.text.lstrip("#").strip()
                if pending_heading is not None and len(emb_tokenizer(f"{pending_heading.text}\n{unit.text}", add_special_tokens=True)["input_ids"]) <= 40:
                    merged_txt = f"{pending_heading.text}\n{unit.text}"
                    merged_orig: list[_CharOrigin] = [*pending_heading.origins, None, *unit.origins]
                    pending_heading = _TypedUnit(
                        block_id=pending_heading.block_id,
                        content_type="heading",
                        text=merged_txt,
                        spans=[*pending_heading.spans, *unit.spans],
                        origins=merged_orig,
                    )
                    current_section = f"{current_section} / {sec_name}" if current_section else sec_name
                else:
                    if pending_heading is not None:
                        grouped.append((current_section, None, "heading", [pending_heading]))
                    current_section = sec_name
                    pending_heading = unit
                continue

            # Discard tiny meaningless noise (< min_prose_keep_tokens) unless formula/code/table
            if unit.content_type == "narrative" and len(unit.text.split()) < ADAPTIVE_CONFIG_V1["min_prose_keep_tokens"]:
                if not re.search(r"[a-zA-ZÀ-ỹ]{2,}", unit.text) and not _FORMULA_RE.search(unit.text):
                    for sp in unit.spans:
                        dropped.append({
                            "document_id": doc.document_id,
                            "page": page.page,
                            "char_start": sp.char_start,
                            "char_end": sp.char_end,
                            "reason": "noise_lt_3_tokens",
                        })
                    continue

            # Try grouping with last compatible unit under same section when no new heading intervened
            can_merge = False
            if pending_heading is None and grouped:
                prev_sec, prev_hp_info, prev_type, prev_list = grouped[-1]
                prev_hp_text = prev_hp_info.text if prev_hp_info else None
                if prev_sec == current_section:
                    is_pure_symbol = not _INDEXABLE_TERM_RE.search(unit.text)
                    compatible_pair = (
                        (prev_type == "narrative" and unit.content_type in ("narrative", "formula"))
                        or (prev_type == "algorithm_steps" and unit.content_type == "algorithm_steps")
                        or (prev_type == "algorithm_explanation" and unit.content_type in ("algorithm_explanation", "algorithm_steps", "narrative", "formula"))
                        or (prev_type == "code_block" and unit.content_type == "code_block" and not unit.text.lstrip().startswith(("def ", "class ", "void ", "int ")))
                        or (prev_type == "pseudocode" and unit.content_type == "pseudocode" and not _PSEUDO_LINE_RE.match(unit.text.lstrip()))
                        or (prev_type == "definition" and unit.content_type == "definition" and not _DEF_MARKER_RE.search(unit.text))
                        or (is_pure_symbol and prev_type in ("code_block", "pseudocode", "formula"))
                    )
                    if compatible_pair:
                        joiner = "\n" if prev_type in ("algorithm_steps", "code_block", "pseudocode") else "\n\n"
                        candidate_body = joiner.join([*(u.text for u in prev_list), unit.text])
                        target_cap = ADAPTIVE_CONTENT_BUDGETS[prev_type].soft_max
                        cand_full = _assemble_text(prev_hp_text, None, "", candidate_body)
                        if total_emb_tokens(emb_tokenizer, cand_full) <= min(hard_limit, target_cap + 2):
                            can_merge = True
                            prev_list.append(unit)

            if not can_merge:
                hp_info: _HeadingPrefixInfo | None = None
                target_unit = unit
                if pending_heading is not None:
                    if unit.content_type != "table":
                        folded = f"{pending_heading.text}\n{unit.text}"
                        target_cap = ADAPTIVE_CONTENT_BUDGETS[unit.content_type].soft_max
                        if total_emb_tokens(emb_tokenizer, folded) <= min(hard_limit, target_cap + 2):
                            target_unit = _TypedUnit(
                                block_id=unit.block_id,
                                content_type=unit.content_type,
                                text=folded,
                                spans=[*pending_heading.spans, *unit.spans],
                                origins=[*pending_heading.origins, None, *unit.origins],
                                table_header=unit.table_header,
                                row_spans=unit.row_spans,
                            )
                        else:
                            hp_text, hp_trunc = _clamp_text_tokens(
                                emb_tokenizer,
                                pending_heading.text,
                                ADAPTIVE_CONFIG_V1["heading_prefix_max_tokens"],
                            )
                            hp_info = _HeadingPrefixInfo(
                                text=hp_text,
                                truncated=hp_trunc,
                                spans=list(pending_heading.spans),
                                original_text=pending_heading.text,
                            )
                    else:
                        hp_text, hp_trunc = _clamp_text_tokens(
                            emb_tokenizer,
                            pending_heading.text,
                            ADAPTIVE_CONFIG_V1["heading_prefix_max_tokens"],
                        )
                        hp_info = _HeadingPrefixInfo(
                            text=hp_text,
                            truncated=hp_trunc,
                            spans=list(pending_heading.spans),
                            original_text=pending_heading.text,
                        )
                grouped.append((current_section, hp_info, unit.content_type, [target_unit]))
                pending_heading = None

        if pending_heading is not None:
            grouped.append((current_section, None, "heading", [pending_heading]))

        # 3. Split groups, apply adaptive overlap, enforce hard token limit <= 128 on final_text
        prev_chunk_id: str | None = None
        for section_label, hp_info, g_type, g_units in grouped:
            heading_prefix = hp_info.text if hp_info else None
            if len(g_units) == 1 and g_type == "table":
                pieces = _split_table_unit(g_units[0], page.page, emb_tokenizer, heading_prefix)
            else:
                group_joiner = "\n" if g_type in ("algorithm_steps", "code_block", "pseudocode") else "\n\n"
                combined_body, combined_origins = _join_tracked_parts(
                    [(u.text, u.origins) for u in g_units],
                    group_joiner,
                )
                combined_spans = [sp for u in g_units for sp in u.spans]
                merged_unit = _TypedUnit(
                    block_id=g_units[0].block_id,
                    content_type=g_type,
                    text=combined_body,
                    spans=combined_spans,
                    origins=combined_origins,
                )
                pieces = _split_non_table_unit(merged_unit, page.page, g_type, emb_tokenizer, heading_prefix, None)

            ov_budget = ADAPTIVE_CONTENT_BUDGETS[g_type].overlap_max_tokens
            if len(pieces) > 1 and g_type in ("code_block", "pseudocode"):
                code_sig, code_sig_trunc, code_sig_spans = _code_signature_header(
                    pieces[0], page.page, emb_tokenizer, ov_budget
                )
            else:
                code_sig, code_sig_trunc, code_sig_spans = "", False, []

            any_hp_kept = False
            for p_idx, piece in enumerate(pieces):
                syn_segs: list[SyntheticSegment] = []
                active_hp = heading_prefix
                if active_hp and hp_info is not None:
                    for hsp in hp_info.spans:
                        seg_len = len(active_hp) if len(hp_info.spans) == 1 else min(hsp.char_end - hsp.char_start, len(active_hp))
                        syn_segs.append(SyntheticSegment(
                            kind="heading_prefix",
                            source_block_id=hsp.block_id,
                            text_len=seg_len,
                            truncated=hp_info.truncated,
                            source_backed=True,
                            page=hsp.page,
                            char_start=hsp.char_start,
                            char_end=hsp.char_end,
                        ))

                active_tbl_hdr = piece.synthetic_override if piece.synthetic_override is not None else None
                if active_tbl_hdr:
                    hdr_sp = piece.table_header_span
                    syn_segs.append(SyntheticSegment(
                        kind="table_header",
                        source_block_id=hdr_sp[0] if hdr_sp else g_units[0].block_id,
                        text_len=len(active_tbl_hdr),
                        truncated=piece.table_header_truncated,
                        source_backed=bool(hdr_sp),
                        page=page.page if hdr_sp else None,
                        char_start=hdr_sp[1] if hdr_sp else None,
                        char_end=hdr_sp[2] if hdr_sp else None,
                    ))
                elif piece.fallback_reason == "oversized_table_row_linearized" and piece.table_header_span is not None:
                    hdr_sp = piece.table_header_span
                    syn_segs.append(SyntheticSegment(
                        kind="table_header",
                        source_block_id=hdr_sp[0],
                        text_len=0,
                        truncated=True,
                        source_backed=True,
                        page=page.page,
                        char_start=hdr_sp[1],
                        char_end=hdr_sp[2],
                    ))

                overlap_text = ""
                overlap_segs: list[OverlapSegment] = []
                if p_idx > 0 and ov_budget > 0 and g_type != "table":
                    if g_type in ("code_block", "pseudocode"):
                        if code_sig and not piece.text.lstrip().startswith(code_sig.split("(")[0]):
                            overlap_text = code_sig
                            if code_sig_spans:
                                for csp in code_sig_spans:
                                    syn_segs.append(SyntheticSegment(
                                        kind="code_signature",
                                        source_block_id=csp.block_id,
                                        text_len=len(code_sig) if len(code_sig_spans) == 1 else min(csp.char_end - csp.char_start, len(code_sig)),
                                        truncated=code_sig_trunc,
                                        source_backed=True,
                                        page=csp.page,
                                        char_start=csp.char_start,
                                        char_end=csp.char_end,
                                    ))
                            else:
                                syn_segs.append(SyntheticSegment(
                                    kind="code_signature",
                                    source_block_id=g_units[0].block_id,
                                    text_len=len(code_sig),
                                    truncated=code_sig_trunc,
                                ))
                    else:
                        prev_piece = pieces[p_idx - 1]
                        overlap_text, ov_spans = _extract_overlap_tail_with_spans(
                            prev_piece, page.page, emb_tokenizer, ov_budget
                        )
                        if overlap_text:
                            if ov_spans:
                                for osp in ov_spans:
                                    overlap_segs.append(OverlapSegment(
                                        from_chunk_id=prev_chunk_id,
                                        page=osp.page,
                                        block_id=osp.block_id,
                                        char_start=osp.char_start,
                                        char_end=osp.char_end,
                                        text_len=osp.char_end - osp.char_start,
                                    ))
                            else:
                                overlap_segs.append(OverlapSegment(
                                    from_chunk_id=prev_chunk_id,
                                    page=page.page,
                                    block_id=None,
                                    char_start=None,
                                    char_end=None,
                                    text_len=len(overlap_text),
                                ))

                # Assemble and enforce mandatory final_text token budget check
                final_text = _assemble_text(active_hp, active_tbl_hdr, overlap_text, piece.text)
                if total_emb_tokens(emb_tokenizer, final_text) > hard_limit:
                    overlap_text = ""
                    overlap_segs = []
                    syn_segs = [s for s in syn_segs if s.kind != "code_signature"]
                    final_text = _assemble_text(active_hp, active_tbl_hdr, "", piece.text)

                if total_emb_tokens(emb_tokenizer, final_text) > hard_limit:
                    active_hp = None
                    syn_segs = [s for s in syn_segs if s.kind != "heading_prefix"]
                    final_text = _assemble_text(None, active_tbl_hdr, "", piece.text)

                if total_emb_tokens(emb_tokenizer, final_text) > hard_limit:
                    emergency = _token_span_window_split(
                        piece.text,
                        piece.origins,
                        page.page,
                        emb_tokenizer,
                        None,
                        None,
                        fallback_spans=piece.spans,
                    )
                    for ep in emergency:
                        ep_toks = total_emb_tokens(emb_tokenizer, ep.text)
                        assert ep_toks <= hard_limit
                        if not _INDEXABLE_TERM_RE.search(ep.text):
                            for sp in ep.spans:
                                dropped.append({
                                    "document_id": doc.document_id,
                                    "page": page.page,
                                    "char_start": sp.char_start,
                                    "char_end": sp.char_end,
                                    "reason": "no_bm25_indexable_terms",
                                })
                            continue
                        rec = build_chunk_record(
                            doc=doc,
                            page=page.page,
                            section=section_label,
                            text=ep.text,
                            chunk_index=chunk_idx,
                            profile_id=PROFILE_B_ID,
                            chunking_config_hash=cfg_hash,
                            content_type=g_type,
                            language=page.language,
                            source_spans=ep.spans,
                            overlap_segments=[],
                            synthetic_segments=[],
                            fallback=FallbackInfo(True, ep.fallback_reason or "hard_budget_emergency_split"),
                            emb_token_count=ep_toks,
                            body_text_for_hash=ep.text,
                        )
                        records.append(rec)
                        prev_chunk_id = rec["chunk_id"]
                        chunk_idx += 1
                    continue

                tok_count = total_emb_tokens(emb_tokenizer, final_text)
                assert tok_count <= hard_limit, f"Token budget violated: {tok_count} > {hard_limit}"
                if not _INDEXABLE_TERM_RE.search(final_text):
                    for sp in piece.spans:
                        dropped.append({
                            "document_id": doc.document_id,
                            "page": page.page,
                            "char_start": sp.char_start,
                            "char_end": sp.char_end,
                            "reason": "no_bm25_indexable_terms",
                        })
                    continue

                if active_hp:
                    any_hp_kept = True
                rec = build_chunk_record(
                    doc=doc,
                    page=page.page,
                    section=section_label,
                    text=final_text,
                    chunk_index=chunk_idx,
                    profile_id=PROFILE_B_ID,
                    chunking_config_hash=cfg_hash,
                    content_type=g_type,
                    language=page.language,
                    source_spans=piece.spans,
                    overlap_segments=overlap_segs,
                    synthetic_segments=syn_segs,
                    fallback=FallbackInfo(piece.is_forced_split, piece.fallback_reason),
                    emb_token_count=tok_count,
                    body_text_for_hash=piece.text,
                )
                records.append(rec)
                prev_chunk_id = rec["chunk_id"]
                chunk_idx += 1

            # Ensure a pending heading is never silently lost if active_hp was dropped from every piece
            if hp_info is not None and not any_hp_kept:
                if _INDEXABLE_TERM_RE.search(hp_info.original_text):
                    hp_toks = total_emb_tokens(emb_tokenizer, hp_info.original_text)
                    if hp_toks <= hard_limit:
                        rec = build_chunk_record(
                            doc=doc,
                            page=page.page,
                            section=section_label,
                            text=hp_info.original_text,
                            chunk_index=chunk_idx,
                            profile_id=PROFILE_B_ID,
                            chunking_config_hash=cfg_hash,
                            content_type="heading",
                            language=page.language,
                            source_spans=hp_info.spans,
                            overlap_segments=[],
                            synthetic_segments=[],
                            fallback=FallbackInfo(False, "heading_prefix_emitted_standalone"),
                            emb_token_count=hp_toks,
                            body_text_for_hash=hp_info.original_text,
                        )
                        records.append(rec)
                        prev_chunk_id = rec["chunk_id"]
                        chunk_idx += 1
                    else:
                        for hsp in hp_info.spans:
                            dropped.append({
                                "document_id": doc.document_id,
                                "page": page.page,
                                "char_start": hsp.char_start,
                                "char_end": hsp.char_end,
                                "reason": "heading_prefix_over_budget",
                            })
                else:
                    for hsp in hp_info.spans:
                        dropped.append({
                            "document_id": doc.document_id,
                            "page": page.page,
                            "char_start": hsp.char_start,
                            "char_end": hsp.char_end,
                            "reason": "no_bm25_indexable_terms",
                        })

    return records, dropped
