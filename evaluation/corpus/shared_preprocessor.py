"""Shared deterministic PDF/DOCX preprocessing adapter for fair Profile A vs Profile B evaluation."""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Ensure HuggingFace offline safety before any optional transformer/tokenizer import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_processing.text_cleaner import (
    _CTRL_CHARS_RE,
    _HEADER_FOOTER_PATTERNS,
    _HSPACE_RE,
    _HYPHEN_BREAK_RE,
    _MULTI_NL_RE,
    _UNICODE_BULLET_RE,
)

from evaluation.corpus.provenance import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    compute_content_hash,
    make_document_id,
    sha256_bytes,
)

PREPROCESSOR_VERSION = "shared_preprocessor_v1"

# Vietnamese-specific accented Latin characters (excluding standard ASCII letters)
_VI_CHAR_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)
_EN_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
_CODE_LINE_HINT_RE = re.compile(
    r"(?:^\s*(?:def |class |struct |void |int |bool |char |float |double |return\b|for\s*\(|while\s*\(|if\s*\(|else\b|elif\b|Procedure\b|Algorithm\b|Function\b|Input:|Output:|//|#\s*\w))"
    r"|[{};]|<-|:=|\b[A-Za-z_]\w*\([^)]*\)"
)
_FORMULA_HINT_RE = re.compile(r"[∑∏∀∃≤≥≠≈∈∪⌊⌋⌈⌉ΘΩ]|T\s*\([^)]+\)\s*=")


def lang_hint_from_path(path: Path) -> str:
    """Infer directory-level language hint; handles the existing 'end_docx' typo cleanly."""
    folder = path.parent.name.lower()
    if folder.startswith("vie"):
        return "vi"
    if folder.startswith("eng") or folder.startswith("end"):
        return "en"
    return "unknown"


def detect_language_deterministic(text: str, hint: str = "unknown") -> str:
    """Deterministic rule-based language classifier without external langdetect.

    Returns 'vi', 'en', 'mixed', or falls back to folder *hint* for code,
    formula, or short/ambiguous text.
    """
    sample = unicodedata.normalize("NFC", text or "")[:2000].strip()
    if not sample:
        return hint

    vi_chars = len(_VI_CHAR_RE.findall(sample))
    words = sample.split()
    alpha_words = [w for w in words if re.search(r"[a-zA-ZÀ-ỹ]", w)]
    en_words = len(_EN_WORD_RE.findall(sample))
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    code_lines = sum(1 for ln in lines if _CODE_LINE_HINT_RE.search(ln))
    formula_hits = len(_FORMULA_HINT_RE.findall(sample))

    # Ambiguous / symbol-heavy / mostly code or formula fallback to hint
    if not alpha_words or len(alpha_words) < 4:
        return hint
    if lines and code_lines / len(lines) >= 0.6 and vi_chars < 3:
        return hint if hint in ("vi", "en") else "code"
    if formula_hits >= 3 and len(alpha_words) < 12 and vi_chars < 2:
        return hint

    if vi_chars >= 4:
        # Mixed when Vietnamese prose co-occurs with substantial non-Vietnamese English prose
        if hint == "en" and en_words >= 25:
            return "mixed"
        vi_words = sum(1 for w in alpha_words if _VI_CHAR_RE.search(w))
        ascii_prose_words = sum(1 for w in alpha_words if re.fullmatch(r"[A-Za-z]{3,}", w))
        if vi_words >= 5 and ascii_prose_words >= 25 and ascii_prose_words > vi_words * 1.5:
            return "mixed"
        return "vi"

    if en_words >= 6 and len(alpha_words) >= 6:
        if hint == "vi" and any(ch in sample for ch in ("đ", "Đ")):
            return "mixed"
        return "en"

    return hint


def clean_canonical_block(raw_text: str, *, preserve_indent: bool = False) -> str:
    """Clean extracted text using production regex rules while optionally keeping indentation."""
    if not raw_text or not raw_text.strip():
        return ""
    text = unicodedata.normalize("NFC", raw_text)
    text = _CTRL_CHARS_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _UNICODE_BULLET_RE.sub("-", text)
    for pattern in _HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            cleaned_lines.append("")
            continue
        if preserve_indent or line.startswith("    ") or line.startswith("\t") or line.lstrip().startswith("|"):
            # Preserve leading indentation and Markdown pipe tables
            lead_len = len(line) - len(line.lstrip(" \t"))
            leading = line[:lead_len].replace("\t", "    ")
            body = _HSPACE_RE.sub(" ", line[lead_len:].rstrip())
            cleaned_lines.append(leading + body)
        else:
            cleaned_lines.append(_HSPACE_RE.sub(" ", line.strip()))

    joined = "\n".join(cleaned_lines)
    joined = _MULTI_NL_RE.sub("\n\n", joined)
    return joined.strip("\n")


def _format_markdown_table(rows: list[list[str | None]]) -> tuple[str, str | None, list[str]]:
    """Normalize a 2D table cell matrix into a clean Markdown pipe table."""
    clean_rows: list[list[str]] = []
    for row in rows:
        cells = [
            _HSPACE_RE.sub(" ", unicodedata.normalize("NFC", str(c or "")).replace("\n", " ").replace("|", "\\|").strip())
            for c in row
        ]
        if any(cells):
            clean_rows.append(cells)
    if len(clean_rows) < 2:
        # Degenerate 1-row matrix: flatten as text rather than broken pipe table
        flat = "\n".join(" | ".join(c for c in r if c) for r in clean_rows)
        return flat, None, [flat] if flat else []

    width = max(len(r) for r in clean_rows)
    padded = [r + [""] * (width - len(r)) for r in clean_rows]
    header_line = "| " + " | ".join(padded[0]) + " |"
    sep_line = "| " + " | ".join("---" for _ in range(width)) + " |"
    header_block = f"{header_line}\n{sep_line}"
    data_lines = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    full_table = "\n".join([header_block, *data_lines])
    return full_table, header_block, data_lines


def _extract_docx_pages(file_path: Path, doc_id: str, hint: str) -> list[CanonicalPage]:
    """Extract DOCX paragraphs and tables in exact XML body order."""
    from docx import Document

    doc = Document(str(file_path))
    para_map = {p._element: p for p in doc.paragraphs}
    table_map = {t._tbl: t for t in doc.tables}

    pages: list[CanonicalPage] = []
    cur_items: list[dict[str, Any]] = []
    cur_page = 1

    def flush_page() -> None:
        nonlocal cur_page, cur_items
        if not cur_items:
            return
        pages.append(_finalize_page(doc_id, cur_page, "docx", cur_items, hint))
        cur_items = []
        cur_page += 1

    for child in doc.element.body:
        if child in para_map:
            para = para_map[child]
            raw_text = para.text
            if not raw_text or not raw_text.strip():
                continue
            style_name = para.style.name if para.style else "Normal"
            if style_name.startswith("Heading 1") and cur_items:
                flush_page()

            is_pre = "preformatted" in style_name.lower() or "code" in style_name.lower()
            is_code_text = is_pre or raw_text.startswith(("    ", "\t")) or bool(_CODE_LINE_HINT_RE.search(raw_text))
            cleaned = clean_canonical_block(raw_text, preserve_indent=is_code_text)
            if not cleaned:
                continue

            if style_name == "Title" or style_name.startswith("Heading"):
                raw_kind = "heading"
            elif "list" in style_name.lower():
                raw_kind = "list_item"
            elif is_pre:
                raw_kind = "code"
            else:
                raw_kind = "paragraph"

            cur_items.append({
                "raw_kind": raw_kind,
                "text": cleaned,
                "style_name": style_name,
                "is_bold": style_name.startswith(("Heading", "Title")),
                "is_monospace": is_pre,
                "indent_level": float(len(cleaned) - len(cleaned.lstrip(" "))),
                "table_header": None,
            })

        elif child in table_map:
            tbl = table_map[child]
            matrix = [[cell.text for cell in row.cells] for row in tbl.rows]
            md_table, tbl_header, _ = _format_markdown_table(matrix)
            if not md_table:
                continue
            cur_items.append({
                "raw_kind": "table",
                "text": md_table,
                "style_name": "Table",
                "is_bold": False,
                "is_monospace": False,
                "indent_level": 0.0,
                "table_header": tbl_header,
            })

    flush_page()
    return pages


def _extract_pdf_pages(file_path: Path, doc_id: str, hint: str) -> list[CanonicalPage]:
    """Extract PDF reading-order blocks using PyMuPDF dict layout without slow find_tables()."""
    import pymupdf

    pages: list[CanonicalPage] = []
    with pymupdf.open(str(file_path)) as doc:
        for page_idx, page in enumerate(doc):
            page_dict = page.get_text("dict")
            raw_text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

            # Estimate baseline body font size and left margin on the page
            sizes: list[float] = []
            x0s: list[float] = []
            for blk in raw_text_blocks:
                for line in blk.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if spans:
                        x0s.append(float(line["bbox"][0]))
                        for sp in spans:
                            sizes.append(float(sp.get("size", 12.0)))
            body_font = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
            base_x0 = min(x0s) if x0s else 0.0

            candidates: list[tuple[float, float, dict[str, Any]]] = []
            for blk in raw_text_blocks:
                lines_out: list[str] = []
                blk_sizes: list[float] = []
                bold_chars = 0
                mono_chars = 0
                total_chars = 0
                line_indents: list[float] = []

                for line in blk.get("lines", []):
                    lb = tuple(line["bbox"])
                    spans = line.get("spans", [])
                    raw_line = "".join(s.get("text", "") for s in spans).rstrip()
                    if not raw_line.strip():
                        continue

                    x_offset = max(0.0, float(lb[0]) - base_x0)
                    line_indents.append(x_offset)
                    for sp in spans:
                        stxt = sp.get("text", "")
                        slen = len(stxt.strip())
                        if slen == 0:
                            continue
                        total_chars += slen
                        blk_sizes.append(float(sp.get("size", body_font)))
                        if int(sp.get("flags", 0)) & 16:
                            bold_chars += slen
                        font_name = (sp.get("font") or "").lower()
                        if any(m in font_name for m in ("mono", "courier", "consola", "code", "cmtt")):
                            mono_chars += slen

                    # Preserve geometric indentation for indented/code lines when >= 10pt past margin
                    if x_offset >= 10.0 and (_CODE_LINE_HINT_RE.search(raw_line) or mono_chars > 0):
                        spaces = min(16, max(2, int(round(x_offset / 6.0))))
                        lines_out.append((" " * spaces) + raw_line.lstrip())
                    else:
                        lines_out.append(raw_line)

                if not lines_out:
                    continue
                joined_raw = "\n".join(lines_out)
                is_mono = (total_chars > 0 and mono_chars / total_chars >= 0.5)
                has_code = is_mono or any(ln.startswith("  ") for ln in lines_out)
                cleaned = clean_canonical_block(joined_raw, preserve_indent=has_code)
                if not cleaned:
                    continue

                avg_size = sum(blk_sizes) / len(blk_sizes) if blk_sizes else body_font
                is_bold = total_chars > 0 and (bold_chars / total_chars) >= 0.7
                avg_indent = sum(line_indents) / len(line_indents) if line_indents else 0.0
                bb = blk["bbox"]
                first_stripped = cleaned.lstrip()
                looks_like_bullet_or_code = (
                    first_stripped.startswith(("-", "*", "•", "#"))
                    or bool(_CODE_LINE_HINT_RE.search(cleaned))
                    or cleaned.rstrip().endswith((".", "?", "!"))
                )
                is_pdf_heading = (
                    not looks_like_bullet_or_code
                    and not is_mono
                    and avg_size > body_font * 1.2
                    and len(cleaned.splitlines()) <= 2
                    and len(cleaned.split()) <= 12
                )
                candidates.append((bb[1], bb[0], {
                    "raw_kind": "heading" if is_pdf_heading else "text_block",
                    "text": cleaned,
                    "style_name": None,
                    "font_size": round(avg_size, 2),
                    "is_bold": is_bold,
                    "is_monospace": is_mono,
                    "indent_level": round(avg_indent, 2),
                    "table_header": None,
                }))

            if not candidates:
                continue
            candidates.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
            items = [c[2] for c in candidates]
            pages.append(_finalize_page(doc_id, page_idx + 1, "pdf", items, hint))

    return pages


def _finalize_page(
    doc_id: str,
    page_no: int,
    file_type: str,
    items: list[dict[str, Any]],
    hint: str,
) -> CanonicalPage:
    """Assemble CanonicalPage canonical_text with exact half-open [char_start, char_end) block bounds."""
    blocks: list[CanonicalBlock] = []
    parts: list[str] = []
    cursor = 0
    seen_records: set[tuple[int, int, str]] = set()
    dropped_dups = 0

    for idx, item in enumerate(items):
        btxt = item["text"]
        if not btxt.strip():
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        char_start = cursor
        char_end = cursor + len(btxt)
        chash = compute_content_hash(btxt)
        sig = (char_start, char_end, chash)
        if sig in seen_records:
            dropped_dups += 1
            if parts and parts[-1] == "\n\n":
                parts.pop()
                cursor -= 2
            continue
        seen_records.add(sig)

        row_spans: list[tuple[int, int]] = []
        if item["raw_kind"] == "table":
            line_pos = char_start
            for ln in btxt.splitlines(keepends=True):
                ln_clean = ln.rstrip("\r\n")
                if ln_clean:
                    row_spans.append((line_pos, line_pos + len(ln_clean)))
                line_pos += len(ln)

        parts.append(btxt)
        cursor = char_end
        blocks.append(CanonicalBlock(
            block_id=f"{doc_id}:p{page_no}:b{idx}",
            raw_kind=item["raw_kind"],
            text=btxt,
            char_start=char_start,
            char_end=char_end,
            style_name=item.get("style_name"),
            font_size=item.get("font_size"),
            is_bold=bool(item.get("is_bold", False)),
            is_monospace=bool(item.get("is_monospace", False)),
            indent_level=float(item.get("indent_level", 0.0)),
            table_header=item.get("table_header"),
            row_spans=row_spans,
        ))

    canonical_text = "".join(parts)
    language = detect_language_deterministic(canonical_text, hint=hint)
    return CanonicalPage(
        document_id=doc_id,
        page=page_no,
        file_type=file_type,
        canonical_text=canonical_text,
        language=language,
        lang_hint=hint,
        blocks=blocks,
        dropped_duplicate_records=dropped_dups,
    )


def preprocess_file(file_path: Path, raw_dir: Path) -> CanonicalDocument:
    """Parse and clean a single PDF/DOCX into the shared CanonicalDocument representation."""
    rel_path = file_path.relative_to(raw_dir).as_posix()
    doc_id = make_document_id(rel_path)
    raw_bytes = file_path.read_bytes()
    revision = sha256_bytes(raw_bytes)
    hint = lang_hint_from_path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        pages = _extract_pdf_pages(file_path, doc_id, hint)
        ftype = "pdf"
    elif suffix in (".docx", ".doc"):
        pages = _extract_docx_pages(file_path, doc_id, hint)
        ftype = "docx"
    else:
        raise ValueError(f"Unsupported raw file suffix: {suffix}")

    return CanonicalDocument(
        document_id=doc_id,
        filename=file_path.name,
        source_path=rel_path,
        source_revision=revision,
        file_type=ftype,
        file_bytes=len(raw_bytes),
        lang_hint=hint,
        pages=pages,
    )


def preprocess_raw_dataset(raw_dir: Path) -> list[CanonicalDocument]:
    """Deterministically preprocess all PDF and DOCX files sorted by relative path."""
    files = sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}),
        key=lambda p: p.relative_to(raw_dir).as_posix(),
    )
    return [preprocess_file(p, raw_dir) for p in files]
