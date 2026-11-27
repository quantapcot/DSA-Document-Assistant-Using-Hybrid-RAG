"""Unit & isolation tests for Stage 02 shared preprocessing, Profile A, Profile B, provenance, and indexing."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from docx import Document
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from evaluation.chunking.profiles import PROFILE_A_ID, PROFILE_B_ID
from evaluation.chunking.recursive_profile import chunk_document_recursive
from evaluation.chunking.structure_adaptive import chunk_document_adaptive, total_emb_tokens
from evaluation.corpus.benchmark_builder import (
    EVAL_COLLECTION,
    EVAL_WORKSPACE_ID,
    build_dual_corpora,
    compute_profile_diagnostics,
    index_benchmark_corpora,
)
from evaluation.corpus.provenance import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    interval_union_length,
    make_document_id,
    validate_chunk_provenance,
)
from evaluation.corpus.shared_preprocessor import (
    clean_canonical_block,
    detect_language_deterministic,
    preprocess_file,
    preprocess_raw_dataset,
)


class _WordWhitespaceTokenizer:
    """Fast deterministic mock tokenizer (2 special tokens + whitespace/punctuation sub-tokens) for unit tests."""

    def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
        import re

        tokens = re.findall(r"\S+", text or "")
        # Simulate XLM-R expansion slightly so punctuation/long words cost extra tokens cleanly
        ids = []
        for idx, t in enumerate(tokens):
            if len(t) > 12:
                parts = [t[i : i + 6] for i in range(0, len(t), 6)]
                ids.extend(range(10 + idx, 10 + idx + len(parts)))
            else:
                ids.append(10 + idx)
        if add_special_tokens:
            ids = [0, *ids, 2]
        return {"input_ids": ids}


def _make_doc(blocks: list[tuple[str, str]], *, page: int = 1, lang: str = "vi", doc_id: str = "doc_unit_1") -> CanonicalDocument:
    canon_parts = []
    c_blocks = []
    cursor = 0
    for idx, (raw_kind, txt) in enumerate(blocks):
        if canon_parts:
            canon_parts.append("\n\n")
            cursor += 2
        s = cursor
        e = s + len(txt)
        row_spans = []
        if raw_kind == "table":
            rp = s
            for ln in txt.splitlines(keepends=True):
                lc = ln.rstrip("\r\n")
                if lc:
                    row_spans.append((rp, rp + len(lc)))
                rp += len(ln)
        c_blocks.append(CanonicalBlock(
            block_id=f"{doc_id}:p{page}:b{idx}",
            raw_kind=raw_kind,
            text=txt,
            char_start=s,
            char_end=e,
            is_bold=(raw_kind == "heading"),
            is_monospace=(raw_kind == "code"),
            row_spans=row_spans,
            table_header="\n".join(txt.splitlines()[:2]) if raw_kind == "table" and len(txt.splitlines()) >= 2 else None,
        ))
        canon_parts.append(txt)
        cursor = e
    return CanonicalDocument(
        document_id=doc_id,
        filename="sample.pdf",
        source_path="pdf_Files/vie_pdf/sample.pdf",
        source_revision="a" * 64,
        file_type="pdf",
        file_bytes=1024,
        lang_hint=lang,
        pages=[CanonicalPage(doc_id, page, "pdf", "".join(canon_parts), lang, lang, c_blocks)],
    )


def test_deterministic_language_detection_and_code_indent_preservation():
    assert detect_language_deterministic("Cây AVL là cây tìm kiếm nhị phân tự cân bằng.", "vi") == "vi"
    assert detect_language_deterministic("Binary search runs in logarithmic time on a sorted array.", "en") == "en"
    assert detect_language_deterministic("def f(x):\n    return x + 1", "en") == "en"
    assert detect_language_deterministic("O(n log n)", "vi") == "vi"

    indented = "def left_rotate(z):\n    y = z.right\n    return y"
    cleaned = clean_canonical_block(indented, preserve_indent=True)
    assert "\n    y = z.right\n" in cleaned


def test_shared_preprocessor_docx_tables_and_zero_doc_id_collisions(tmp_path):
    raw = tmp_path / "dataset_raw"
    (raw / "vie_docx").mkdir(parents=True)
    (raw / "vie_pdf").mkdir(parents=True)

    # Same stem .docx and .pdf must yield distinct document_id
    docx = Document()
    docx.add_heading("Cấu trúc bảng", level=1)
    docx.add_paragraph("Đoạn mở đầu giới thiệu về bảng độ phức tạp thuật toán sắp xếp.")
    tbl = docx.add_table(rows=3, cols=2)
    tbl.cell(0, 0).text = "Algorithm"
    tbl.cell(0, 1).text = "Worst"
    tbl.cell(1, 0).text = "MergeSort"
    tbl.cell(1, 1).text = "O(n log n)"
    tbl.cell(2, 0).text = "QuickSort"
    tbl.cell(2, 1).text = "O(n^2)"
    docx_path = raw / "vie_docx" / "001_sample.docx"
    docx.save(docx_path)

    pdf_path = raw / "vie_pdf" / "001_sample.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 500, 200), "Cấu trúc dữ liệu cây nhị phân tìm kiếm và độ phức tạp.")
        doc.save(str(pdf_path))

    docs = preprocess_raw_dataset(raw)
    assert len(docs) == 2
    assert len({d.document_id for d in docs}) == 2
    assert make_document_id("vie_docx/001_sample.docx") != make_document_id("vie_pdf/001_sample.pdf")

    docx_canonical = docs[0]
    assert "| Algorithm | Worst |" in docx_canonical.pages[0].canonical_text
    assert "| MergeSort | O(n log n) |" in docx_canonical.pages[0].canonical_text


def test_recursive_profile_preserves_repeated_text_spans_and_interval_union():
    repeated = (
        "Định lý: Một cây nhị phân đầy đủ có chiều cao h sẽ có tối đa 2^(h+1) - 1 nút trong toàn bộ cây "
        "và số lá ở tầng cuối cùng tăng theo cấp số nhân. " * 2
    ).strip()
    spacer = (" ".join(f"tu{i}" for i in range(90))).strip()
    doc = _make_doc([("paragraph", repeated), ("paragraph", spacer), ("paragraph", repeated)], page=1)
    tok = _WordWhitespaceTokenizer()
    chunks, _ = chunk_document_recursive(doc, tok)

    # First occurrence vs second identical occurrence must NOT map to same offset
    spans_list = [c["metadata"]["source_spans"] for c in chunks]
    starts = [sp[0]["char_start"] for sp in spans_list if sp]
    assert len(starts) == len(chunks) and len(set(starts)) == len(starts)
    assert starts[-1] > starts[0]
    # Overlap must not push interval-union coverage over canonical length
    all_ivs = [(sp["char_start"], sp["char_end"]) for c in chunks for sp in c["metadata"]["source_spans"]]
    assert interval_union_length(all_ivs) <= len(doc.pages[0].canonical_text)


def test_adaptive_definition_algorithm_code_table_and_hard_budget():
    tok = _WordWhitespaceTokenizer()
    table_text = "\n".join([
        "| Thuật toán | Tốt nhất | Trung bình | Xấu nhất | Bộ nhớ |",
        "| --- | --- | --- | --- | --- |",
        *[f"| Algo_{i} | O(n log n) | O(n log n) | O(n^2) | O(1) |" for i in range(1, 25)],
    ])
    long_code = "def helper(a):\n    return a + 1\n\ndef big_func(arr):\n" + "\n".join(
        f"    x_{i} = arr[{i}] + {i} * 2 + 3" for i in range(45)
    )
    steps = "\n".join(f"Bước {i}: So sánh phần tử khóa x với nút hiện tại và rẽ nhánh sang cây con." for i in range(1, 18))
    oversized_sentence = " ".join(f"word{i}" for i in range(180))  # No sentence boundary
    mixed_block = "def search(a, k):\n    return k in a\nĐộ phức tạp thời gian của hàm tìm kiếm tuyến tính là O(n)."

    doc = _make_doc([
        ("heading", "1.1 Khái niệm cơ bản"),
        ("paragraph", "Định nghĩa: Hàng đợi là cấu trúc dữ liệu FIFO cơ bản."),
        ("heading", "1.2 Thuật toán"),
        ("paragraph", steps),
        ("paragraph", long_code),
        ("table", table_text),
        ("paragraph", oversized_sentence),
        ("paragraph", mixed_block),
    ])

    chunks, _ = chunk_document_adaptive(doc, tok)
    assert chunks
    # Every chunk strictly obeys <= 128 tokens including special tokens
    for ch in chunks:
        assert total_emb_tokens(tok, ch["text"]) <= 128
        assert ch["metadata"]["emb_token_count"] <= 128
        errs = validate_chunk_provenance(ch, doc.pages[0])
        assert not errs, errs

    # Table chunks retain pipe header row
    tbl_chunks = [c for c in chunks if c["metadata"]["content_type"] == "table"]
    assert len(tbl_chunks) >= 2
    assert all("| Thuật toán |" in c["text"] for c in tbl_chunks)

    # Oversized single sentence triggered forced token-span split with guaranteed progress
    assert any(c["metadata"]["fallback"]["is_forced_split"] for c in chunks)

    # Code indentation preserved
    code_chunks = [c for c in chunks if c["metadata"]["content_type"] == "code_block"]
    assert any("    x_0 =" in c["text"] for c in code_chunks)

    # Short complete definition preserved (not dropped)
    assert any("Hàng đợi là cấu trúc dữ liệu FIFO" in c["text"] for c in chunks)


def test_oversized_table_row_and_special_token_boundary():
    tok = _WordWhitespaceTokenizer()
    long_row_table = "\n".join([
        "| Structure | Detailed_Properties_And_Complex_Notes |",
        "| --- | --- |",
        "| AVLTree | " + " ".join(f"prop{i}" for i in range(170)) + " |",
    ])
    boundary_text = " ".join(f"w{i}" for i in range(127))  # 127 content + 2 special = 129 > 128!
    doc = _make_doc([("table", long_row_table), ("paragraph", boundary_text), ("code", "->"), ("paragraph", "≠")])
    chunks, dropped = chunk_document_adaptive(doc, tok)
    assert all(total_emb_tokens(tok, c["text"]) <= 128 for c in chunks)
    assert any(c["metadata"]["fallback"]["reason"] == "oversized_table_row_linearized" for c in chunks)
    assert not any(c["text"] in ("->", "≠") for c in chunks)
    assert any(d["reason"] == "no_bm25_indexable_terms" for d in dropped)


def test_end_to_end_determinism_and_qdrant_scope_isolation(tmp_path):
    raw = tmp_path / "raw"
    (raw / "vie_pdf").mkdir(parents=True)
    with pymupdf.open() as pdf_doc:
        pdf_doc.new_page().insert_textbox(
            pymupdf.Rect(40, 40, 520, 400),
            "4.2 Cây AVL\n\nĐịnh nghĩa: Cây AVL là cây tìm kiếm nhị phân tự cân bằng với hệ số cân bằng thuộc {-1, 0, 1}.\n\n"
            + ("Thao tác quay trái và quay phải giúp khôi phục chiều cao O(log n) sau khi chèn hoặc xóa nút. " * 6),
        )
        pdf_doc.save(str(raw / "vie_pdf" / "001_avl.pdf"))

    tok = _WordWhitespaceTokenizer()
    art = tmp_path / "artifacts"
    summary = build_dual_corpora(raw_dir=raw, artifacts_dir=art, verify_determinism=True, emb_tokenizer=tok)
    assert summary["determinism_verified_two_runs"] is True
    assert summary["profiles"][PROFILE_A_ID]["diagnostics"]["duplicate_chunk_ids"] == 0
    assert summary["profiles"][PROFILE_B_ID]["diagnostics"]["embedding_truncation_count"] == 0

    # Index into isolated in-memory Qdrant and test retrieval isolation across both profiles
    mem_client = QdrantClient(":memory:")
    q_dir = tmp_path / "qdrant_eval"
    fake_embedder = lambda texts: [[0.6, 0.8] for _ in texts]
    cfg = RetrievalConfig(qdrant_mode="memory", collection=EVAL_COLLECTION, batch_size=8, reranker_model=None)

    from app.retrieval import indexing
    orig_dim, orig_emb = indexing.embedding_dimension, indexing.embed_texts
    indexing.embedding_dimension = lambda *a, **kw: 2
    indexing.embed_texts = lambda texts, *a, **kw: [[0.6, 0.8] for _ in texts]
    try:
        # Index both scopes into isolated memory client via build_index
        res_a = indexing.build_index(
            art / PROFILE_A_ID / "chunks.jsonl",
            corpus_id="eval-recursive-512-v1",
            workspace_id=EVAL_WORKSPACE_ID,
            client=mem_client,
            config=cfg,
            rebuild=True,
        )
        res_b = indexing.build_index(
            art / PROFILE_B_ID / "chunks.jsonl",
            corpus_id="eval-structure-adaptive-v1",
            workspace_id=EVAL_WORKSPACE_ID,
            client=mem_client,
            config=cfg,
            rebuild=True,
        )
        assert res_a["indexed"] == summary["profiles"][PROFILE_A_ID]["diagnostics"]["chunk_count"]
        assert res_b["indexed"] == summary["profiles"][PROFILE_B_ID]["diagnostics"]["chunk_count"]

        engine = RetrievalEngine(
            client=mem_client,
            config=cfg,
            query_embedder=lambda _: [0.6, 0.8],
            rerank_scorer=lambda pairs: [float(len(t)) for _, t in pairs],
        )
        for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
            hits_a = engine.retrieve("Cây AVL", method, top_k=5, corpus_id="eval-recursive-512-v1", workspace_id=EVAL_WORKSPACE_ID)
            hits_b = engine.retrieve("Cây AVL", method, top_k=5, corpus_id="eval-structure-adaptive-v1", workspace_id=EVAL_WORKSPACE_ID)
            assert hits_a and all(h["metadata"]["profile_id"] == PROFILE_A_ID for h in hits_a)
            assert hits_b and all(h["metadata"]["profile_id"] == PROFILE_B_ID for h in hits_b)
            assert not ({h["chunk_id"] for h in hits_a} & {h["chunk_id"] for h in hits_b})
    finally:
        indexing.embedding_dimension = orig_dim
        indexing.embed_texts = orig_emb
        mem_client.close()


def test_recursive_nested_overlap_resplit_exact_provenance_and_prod_parity():
    """Regression for AC-03 #1: overlapped recursive re-splits must never fall back to coarse parent spans."""
    from app.document_processing.chunker import chunk_text
    from app.document_processing.text_cleaner import is_valid_chunk_text

    # Long text with paragraphs > 430 chars so adding 100-char overlap triggers recursive re-splitting
    para1 = "Đoạn một giải thích chi tiết cấu trúc cây nhị phân tìm kiếm và phép chèn nút mới vào đúng vị trí. " * 5
    para2 = "Đoạn hai phân tích phép xóa nút có hai con bằng cách tìm phần tử nhỏ nhất ở cây con bên phải. " * 5
    para3 = para1  # Repeated identical text later on the same page
    doc = _make_doc([("paragraph", para1.strip()), ("paragraph", para2.strip()), ("paragraph", para3.strip())])
    tok = _WordWhitespaceTokenizer()

    chunks, dropped = chunk_document_recursive(doc, tok)
    assert len(chunks) >= 3
    page = doc.pages[0]

    # Exact parity with production chunk_text + is_valid_chunk_text(min_words=15)
    prod = [c for c in chunk_text(page.canonical_text) if is_valid_chunk_text(c["text"], min_words=15)]
    assert [c["text"] for c in chunks] == [p["text"] for p in prod]
    assert [c["section"] for c in chunks] == [p["section"] for p in prod]

    # Zero provenance errors & every span literally corresponds to chunk text
    for c in chunks:
        errs = validate_chunk_provenance(c, page)
        assert not errs, errs

    diag = compute_profile_diagnostics(PROFILE_A_ID, [doc], chunks, dropped)
    assert diag["missing_provenance_count"] == 0
    assert diag["coverage_partition_exact_match"] is True
    assert diag["uncovered_other_meaningful_chars"] == 0


def test_adaptive_merged_headings_source_backed_context_and_short_formulas():
    """Regression for AC-03 #2 & #3: non-contiguous heading spans, source-backed prefixes, short formulas, symbol attach."""
    tok = _WordWhitespaceTokenizer()
    long_heading_1 = "1.1 Giới thiệu về cấu trúc cây cân bằng và thuật toán tìm kiếm nhị phân nâng cao"
    long_heading_2 = "1.2 Phân tích chi tiết phép quay đơn và phép quay kép trong cây AVL"
    long_para = (
        "Cây AVL duy trì chiều cao lôgarit bằng cách kiểm tra hệ số cân bằng tại mỗi nút sau khi chèn hoặc xóa. " * 9
    ).strip()
    table_text = "\n".join([
        "| Thuật toán | Thời gian trung bình | Thời gian xấu nhất |",
        "| --- | --- | --- |",
        *[f"| Sort_{i} | O(n log n) | O(n^2) |" for i in range(1, 15)],
    ])
    doc = _make_doc([
        ("heading", long_heading_1),
        ("heading", long_heading_2),
        ("heading", "2.1 Chủ đề dài tiếp theo cần tách thành heading prefix"),
        ("paragraph", long_para),
        ("heading", "2.2 Bảng so sánh"),
        ("table", table_text),
        ("paragraph", "i=0\ni=0"),
        ("paragraph", "k=i+1\nk=i"),
        ("paragraph", "m=0; t=2;"),
        ("paragraph", "Z(S)=[16,0,0]."),
        ("paragraph", "𝑗=1÷𝑚{𝑓𝑝,𝑗}"),
        ("code", "void f(int x) {\n    return x;\n"),
        ("code", "}"),
        ("paragraph", "42"),
    ])

    chunks, dropped = chunk_document_adaptive(doc, tok)
    page = doc.pages[0]

    for ch in chunks:
        assert total_emb_tokens(tok, ch["text"]) <= 128
        errs = validate_chunk_provenance(ch, page)
        assert not errs, errs

    # Source-backed repeated context must carry coordinates without inflating original source_spans
    hp_syns = [
        syn for c in chunks for syn in c["metadata"]["synthetic_segments"]
        if syn["kind"] == "heading_prefix"
    ]
    tbl_syns = [
        syn for c in chunks for syn in c["metadata"]["synthetic_segments"]
        if syn["kind"] == "table_header"
    ]
    assert hp_syns and all(s["source_backed"] and s["char_start"] is not None for s in hp_syns)
    assert tbl_syns and all(s["source_backed"] and s["char_start"] is not None for s in tbl_syns)

    # Short compact formulas/assignments must be preserved (not dropped by noise_lt_3_tokens)
    all_text = "\n".join(c["text"] for c in chunks)
    for formula_sample in ("i=0", "k=i+1", "m=0; t=2;", "Z(S)=[16,0,0].", "𝑗=1÷𝑚{𝑓𝑝,𝑗}"):
        assert formula_sample in all_text, f"Short formula dropped: {formula_sample}"

    # Pure closing brace following code_block should be attached to code_block
    assert any("void f(int x)" in c["text"] and "}" in c["text"] for c in chunks)
    # Pure page number '42' should still be dropped as noise_lt_3_tokens
    assert any(d["reason"] == "noise_lt_3_tokens" for d in dropped)

    diag = compute_profile_diagnostics(PROFILE_B_ID, [doc], chunks, dropped)
    assert diag["missing_provenance_count"] == 0
    assert diag["coverage_partition_exact_match"] is True
    assert diag["uncovered_other_meaningful_chars"] == 0
    assert diag["repeated_context_only_chars_union"] > 0
