"""Unit & integration tests cho OCR trên PDF scan, PDF ảnh nhúng, DOCX ảnh nhúng và file ảnh."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pymupdf
from docx import Document
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont

from app.document_processing import document_parser, ocr_engine
from app.document_processing.document_parser import parse_document


def _make_text_image_bytes(text: str, size: tuple[int, int] = (700, 180)) -> bytes:
    """Tạo ảnh PNG chứa văn bản rõ nét để test OCR thật hoặc mock."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 60), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_scanned_pdf(path: Path, text: str) -> Path:
    """Tạo PDF scan (chỉ gồm ảnh bitmap, không có text layer)."""
    img_bytes = _make_text_image_bytes(text, size=(800, 220))
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=250)
        page.insert_image(pymupdf.Rect(20, 20, 580, 220), stream=img_bytes)
        doc.save(str(path))
    return path


def _make_hybrid_pdf(path: Path, selectable_text: str, image_text: str) -> Path:
    """Tạo PDF vừa có text chọn được vừa có ảnh nhúng chứa text."""
    img_bytes = _make_text_image_bytes(image_text, size=(600, 180))
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=500)
        page.insert_textbox(pymupdf.Rect(30, 30, 560, 150), selectable_text)
        page.insert_image(pymupdf.Rect(30, 180, 560, 380), stream=img_bytes)
        doc.save(str(path))
    return path


def test_parse_scanned_pdf_triggers_ocr_mocked(tmp_path, monkeypatch):
    pdf_path = _make_scanned_pdf(tmp_path / "scan.pdf", "Binary Search Tree AVL")
    calls = []

    def fake_ocr(image_bytes: bytes, **kwargs) -> str:
        calls.append(len(image_bytes))
        return "Binary Search Tree AVL O(log n)"

    monkeypatch.setattr(document_parser, "extract_text_from_image", fake_ocr)

    pages = parse_document(pdf_path, use_ocr=True)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].file_type == "pdf"
    assert "Binary Search Tree AVL" in pages[0].raw_text
    assert len(calls) == 1

    # Khi tắt OCR thì trang scan không có text
    assert parse_document(pdf_path, use_ocr=False) == []


def test_parse_hybrid_pdf_extracts_both_text_and_embedded_image(tmp_path, monkeypatch):
    pdf_path = _make_hybrid_pdf(
        tmp_path / "hybrid.pdf",
        "Merge Sort divides the array into two halves and sorts each half recursively.",
        "Time Complexity O(n log n)",
    )

    monkeypatch.setattr(
        document_parser,
        "extract_text_from_image",
        lambda _bytes, **kwargs: "Time Complexity O(n log n)",
    )

    pages = parse_document(pdf_path, use_ocr=True, extract_embedded_images=True)
    assert len(pages) == 1
    assert "Merge Sort divides the array" in pages[0].raw_text
    assert "[OCR Image]" in pages[0].raw_text
    assert "Time Complexity O(n log n)" in pages[0].raw_text

    # Tắt extract_embedded_images vẫn lấy được selectable text
    pages_no_img = parse_document(pdf_path, use_ocr=True, extract_embedded_images=False)
    assert len(pages_no_img) == 1
    assert "[OCR Image]" not in pages_no_img[0].raw_text


def test_parse_pdf_skips_tiny_and_repeated_logo_images(tmp_path, monkeypatch):
    logo_bytes = _make_text_image_bytes("UNI LOGO", size=(200, 100))
    tiny_bytes = _make_text_image_bytes("x", size=(40, 30))

    pdf_path = tmp_path / "repeated_logo.pdf"
    with pymupdf.open() as doc:
        for i in range(6):
            page = doc.new_page(width=600, height=500)
            page.insert_textbox(
                pymupdf.Rect(30, 30, 560, 150),
                f"Page {i + 1}: Breadth First Search explores neighbors level by level using a queue.",
            )
            page.insert_image(pymupdf.Rect(30, 200, 230, 300), stream=logo_bytes)
            page.insert_image(pymupdf.Rect(30, 320, 70, 350), stream=tiny_bytes)
        doc.save(str(pdf_path))

    ocr_calls = 0

    def counting_ocr(_bytes: bytes, **kwargs) -> str:
        nonlocal ocr_calls
        ocr_calls += 1
        return "SHOULD NOT BE CALLED FOR REPEATED LOGO OR TINY ICON"

    monkeypatch.setattr(document_parser, "extract_text_from_image", counting_ocr)
    pages = parse_document(pdf_path, use_ocr=True, extract_embedded_images=True)
    assert len(pages) == 6
    assert ocr_calls == 0


def test_parse_docx_with_embedded_image_and_table(tmp_path, monkeypatch):
    docx_path = tmp_path / "notes_with_image.docx"
    img_bytes = _make_text_image_bytes("QuickSort Pivot Partition", size=(600, 160))

    doc = Document()
    doc.add_heading("Sorting Algorithms", level=1)
    doc.add_paragraph("QuickSort picks a pivot and partitions the array.")
    doc.add_picture(io.BytesIO(img_bytes), width=Inches(3))
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Algorithm"
    table.cell(0, 1).text = "Average"
    table.cell(1, 0).text = "QuickSort"
    table.cell(1, 1).text = "O(n log n)"
    doc.save(str(docx_path))

    monkeypatch.setattr(
        document_parser,
        "extract_text_from_image",
        lambda _bytes, **kwargs: "QuickSort Pivot Partition",
    )

    pages = parse_document(docx_path, use_ocr=True, extract_embedded_images=True)
    assert len(pages) == 1
    assert "QuickSort picks a pivot" in pages[0].raw_text
    assert "[OCR Image]" in pages[0].raw_text
    assert "QuickSort Pivot Partition" in pages[0].raw_text
    assert "Algorithm | Average" in pages[0].raw_text


def test_parse_standalone_image(tmp_path, monkeypatch):
    img_path = tmp_path / "slide.png"
    img_path.write_bytes(_make_text_image_bytes("Dijkstra Shortest Path", size=(600, 160)))

    monkeypatch.setattr(
        document_parser,
        "extract_text_from_image",
        lambda _bytes, **kwargs: "Dijkstra Shortest Path",
    )

    pages = parse_document(img_path, use_ocr=True)
    assert len(pages) == 1
    assert pages[0].file_type == "image"
    assert "Dijkstra Shortest Path" in pages[0].raw_text


def test_real_ocr_on_scanned_pdf_and_png(tmp_path):
    """Kiểm tra OCR thực tế bằng EasyOCR trên file PNG và PDF scan tạo mới."""
    img_bytes = _make_text_image_bytes("AVL Tree Rotation", size=(700, 180))
    png_path = tmp_path / "avl_scan.png"
    png_path.write_bytes(img_bytes)

    pdf_path = _make_scanned_pdf(tmp_path / "avl_scan.pdf", "AVL Tree Rotation")

    png_pages = parse_document(png_path, use_ocr=True)
    assert len(png_pages) == 1
    assert "avl" in png_pages[0].raw_text.lower()
    assert "tree" in png_pages[0].raw_text.lower()

    pdf_pages = parse_document(pdf_path, use_ocr=True)
    assert len(pdf_pages) == 1
    assert "avl" in pdf_pages[0].raw_text.lower()


def test_sort_and_join_boxes_preserves_reading_order():
    detections = [
        ([[120, 50], [220, 50], [220, 80], [120, 80]], "world", 0.95),
        ([[10, 52], [100, 52], [100, 82], [10, 82]], "Hello", 0.96),
        ([[10, 10], [200, 10], [200, 35], [10, 35]], "Title Line", 0.92),
        ([[10, 100], [100, 100], [100, 120], [10, 120]], "noise", 0.10),
    ]
    text = ocr_engine._sort_and_join_boxes(detections, min_confidence=0.25)
    assert text == "Title Line\nHello world"


def test_ocr_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(ocr_engine.settings, "OCR_ENABLED", False)
    with patch.object(ocr_engine, "_ocr_with_easyocr") as mocked:
        assert ocr_engine.extract_text_from_image(_make_text_image_bytes("test")) == ""
        mocked.assert_not_called()


def test_upload_api_accepts_scanned_pdf_and_image(tmp_path, monkeypatch):
    """Kiểm tra end-to-end /api/upload với PDF scan và ảnh PNG (mock OCR & embedder cho tốc độ)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from app.api.routes import upload
    from app.retrieval import indexing
    from app.retrieval.config import RetrievalConfig
    from app.retrieval.hybrid_search import RetrievalEngine

    monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        indexing,
        "embed_texts",
        lambda texts, *args, **kwargs: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        document_parser,
        "extract_text_from_image",
        lambda _bytes, **kwargs: (
            "Binary Search Tree supports insertion, deletion, and lookup in O(log n) "
            "average time complexity when the tree remains balanced across recursive steps."
        ),
    )

    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory", batch_size=2)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda _: [1.0, 0.0])
    app = FastAPI()
    app.state.retrieval_engine = engine
    app.include_router(upload.router)

    scanned_pdf_bytes = _make_scanned_pdf(tmp_path / "scanned.pdf", "Binary Search Tree").read_bytes()
    png_bytes = _make_text_image_bytes("Binary Search Tree")

    with TestClient(app) as api:
        res_pdf = api.post(
            "/api/upload",
            files={"file": ("scanned.pdf", scanned_pdf_bytes)},
            data={"workspace_id": "ocr-ws", "corpus_id": "user-uploads"},
        )
        assert res_pdf.status_code == 200, res_pdf.text
        assert res_pdf.json()["indexed_chunks"] >= 1

        res_png = api.post(
            "/api/upload",
            files={"file": ("diagram.png", png_bytes)},
            data={"workspace_id": "ocr-ws", "corpus_id": "user-uploads"},
        )
        assert res_png.status_code == 200, res_png.text
        assert res_png.json()["indexed_chunks"] >= 1

        hits = engine.retrieve("balanced", "bm25", top_k=10, workspace_id="ocr-ws", corpus_id="user-uploads")
        filenames = {hit["filename"] for hit in hits}
        assert filenames == {"scanned.pdf", "diagram.png"}

    client.close()

