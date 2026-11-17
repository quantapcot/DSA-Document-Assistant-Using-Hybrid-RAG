"""A1: coordinate document processing (B1) and indexing (C1)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.document_processing.chunker import chunk_text
from app.document_processing.document_parser import parse_document
from app.document_processing.metadata import build_chunk
from app.document_processing.text_cleaner import clean_text, is_valid_chunk_text
from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index

logger = logging.getLogger("app.orchestration.ingestion_flow")


def process_and_index_document(
    file_path: str | Path,
    *,
    workspace_id: str,
    corpus_id: str,
    config: RetrievalConfig | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """
    Chuỗi xử lý Ingestion:
    File PDF/DOCX -> B1 (parse -> clean -> chunk -> build DocumentChunk) -> C1 (build_index Qdrant).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tìm thấy: {file_path}")

    filename = path.name
    logger.info("Bắt đầu trích xuất văn bản từ tài liệu: %s", filename)
    pages = parse_document(path)
    if not pages:
        raise ValueError(f"Không thể trích xuất văn bản từ file {filename}")

    all_chunks = []
    chunk_idx = 0

    # 1. Pipeline xử lý và làm sạch văn bản B1
    for page in pages:
        cleaned = clean_text(page.raw_text)
        if not cleaned:
            continue

        raw_chunks = chunk_text(cleaned, chunk_size=512, chunk_overlap=100)
        for ch in raw_chunks:
            text = ch["text"]
            if not is_valid_chunk_text(text, min_words=15):
                continue

            doc_chunk = build_chunk(
                filename=filename,
                page=page.page,
                text=text,
                chunk_index=chunk_idx,
                section=ch.get("section"),
                extra_metadata={"file_type": page.file_type},
            )
            all_chunks.append(doc_chunk)
            chunk_idx += 1

    if not all_chunks:
        raise ValueError(f"Tài liệu '{filename}' không tạo được đoạn văn bản (chunk) hợp lệ nào sau khi làm sạch.")

    logger.info("Đã tạo %d chunks cho '%s'. Bắt đầu lập chỉ mục vào Vector DB (C1)...", len(all_chunks), filename)

    cfg = config or RetrievalConfig.from_env()

    # 2. Lập chỉ mục vào C1 (dùng chung client với RetrievalEngine để tránh khóa file Qdrant local)
    index_result = build_index(
        chunks=all_chunks,
        corpus_id=corpus_id,
        workspace_id=workspace_id,
        client=client,
        config=cfg,
        rebuild=False,
    )

    logger.info("Hoàn tất nạp '%s': %d chunks đã được lưu.", filename, index_result["indexed"])
    return {
        "filename": filename,
        "total_chunks": len(all_chunks),
        "indexed_chunks": index_result["indexed"],
        "scope_count": index_result["scope_count"],
        "corpus_id": corpus_id,
        "workspace_id": workspace_id,
    }
