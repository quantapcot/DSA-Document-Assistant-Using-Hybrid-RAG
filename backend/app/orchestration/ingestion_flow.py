"""A1: coordinate document processing (B1) and indexing (C1)."""
from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, is_zipfile

from app.document_processing.chunker import chunk_text
from app.document_processing.document_parser import parse_document
from app.document_processing.metadata import build_chunk
from app.document_processing.text_cleaner import clean_text, is_valid_chunk_text
from app.retrieval.config import RetrievalConfig
from app.retrieval.indexing import build_index

logger = logging.getLogger("app.orchestration.ingestion_flow")


class InvalidDocumentError(ValueError):
    """The uploaded file cannot produce searchable text."""


def upload_document_id(filename: str) -> str:
    return sha256(filename.encode("utf-8")).hexdigest()


def process_and_index_document(
    file_path: str | Path,
    *,
    workspace_id: str,
    corpus_id: str,
    config: RetrievalConfig | None = None,
    client: Any | None = None,
    original_filename: str | None = None,
    upload_metadata: dict | None = None,
    legacy_document_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    Chuỗi xử lý Ingestion:
    File PDF/DOCX -> B1 (parse -> clean -> chunk -> build DocumentChunk) -> C1 (build_index Qdrant).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tìm thấy: {file_path}")

    filename = original_filename or path.name
    document_id = upload_document_id(filename)
    if path.suffix.lower() == ".docx" and not is_zipfile(path):
        raise InvalidDocumentError(f"Cannot read '{filename}' as DOCX.")
    logger.info("Bắt đầu trích xuất văn bản từ tài liệu: %s", filename)
    try:
        pages = parse_document(path)
    except (RuntimeError, ValueError, BadZipFile, KeyError) as exc:
        raise InvalidDocumentError(f"Cannot read '{filename}' as PDF or DOCX.") from exc
    if not pages:
        raise InvalidDocumentError(f"No readable text in '{filename}'.")

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
                extra_metadata={"file_type": page.file_type, **(upload_metadata or {})},
            )
            # The B1 stem slug alone can collide across distinct filenames.
            doc_chunk.document_id = document_id
            doc_chunk.chunk_id = sha256(
                f"{document_id}:{doc_chunk.chunk_id}".encode("ascii")
            ).hexdigest()
            all_chunks.append(doc_chunk)
            chunk_idx += 1

    if not all_chunks:
        raise InvalidDocumentError(f"No searchable text in '{filename}'.")

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
        replace_document_id=document_id,
        legacy_document_ids=legacy_document_ids,
    )

    logger.info("Hoàn tất nạp '%s': %d chunks đã được lưu.", filename, index_result["indexed"])
    return {
        "filename": filename,
        "document_id": document_id,
        "total_chunks": len(all_chunks),
        "indexed_chunks": index_result["indexed"],
        "scope_count": index_result["scope_count"],
        "corpus_id": corpus_id,
        "workspace_id": workspace_id,
    }
