"""A2/A1: Document Management API — liệt kê và xóa tài liệu đã nạp trong workspace.

Endpoints:
  GET  /api/documents            — Danh sách tài liệu trong workspace/corpus.
  DELETE /api/documents/{doc_id} — Xóa tất cả chunks của một document_id khỏi Qdrant
                                   và file vật lý tương ứng trên ổ đĩa.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.routes.upload import get_upload_backend, physical_upload, validate_mutable_scope
from app.retrieval.indexing import document_filter, delete_document_from_index, scope_mutation_lock

logger = logging.getLogger("app.api.routes.documents")

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"


# ---------------------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------------------

@router.get("/api/documents", tags=["Document Management"])
def list_documents(
    request: Request,
    workspace_id: str = Query("default-workspace", description="Workspace cần liệt kê tài liệu"),
    corpus_id: str = Query("user-uploads", description="Corpus cần liệt kê tài liệu"),
):
    """Trả về danh sách tài liệu đã lập chỉ mục trong workspace/corpus chỉ định.

    Dữ liệu được đọc trực tiếp từ payload của Qdrant (không cần CSDL phụ).
    Mỗi document được tổng hợp từ các chunks có cùng ``document_id``.
    """
    client, config = get_upload_backend(request)

    if not client.collection_exists(config.collection):
        return {"documents": [], "total": 0, "workspace_id": workspace_id, "corpus_id": corpus_id}

    # Scroll toàn bộ payload của scope này
    from app.retrieval.vector_store import scope_filter
    scope = scope_filter(corpus_id, workspace_id)

    docs: dict[str, dict] = defaultdict(lambda: {
        "document_id": "",
        "filename": "",
        "chunk_count": 0,
        "pages": set(),
        "file_type": "unknown",
    })

    offset = None
    while True:
        points, offset = client.scroll(
            config.collection,
            scroll_filter=scope,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("document_id", "")
            if not doc_id:
                continue
            d = docs[doc_id]
            d["document_id"] = doc_id
            d["filename"] = payload.get("filename", "")
            d["chunk_count"] += 1
            meta = payload.get("metadata") or {}
            d["size_kb"] = meta.get("size_kb")
            d["uploaded_at"] = meta.get("uploaded_at")
            page = payload.get("page")
            if page:
                d["pages"].add(page)
            if not d["file_type"] or d["file_type"] == "unknown":
                d["file_type"] = meta.get("file_type", payload.get("file_type", "unknown"))
        if offset is None:
            break

    # Serialize để trả JSON
    result = []
    for d in docs.values():
        result.append({
            "document_id": d["document_id"],
            "filename": d["filename"],
            "chunk_count": d["chunk_count"],
            "page_count": len(d["pages"]),
            "file_type": d["file_type"],
            "size_kb": d["size_kb"],
            "uploaded_at": d["uploaded_at"],
        })

    result.sort(key=lambda x: x["filename"])
    return {
        "documents": result,
        "total": len(result),
        "workspace_id": workspace_id,
        "corpus_id": corpus_id,
    }


# ---------------------------------------------------------------------------
# DELETE /api/documents/{document_id}
# ---------------------------------------------------------------------------

@router.delete("/api/documents/{document_id}", tags=["Document Management"])
def delete_document(
    document_id: str,
    request: Request,
    workspace_id: str = Query("default-workspace"),
    corpus_id: str = Query("user-uploads"),
):
    """Xóa tất cả chunks của ``document_id`` khỏi Qdrant và file vật lý tương ứng.

    Sau khi xóa thành công, BM25 cache của scope được invalidate tự động.
    """
    validate_mutable_scope(corpus_id, workspace_id)
    client, config = get_upload_backend(request)
    selector = document_filter(document_id, corpus_id, workspace_id)
    with scope_mutation_lock(config.collection, corpus_id, workspace_id):
        points = []
        if client.collection_exists(config.collection):
            points, _ = client.scroll(
                config.collection, scroll_filter=selector, limit=1,
                with_payload=True, with_vectors=False,
            )
        if not points:
            raise HTTPException(status_code=404, detail="Document not found in this scope.")
        payload = points[0].payload or {}
        path = physical_upload(UPLOAD_DIR, corpus_id, workspace_id, payload)
        deleted_chunks = delete_document_from_index(
            document_id, corpus_id=corpus_id, workspace_id=workspace_id,
            client=client, config=config,
        )
        deleted_files = []
        if path is not None:
            try:
                path.unlink()
                deleted_files.append(path.name)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.exception("Could not remove upload %s", path.name)
                raise HTTPException(status_code=500, detail="Index deleted but file cleanup failed.") from exc
    filename = payload.get("filename", "")
    return {
        "status": "deleted", "document_id": document_id, "filename": filename,
        "workspace_id": workspace_id, "corpus_id": corpus_id,
        "deleted_chunks": deleted_chunks, "deleted_physical_files": deleted_files,
        "message": f"Deleted '{filename}'.",
    }
