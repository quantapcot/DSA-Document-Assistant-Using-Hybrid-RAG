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

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger("app.api.routes.documents")

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"


# ---------------------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------------------

@router.get("/api/documents", tags=["Document Management"])
async def list_documents(
    request: Request,
    workspace_id: str = Query("default-workspace", description="Workspace cần liệt kê tài liệu"),
    corpus_id: str = Query("user-uploads", description="Corpus cần liệt kê tài liệu"),
):
    """Trả về danh sách tài liệu đã lập chỉ mục trong workspace/corpus chỉ định.

    Dữ liệu được đọc trực tiếp từ payload của Qdrant (không cần CSDL phụ).
    Mỗi document được tổng hợp từ các chunks có cùng ``document_id``.
    """
    retrieval_engine = getattr(request.app.state, "retrieval_engine", None)
    if not retrieval_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval Engine chưa sẵn sàng.",
        )

    client = getattr(retrieval_engine, "client", None)
    config = getattr(request.app.state, "retrieval_config", None)
    if client is None or config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client chưa sẵn sàng.",
        )

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
            page = payload.get("page")
            if page:
                d["pages"].add(page)
            if not d["file_type"] or d["file_type"] == "unknown":
                meta = payload.get("metadata", {})
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
async def delete_document(
    document_id: str,
    request: Request,
    workspace_id: str = Query("default-workspace"),
    corpus_id: str = Query("user-uploads"),
):
    """Xóa tất cả chunks của ``document_id`` khỏi Qdrant và file vật lý tương ứng.

    Sau khi xóa thành công, BM25 cache của scope được invalidate tự động.
    """
    retrieval_engine = getattr(request.app.state, "retrieval_engine", None)
    if not retrieval_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval Engine chưa sẵn sàng.",
        )

    client = getattr(retrieval_engine, "client", None)
    config = getattr(request.app.state, "retrieval_config", None)
    if client is None or config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client chưa sẵn sàng.",
        )

    if not client.collection_exists(config.collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy document '{document_id}' trong hệ thống.",
        )

    try:
        from qdrant_client import models as qdrant_models
        from app.retrieval.vector_store import scope_filter

        # Lọc theo scope + document_id
        scope = scope_filter(corpus_id, workspace_id)
        doc_filter = qdrant_models.Filter(
            must=[
                *scope.must,
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=document_id),
                ),
            ]
        )

        # Lấy filename trước khi xóa (để xóa file vật lý)
        sample_points, _ = client.scroll(
            config.collection,
            scroll_filter=doc_filter,
            limit=1,
            with_payload=["filename"],
            with_vectors=False,
        )
        filename = sample_points[0].payload.get("filename", "") if sample_points else ""

        if not sample_points:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy document '{document_id}' trong workspace '{workspace_id}'.",
            )

        # Xóa tất cả points của document này khỏi Qdrant
        client.delete(
            config.collection,
            points_selector=qdrant_models.FilterSelector(filter=doc_filter),
            wait=True,
        )

        # Invalidate BM25 cache của scope này
        try:
            from app.retrieval.hybrid_search import invalidate_all_bm25_caches
            invalidate_all_bm25_caches(config.collection, corpus_id, workspace_id)
        except Exception as cache_exc:
            logger.warning("BM25 cache invalidation sau khi xóa thất bại: %s", cache_exc)

        # Xóa file vật lý trên ổ đĩa (tìm theo pattern vì tên có UUID prefix)
        deleted_files: list[str] = []
        if filename and UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.glob(f"*_{filename}"):
                f.unlink(missing_ok=True)
                deleted_files.append(f.name)

        logger.info(
            "Đã xóa document '%s' (file: '%s') khỏi workspace '%s/%s'. "
            "File vật lý đã xóa: %s",
            document_id, filename, workspace_id, corpus_id, deleted_files,
        )

        return {
            "status": "deleted",
            "document_id": document_id,
            "filename": filename,
            "workspace_id": workspace_id,
            "corpus_id": corpus_id,
            "deleted_physical_files": deleted_files,
            "message": f"Tài liệu '{filename}' đã được xóa hoàn toàn khỏi hệ thống.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Lỗi khi xóa document '%s': %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa tài liệu: {exc}",
        )
