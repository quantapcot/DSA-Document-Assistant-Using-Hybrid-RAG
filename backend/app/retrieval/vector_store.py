"""C1: vector database access — Qdrant local (file-based, no Docker required)."""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

logger = logging.getLogger(__name__)

# Tên collection và tên các vector field trong Qdrant
COLLECTION_NAME = "dsa_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DIM = 384

# Thư mục lưu dữ liệu Qdrant local (file-based, không cần Docker)
_DEFAULT_QDRANT_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "runtime" / "qdrant_local"
)


def get_client(path: str | Path | None = None) -> QdrantClient:
    """
    Lấy Qdrant client.
    - Nếu `path` được cung cấp hoặc tồn tại QDRANT_PATH → dùng local file.
    - Dùng in-memory khi path=':memory:'.
    """
    import os
    resolved = path or os.environ.get("QDRANT_PATH") or str(_DEFAULT_QDRANT_PATH)
    if resolved == ":memory:":
        logger.info("Using Qdrant in-memory mode")
        return QdrantClient(":memory:")
    p = Path(resolved)
    p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Qdrant local path: {p}")
    return QdrantClient(path=str(p))


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Tạo collection DSA nếu chưa tồn tại.
    Cấu hình: dense vector (cosine) + sparse vector (BM25).
    """
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not recreate:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping create.")
        return
    if exists and recreate:
        logger.info(f"Dropping existing collection '{COLLECTION_NAME}'.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams()
        },
    )
    logger.info(f"Created collection '{COLLECTION_NAME}'.")


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    batch_size: int = 128,
) -> None:
    """Upsert danh sách PointStruct vào Qdrant theo batch."""
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    logger.info(f"Upserted {len(points)} points to '{COLLECTION_NAME}'.")
