"""C1: build and update keyword/vector indexes — load chunks → Qdrant."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client.models import PointStruct, SparseVector

from app.retrieval.embeddings import embed_texts
from app.retrieval.keyword_search import build_vocab, text_to_sparse_vector, save_vocab
from app.retrieval.vector_store import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_client,
    upsert_points,
)

logger = logging.getLogger(__name__)

CHUNKS_JSONL = (
    Path(__file__).resolve().parents[4]
    / "data" / "corpus" / "dataset_clean" / "chunks_all.jsonl"
)
VOCAB_PATH = (
    Path(__file__).resolve().parents[4]
    / "data" / "runtime" / "qdrant_local" / "vocab.json"
)


def load_chunks(jsonl_path: Path = CHUNKS_JSONL) -> list[dict]:
    """Load tất cả chunks từ file JSONL."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks


def build_index(
    jsonl_path: Path = CHUNKS_JSONL,
    qdrant_path: str | None = None,
    recreate: bool = False,
    batch_size: int = 128,
) -> None:
    """
    Pipeline đầy đủ: load chunks → embed → sparse → upsert vào Qdrant.

    Parameters
    ----------
    jsonl_path  : Đường dẫn file chunks_all.jsonl.
    qdrant_path : Thư mục Qdrant local (None → dùng default).
    recreate    : True → xóa collection cũ và tạo lại.
    batch_size  : Số chunk xử lý mỗi lần.
    """
    client = get_client(qdrant_path)
    ensure_collection(client, recreate=recreate)

    chunks = load_chunks(jsonl_path)
    texts = [c["text"] for c in chunks]

    # --- Build BM25 vocab ---
    logger.info("Building BM25 vocabulary...")
    vocab = build_vocab(texts)
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(str(VOCAB_PATH))

    # --- Embed + upsert theo batch ---
    total = len(chunks)
    logger.info(f"Embedding and indexing {total} chunks (batch_size={batch_size})...")

    for start in range(0, total, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_texts = [c["text"] for c in batch_chunks]

        # Dense embeddings
        dense_vecs = embed_texts(batch_texts)

        # Build Qdrant PointStructs
        points: list[PointStruct] = []
        for i, (chunk, dense) in enumerate(zip(batch_chunks, dense_vecs)):
            idx, vals = text_to_sparse_vector(chunk["text"], vocab)

            # Qdrant cần ít nhất 1 phần tử trong sparse vector
            if not idx:
                idx, vals = [0], [0.0]

            points.append(
                PointStruct(
                    id=start + i,   # integer id (0-based)
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: SparseVector(indices=idx, values=vals),
                    },
                    payload={
                        "chunk_id":    chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "filename":    chunk["filename"],
                        "page":        chunk["page"],
                        "section":     chunk["section"],
                        "text":        chunk["text"],
                        "language":    chunk["metadata"].get("language", "unknown"),
                        "char_count":  chunk["metadata"].get("char_count", 0),
                    },
                )
            )

        upsert_points(client, points, batch_size=batch_size)
        logger.info(f"  → {min(start + batch_size, total)}/{total} done")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Index complete. Points in collection: {info.points_count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_index(recreate=True)
