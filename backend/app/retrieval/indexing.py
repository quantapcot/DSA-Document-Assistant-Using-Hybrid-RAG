"""Index explicit scopes; BM25 is reconstructed from stored Qdrant payloads."""
import argparse
import json
import logging
import uuid
from qdrant_client import models
from .config import RetrievalConfig
from .embeddings import embed_texts, embedding_dimension
from .keyword_search import normalize_chunk
from .vector_store import get_client, ensure_collection, scope_filter, check_model

logger = logging.getLogger(__name__)


def load_chunks(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(normalize_chunk(json.loads(line)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
    return records


def build_index(chunks, *, corpus_id, workspace_id, client=None, config=None,
                rebuild=False, embedder=None, dimension=None):
    config = config or RetrievalConfig.from_env()
    scope = scope_filter(corpus_id, workspace_id)
    client = client or get_client(config=config)
    if isinstance(chunks, (str, __import__('pathlib').Path)):
        chunks = load_chunks(chunks)
    records = [normalize_chunk(c) for c in chunks]
    ids = [c["chunk_id"] for c in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk_id in corpus")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records and not rebuild:
        return {"indexed": 0, "failed": 0, "scope_count": 0}
    dimension = dimension or embedding_dimension(
        config.embedding_model, device=config.embedding_device,
    )
    ensure_collection(client, dimension, config=config)
    check_model(client, config)
    # Rebuild replaces only this explicit scope. Precompute all embeddings
    # first; a write failure during replacement can still require retrying.
    points = []
    embedder = embedder or (
        lambda texts: embed_texts(
            texts, config.embedding_model, config.batch_size,
            device=config.embedding_device,
        )
    )
    for start in range(0, len(records), config.batch_size):
        batch = records[start:start + config.batch_size]
        vectors = embedder([c["text"] for c in batch])
        if len(vectors) != len(batch) or any(
            len(v) != dimension for v in vectors
        ):
            raise ValueError("Embedding count or dimension mismatch")
        for chunk, vector in zip(batch, vectors):
            payload = {
                **chunk,
                "workspace_id": workspace_id,
                "corpus_id": corpus_id,
                "embedding_model": config.embedding_model,
            }
            identity = json.dumps(
                [workspace_id, corpus_id, chunk["chunk_id"]],
                ensure_ascii=False, separators=(",", ":"),
            )
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                vector={"dense": vector},
                payload=payload,
            ))
    if rebuild:
        client.delete(
            config.collection,
            points_selector=models.FilterSelector(filter=scope),
            wait=True,
        )
    for start in range(0, len(points), config.batch_size):
        client.upsert(
            config.collection,
            points=points[start:start + config.batch_size],
            wait=True,
        )
    count = client.count(
        config.collection, count_filter=scope, exact=True,
    ).count
    if count < len(points) or (rebuild and count != len(points)):
        raise RuntimeError("Qdrant scope count verification failed")
    logger.info(
        "Indexed %d chunks, failed 0; scope contains %d",
        len(points), count,
    )

    # Invalidate BM25 caches across all live engines.
    try:
        from .hybrid_search import invalidate_all_bm25_caches
        invalidate_all_bm25_caches(config.collection, corpus_id, workspace_id)
    except Exception:
        logger.warning(
            "BM25 cache invalidation failed; retrieval may serve "
            "stale results until TTL expires"
        )

    return {"indexed": len(points), "failed": 0, "scope_count": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = (RetrievalConfig.from_yaml(args.config)
              if args.config else RetrievalConfig.from_env())
    print(build_index(
        args.jsonl, corpus_id=args.corpus_id,
        workspace_id=args.workspace_id, config=config,
        rebuild=args.rebuild,
    ))


if __name__ == "__main__":
    main()
