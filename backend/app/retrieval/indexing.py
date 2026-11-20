"""Index explicit scopes; BM25 is reconstructed from stored Qdrant payloads."""
import argparse
import json
import logging
import math
from pathlib import Path
from threading import RLock
import uuid

try:
    from qdrant_client import models
except ImportError:
    models = None

from .config import RetrievalConfig
from .embeddings import embed_texts, embedding_dimension
from .keyword_search import normalize_chunk
from .vector_store import get_client, ensure_collection, scope_filter, check_model

logger = logging.getLogger(__name__)
_mutation_locks = tuple(RLock() for _ in range(64))


def scope_mutation_lock(collection, corpus_id, workspace_id):
    return _mutation_locks[hash((collection, corpus_id, workspace_id)) % len(_mutation_locks)]


def document_filter(document_id, corpus_id, workspace_id):
    if not document_id:
        raise ValueError("document_id is required")
    scope = scope_filter(corpus_id, workspace_id)
    return models.Filter(must=[*scope.must, models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id),
    )])


def _invalidate(config, corpus_id, workspace_id):
    from .hybrid_search import invalidate_all_bm25_caches
    invalidate_all_bm25_caches(config.collection, corpus_id, workspace_id)


def delete_document_from_index(document_id, *, corpus_id, workspace_id,
                               client=None, config=None):
    config = config or RetrievalConfig.from_env()
    selector = document_filter(document_id, corpus_id, workspace_id)
    if client is None:
        owned_client = get_client(config=config)
        try:
            return delete_document_from_index(document_id, corpus_id=corpus_id,
                                              workspace_id=workspace_id,
                                              client=owned_client, config=config)
        finally:
            if config.qdrant_mode != "local":
                owned_client.close()
    with scope_mutation_lock(config.collection, corpus_id, workspace_id):
        if not client.collection_exists(config.collection):
            return 0
        count = client.count(config.collection, count_filter=selector, exact=True).count
        if count:
            client.delete(config.collection, models.FilterSelector(filter=selector), wait=True)
            _invalidate(config, corpus_id, workspace_id)
        return count


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
                rebuild=False, embedder=None, dimension=None,
                replace_document_id=None, legacy_document_ids=()):
    if models is None:
        raise ImportError(
            "Thư viện 'qdrant-client' chưa được cài đặt. Vui lòng chạy: pip install qdrant-client"
        )
    config = config or RetrievalConfig.from_env()
    scope = scope_filter(corpus_id, workspace_id)
    if client is None:
        owned_client = get_client(config=config)
        try:
            return build_index(
                chunks, corpus_id=corpus_id, workspace_id=workspace_id,
                client=owned_client, config=config, rebuild=rebuild,
                embedder=embedder, dimension=dimension,
                replace_document_id=replace_document_id,
                legacy_document_ids=legacy_document_ids,
            )
        finally:
            if config.qdrant_mode != "local":
                owned_client.close()
    if isinstance(chunks, (str, Path)):
        chunks = load_chunks(chunks)
    records = [normalize_chunk(c) for c in chunks]
    ids = [c["chunk_id"] for c in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk_id in corpus")
    if replace_document_id is not None and (
        rebuild or not records or any(c["document_id"] != replace_document_id for c in records)
    ):
        raise ValueError("Document replacement requires a nonempty batch for one document")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records and not rebuild:
        return {"indexed": 0, "failed": 0, "scope_count": 0}
    dimension = dimension or embedding_dimension(
        config.embedding_model, device=config.embedding_device,
    )
    # Stage and validate every batch before changing stored points.
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
            len(v) != dimension or not all(map(math.isfinite, v)) or not any(v)
            for v in vectors
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
        logger.info("Staged embeddings: %d/%d chunks for %s/%s",
                    len(points), len(records), corpus_id, workspace_id)
    with scope_mutation_lock(config.collection, corpus_id, workspace_id):
        ensure_collection(client, dimension, config=config)
        check_model(client, config)
        count = _commit_index(client, config, scope, points, rebuild,
                              replace_document_id, legacy_document_ids,
                              corpus_id, workspace_id)
    return {"indexed": len(points), "failed": 0, "scope_count": count}


def _commit_index(client, config, scope, points, rebuild, document_id,
                  legacy_document_ids, corpus_id, workspace_id):
    collection = config.collection
    snapshot = []
    if document_id is not None:
        for old_id in dict.fromkeys((document_id, *legacy_document_ids)):
            selector = document_filter(old_id, corpus_id, workspace_id)
            offset = None
            while True:
                records, offset = client.scroll(collection, scroll_filter=selector,
                                                offset=offset, limit=256,
                                                with_payload=True, with_vectors=True)
                snapshot.extend(models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                                for p in records)
                if offset is None:
                    break
    new_ids = {p.id for p in points}
    old_ids = {p.id for p in snapshot}
    if rebuild:
        client.delete(
            config.collection,
            points_selector=models.FilterSelector(filter=scope),
            wait=True,
        )
    try:
        _upsert_batches(client, config, points)
        # Publish staged points before removing stale IDs to avoid a gap in coverage.
        if old_ids - new_ids:
            client.delete(collection, list(old_ids - new_ids), wait=True)
        count = client.count(collection, count_filter=scope, exact=True).count
        if count < len(points) or (rebuild and count != len(points)):
            raise RuntimeError("Qdrant scope count verification failed")
        if document_id is not None:
            selector = document_filter(document_id, corpus_id, workspace_id)
            if client.count(collection, count_filter=selector, exact=True).count != len(points):
                raise RuntimeError("Qdrant document count verification failed")
    except Exception:
        if document_id is not None:
            try:
                _upsert_batches(client, config, snapshot)
                if new_ids - old_ids:
                    client.delete(collection, list(new_ids - old_ids), wait=True)
            except Exception:
                logger.exception("Could not restore document %s after a Qdrant write failure", document_id)
            finally:
                _invalidate(config, corpus_id, workspace_id)
        raise
    _invalidate(config, corpus_id, workspace_id)
    logger.info(
        "Indexed %d chunks, failed 0; scope contains %d",
        len(points), count,
    )

    return count


def _upsert_batches(client, config, points):
    for start in range(0, len(points), config.batch_size):
        client.upsert(config.collection,
                      points=points[start:start + config.batch_size], wait=True)


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
