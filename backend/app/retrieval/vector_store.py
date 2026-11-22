"""Dense-only Qdrant storage. Local mode never opens an HTTP connection."""
import logging
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

from .config import RetrievalConfig

logger = logging.getLogger(__name__)
DENSE_VECTOR_NAME = "dense"

# Cache client instances by path within the same process to avoid concurrent file lock errors
_LOCAL_CLIENTS: dict[str, Any] = {}
_LOCAL_CLIENTS_LOCK = Lock()


def _local_client(path):
    key = str(Path(path).resolve())
    with _LOCAL_CLIENTS_LOCK:
        if key not in _LOCAL_CLIENTS:
            _LOCAL_CLIENTS[key] = QdrantClient(path=key)
        return _LOCAL_CLIENTS[key]


def get_client(path=None, *, config=None):
    if QdrantClient is None:
        raise ImportError(
            "Thư viện 'qdrant-client' chưa được cài đặt. Vui lòng chạy: pip install qdrant-client"
        )
    config = config or RetrievalConfig.from_env()
    if path is not None:
        if str(path) == ":memory:":
            return QdrantClient(":memory:")
        return _local_client(path)

    if config.qdrant_mode == "local":
        return _local_client(config.qdrant_path)

    if config.qdrant_mode == "memory":
        return QdrantClient(":memory:")
    if config.qdrant_mode == "server" and config.qdrant_url:
        return QdrantClient(url=config.qdrant_url)
    raise ValueError("Choose qdrant_mode local, memory, or server with qdrant_url")


def ensure_collection(client, dimension, *, config=None, recreate=False):
    config = config or RetrievalConfig()
    name = config.collection
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if client.collection_exists(name) and recreate:
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE,
                ),
            },
        )
    else:
        vectors = client.get_collection(name).config.params.vectors
        if (not isinstance(vectors, dict)
                or DENSE_VECTOR_NAME not in vectors
                or vectors[DENSE_VECTOR_NAME].size != dimension
                or vectors[DENSE_VECTOR_NAME].distance != models.Distance.COSINE):
            raise ValueError(
                "Incompatible Qdrant collection; rebuild into a new collection"
            )


def scope_filter(corpus_id, workspace_id):
    if not corpus_id or not workspace_id:
        raise ValueError("Both corpus_id and workspace_id are required")
    return models.Filter(must=[
        models.FieldCondition(
            key="corpus_id", match=models.MatchValue(value=corpus_id),
        ),
        models.FieldCondition(
            key="workspace_id", match=models.MatchValue(value=workspace_id),
        ),
    ])


def check_model(client, config, *, sample_size: int = 10):
    """Validate that vectors in *config.collection* were created by the
    configured embedding model.
    """
    name = config.collection
    if not client.collection_exists(name):
        return  # Nothing to validate yet.

    points, _ = client.scroll(
        name, limit=sample_size, with_payload=True, with_vectors=False,
    )
    if not points:
        return  # Empty collection is valid.

    expected = config.embedding_model
    models_found: set[str] = set()

    for point in points:
        payload = point.payload or {}
        stored_model = payload.get("embedding_model")
        if stored_model is None:
            raise ValueError(
                f"Point {point.id} in collection '{name}' has no embedding_model "
                f"metadata. Cannot verify model identity; rebuild with the current "
                f"pipeline to ensure all points carry model metadata."
            )
        models_found.add(stored_model)

    if len(models_found) > 1:
        raise ValueError(
            f"Mixed embedding models detected in collection '{name}': "
            f"{models_found}. All vectors must come from the same model. "
            f"Rebuild the collection from scratch."
        )

    actual = models_found.pop()
    if actual != expected:
        raise ValueError(
            f"Embedding model mismatch in collection '{name}': "
            f"expected '{expected}', found '{actual}'. "
            f"Rebuild into a new collection or update the configuration."
        )


def check_model_exhaustive(client, config, *, batch_size: int = 256):
    """Full-scan variant of :func:`check_model`."""
    name = config.collection
    if not client.collection_exists(name):
        return

    expected = config.embedding_model
    offset = None
    checked = 0

    while True:
        points, offset = client.scroll(
            name, limit=batch_size, offset=offset,
            with_payload=["embedding_model"], with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            stored = payload.get("embedding_model")
            if stored is None:
                raise ValueError(
                    f"Point {point.id} has no embedding_model metadata"
                )
            if stored != expected:
                raise ValueError(
                    f"Point {point.id} uses model '{stored}', "
                    f"expected '{expected}'"
                )
            checked += 1
        if offset is None:
            break

    logger.info("Exhaustive model check passed: %d points verified", checked)
