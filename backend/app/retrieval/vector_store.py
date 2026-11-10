"""Dense-only Qdrant storage. Local mode never opens an HTTP connection."""
import logging
from qdrant_client import QdrantClient, models
from .config import RetrievalConfig

logger = logging.getLogger(__name__)
DENSE_VECTOR_NAME = "dense"


def get_client(path=None, *, config=None):
    config = config or RetrievalConfig.from_env()
    if path is not None:
        return QdrantClient(":memory:") if str(path) == ":memory:" else QdrantClient(path=str(path))
    if config.qdrant_mode == "local":
        return QdrantClient(path=config.qdrant_path)
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

    Strategy
    --------
    1. Sample up to *sample_size* points across the collection and verify
       that **every** sampled point carries ``embedding_model`` matching
       ``config.embedding_model``.
    2. A point with a missing ``embedding_model`` payload is treated as
       **invalid** (fail-closed), preventing silent corruption.
    3. If different ``embedding_model`` values are found among sampled
       points, the collection is rejected as mixed-model.

    This is a statistical check — it does not scan every point — but it
    provides strong guarantees when combined with ``build_index`` always
    writing ``embedding_model`` on every upserted point.

    The check runs on every ``build_index`` and ``search_dense`` call.
    For one-time full audits, use ``check_model_exhaustive``.
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
    """Full-scan variant of :func:`check_model`.

    Iterates every point in the collection and verifies that all carry the
    same ``embedding_model``.  Use for one-time audits, not per-query checks.
    """
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
