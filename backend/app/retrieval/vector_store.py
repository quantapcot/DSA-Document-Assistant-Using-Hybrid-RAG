"""Dense-only Qdrant storage. Local mode never opens an HTTP connection."""
from qdrant_client import QdrantClient, models
from .config import RetrievalConfig

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
        client.create_collection(collection_name=name, vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)})
    else:
        vectors = client.get_collection(name).config.params.vectors
        if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors or vectors[DENSE_VECTOR_NAME].size != dimension or vectors[DENSE_VECTOR_NAME].distance != models.Distance.COSINE:
            raise ValueError("Incompatible Qdrant collection; rebuild into a new collection")

def scope_filter(corpus_id, workspace_id):
    if not corpus_id or not workspace_id:
        raise ValueError("Both corpus_id and workspace_id are required")
    return models.Filter(must=[models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id)), models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))])

def check_model(client, config):
    points, _ = client.scroll(config.collection, limit=1, with_payload=True, with_vectors=False)
    if points and (points[0].payload or {}).get("embedding_model") != config.embedding_model:
        raise ValueError("Embedding model differs or is unknown; rebuild into a new collection")
