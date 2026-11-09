"""Central retrieval configuration."""
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str | None = None
    collection: str = "dsa_chunks_v1"
    qdrant_mode: str = "local"
    qdrant_path: str = str(PROJECT_ROOT / "data" / "runtime" / "qdrant_local")
    qdrant_url: str | None = None
    candidate_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    accent_insensitive: bool = False
    batch_size: int = 32

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("retrieval configuration must be a mapping")
        return cls(**data)

    @classmethod
    def from_env(cls):
        return cls(qdrant_mode=os.getenv("QDRANT_MODE", "local"), qdrant_path=os.getenv("QDRANT_PATH", cls.qdrant_path), qdrant_url=os.getenv("QDRANT_URL"))
