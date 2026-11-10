import numpy as np
import pytest
from app.retrieval import embeddings

class Model:
    def get_sentence_embedding_dimension(self): return 3
    def encode(self, texts, **kwargs): return np.array([[3., 4., 0.]] * len(texts))

def test_empty_and_dimension(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda *_args: Model())
    assert embeddings.embed_texts([]) == []
    vectors = embeddings.embed_texts(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 3
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert embeddings.embedding_dimension() == 3
    assert embeddings.embed_query("") == []
