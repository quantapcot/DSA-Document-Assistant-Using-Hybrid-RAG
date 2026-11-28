import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from app.retrieval import embeddings


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    embeddings._get_model.cache_clear()
    yield
    embeddings._get_model.cache_clear()


class Model:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[3.0, 4.0, 0.0]] * len(texts))


def _install_fake_sentence_transformer(monkeypatch, *, cuda_available: bool):
    init_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, device="cpu"):
            self.model_name = model_name
            self.device = device
            init_calls.append((model_name, device))

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            return np.array([[3.0, 4.0, 0.0]] * len(texts))

    fake_st = ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = FakeSentenceTransformer
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return init_calls


def test_empty_and_dimension(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda *_args: Model())
    assert embeddings.embed_texts([]) == []
    vectors = embeddings.embed_texts(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 3
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert embeddings.embedding_dimension() == 3
    assert embeddings.embed_query("") == []


def test_auto_and_cuda_share_single_embedding_instance(monkeypatch):
    init_calls = _install_fake_sentence_transformer(monkeypatch, cuda_available=True)
    m_auto = embeddings._get_model(embeddings.DEFAULT_MODEL, "auto")
    m_cuda = embeddings._get_model(embeddings.DEFAULT_MODEL, device="cuda")
    assert m_auto is m_cuda
    assert len(init_calls) == 1
    assert init_calls[0] == (embeddings.DEFAULT_MODEL, "cuda")

    vectors = embeddings.embed_texts(["AVL tree", "BFS"], device="auto")
    q_vec = embeddings.embed_query("AVL tree", device="cuda")
    assert len(vectors) == 2 and len(vectors[0]) == 3 and len(q_vec) == 3
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert np.linalg.norm(q_vec) == pytest.approx(1.0)
    assert len(init_calls) == 1


def test_auto_and_cpu_share_single_embedding_instance(monkeypatch):
    init_calls = _install_fake_sentence_transformer(monkeypatch, cuda_available=False)
    m_auto = embeddings._get_model(embeddings.DEFAULT_MODEL, "auto")
    m_cpu = embeddings._get_model(embeddings.DEFAULT_MODEL)
    assert m_auto is m_cpu
    assert len(init_calls) == 1
    assert init_calls[0] == (embeddings.DEFAULT_MODEL, "cpu")


def test_embedding_device_and_model_isolation_and_cache_clear(monkeypatch):
    init_calls = _install_fake_sentence_transformer(monkeypatch, cuda_available=True)
    m_cuda = embeddings._get_model("model-a", "cuda")
    m_cpu = embeddings._get_model("model-a", "cpu")
    m_other = embeddings._get_model("model-b", "cuda")
    assert len({id(m_cuda), id(m_cpu), id(m_other)}) == 3
    assert len(init_calls) == 3

    embeddings._get_model.cache_clear()
    m_reloaded = embeddings._get_model("model-a", "cuda")
    assert m_reloaded is not m_cuda
    assert len(init_calls) == 4

    with pytest.raises(ValueError, match="Unknown device"):
        embeddings._get_model("model-a", "tpu")
