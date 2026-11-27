import sys
from types import ModuleType, SimpleNamespace
import pytest

from app.retrieval import reranker
from app.retrieval.reranker import rerank


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    reranker._get_model.cache_clear()
    yield
    reranker._get_model.cache_clear()


def _install_fake_cross_encoder(monkeypatch, *, cuda_available: bool):
    init_calls = []

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = dict(kwargs)
            self.model = SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device=kwargs.get("device", "cpu"))])
            )
            init_calls.append((name, dict(kwargs)))

        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            return [float(len(text)) / 100.0 for _, text in pairs]

    fake_st = ModuleType("sentence_transformers")
    fake_st.CrossEncoder = FakeCrossEncoder
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return init_calls


def test_mock_and_missing_model():
    chunks = [{"chunk_id": "a", "text": "a", "page": 1}, {"chunk_id": "b", "text": "b", "page": 2}]
    result = rerank("q", chunks, scorer=lambda pairs: [0.1, 0.9])
    assert result[0]["chunk_id"] == "b" and result[0]["page"] == 2
    assert rerank("q", []) == []
    with pytest.raises(ValueError, match="not configured"):
        rerank("q", chunks)


def test_auto_and_cuda_share_single_model_instance(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=True)
    m_auto = reranker._get_model("BAAI/bge-reranker-v2-m3", "auto", 512)
    m_cuda_kw = reranker._get_model("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
    m_cuda_pos = reranker._get_model("BAAI/bge-reranker-v2-m3", "cuda", 512)
    assert m_auto is m_cuda_kw is m_cuda_pos
    assert len(init_calls) == 1
    assert init_calls[0] == ("BAAI/bge-reranker-v2-m3", {"device": "cuda", "max_length": 512})
    assert type(init_calls[0][1]["max_length"]) is int
    info = reranker._get_model.cache_info()
    assert info.misses == 1 and info.hits == 2


def test_auto_and_cpu_share_single_model_instance(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=False)
    m_auto = reranker._get_model("BAAI/bge-reranker-v2-m3", "auto", 0)
    m_cpu_default = reranker._get_model("BAAI/bge-reranker-v2-m3")
    m_cpu_none = reranker._get_model("BAAI/bge-reranker-v2-m3", device="cpu", max_length=None)
    assert m_auto is m_cpu_default is m_cpu_none
    assert len(init_calls) == 1
    assert init_calls[0] == ("BAAI/bge-reranker-v2-m3", {"device": "cpu"})


def test_different_devices_models_and_max_length_remain_isolated(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=True)
    m_cuda_512 = reranker._get_model("model-a", "cuda", 512)
    m_cpu_512 = reranker._get_model("model-a", "cpu", 512)
    m_cuda_256 = reranker._get_model("model-a", "cuda", 256)
    m_other = reranker._get_model("model-b", "cuda", 512)
    assert len({id(m_cuda_512), id(m_cpu_512), id(m_cuda_256), id(m_other)}) == 4
    assert len(init_calls) == 4


def test_warmup_then_repeated_rerank_reuses_cached_model_and_preserves_ranking(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=True)
    warmed = reranker._get_model("BAAI/bge-reranker-v2-m3", "auto", 512)
    chunks = [
        {"chunk_id": "c1", "text": "short", "page": 1},
        {"chunk_id": "c2", "text": "longer text for AVL tree rotation", "page": 2},
    ]
    for _ in range(5):
        out = rerank(
            "AVL tree",
            chunks,
            model_name="BAAI/bge-reranker-v2-m3",
            device="auto",
            batch_size=4,
            max_length=512,
            top_k=2,
        )
        assert [row["chunk_id"] for row in out] == ["c2", "c1"]
        assert [row["rank"] for row in out] == [1, 2]
        assert all(row["retrieval_method"] == "hybrid_rerank" for row in out)

    assert len(init_calls) == 1
    assert reranker._get_model("BAAI/bge-reranker-v2-m3", "cuda", 512) is warmed
    info = reranker._get_model.cache_info()
    assert info.misses == 1 and info.hits == 6


def test_cache_clear_and_legacy_error_arguments(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=False)
    m1 = reranker._get_model("model-a", "cpu", 512)
    reranker._get_model.cache_clear()
    m2 = reranker._get_model("model-a", "cpu", 512)
    assert m1 is not m2
    assert len(init_calls) == 2

    with pytest.raises(ValueError, match="Unknown reranker device"):
        reranker._get_model("model-a", "tpu", 512)
    with pytest.raises(TypeError):
        reranker._get_model("model-a", "cpu", "512")
    with pytest.raises(TypeError):
        reranker._get_model("model-a", "cpu", [])
    with pytest.raises(TypeError):
        reranker._get_model("model-a", "cpu", [512])


def test_legacy_default_max_length_values_and_rerank_negative(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=False)
    m_default = reranker._get_model("model-a")
    m_zero = reranker._get_model("model-a", "cpu", 0)
    m_none = reranker._get_model("model-a", device="cpu", max_length=None)
    m_neg1 = reranker._get_model("model-a", "cpu", -1)
    m_neg5 = reranker._get_model("model-a", device="cpu", max_length=-5)
    m_false = reranker._get_model("model-a", "cpu", False)
    m_zero_float = reranker._get_model("model-a", "cpu", 0.0)
    m_empty_str = reranker._get_model("model-a", "cpu", "")

    assert (
        m_default is m_zero is m_none is m_neg1 is m_neg5
        is m_false is m_zero_float is m_empty_str
    )
    assert len(init_calls) == 1
    assert init_calls[0] == ("model-a", {"device": "cpu"})
    assert "max_length" not in init_calls[0][1]

    chunks = [
        {"chunk_id": "c1", "text": "short", "page": 1},
        {"chunk_id": "c2", "text": "longer text for AVL tree rotation", "page": 2},
    ]
    out = rerank("AVL tree", chunks, model_name="model-a", device="cpu", max_length=-1, top_k=2)
    assert [row["chunk_id"] for row in out] == ["c2", "c1"]
    assert len(init_calls) == 1
    info = reranker._get_model.cache_info()
    assert info.misses == 1 and info.hits == 8


def test_forwarded_max_length_preserves_type_and_isolates_cache(monkeypatch):
    init_calls = _install_fake_cross_encoder(monkeypatch, cuda_available=False)
    m_int_one = reranker._get_model("model-a", "cpu", 1)
    m_bool_true = reranker._get_model("model-a", "cpu", True)
    assert m_int_one is not m_bool_true
    assert len(init_calls) == 2
    assert type(init_calls[0][1]["max_length"]) is int
    assert init_calls[0][1]["max_length"] == 1
    assert type(init_calls[1][1]["max_length"]) is bool
    assert init_calls[1][1]["max_length"] is True

    reranker._get_model.cache_clear()
    init_calls.clear()

    m_int_512 = reranker._get_model("model-a", "cpu", 512)
    m_float_512 = reranker._get_model("model-a", "cpu", 512.0)
    m_float_512_5 = reranker._get_model("model-a", "cpu", 512.5)
    assert len({id(m_int_512), id(m_float_512), id(m_float_512_5)}) == 3
    assert len(init_calls) == 3
    assert type(init_calls[0][1]["max_length"]) is int
    assert init_calls[0][1]["max_length"] == 512
    assert type(init_calls[1][1]["max_length"]) is float
    assert init_calls[1][1]["max_length"] == 512.0
    assert type(init_calls[2][1]["max_length"]) is float
    assert init_calls[2][1]["max_length"] == 512.5
