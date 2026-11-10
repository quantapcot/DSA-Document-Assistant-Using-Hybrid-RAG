import pytest
from app.retrieval.reranker import rerank

def test_mock_and_missing_model():
    chunks = [{"chunk_id":"a", "text":"a", "page":1}, {"chunk_id":"b", "text":"b", "page":2}]
    result = rerank("q", chunks, scorer=lambda pairs: [0.1, 0.9])
    assert result[0]["chunk_id"] == "b" and result[0]["page"] == 2
    assert rerank("q", []) == []
    with pytest.raises(ValueError, match="not configured"): rerank("q", chunks)
