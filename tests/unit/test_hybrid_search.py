import pytest
from app.retrieval.hybrid_search import rrf_fuse

def test_rrf_and_dedup():
    a = {"chunk_id": "a", "text": "A", "filename": "a.pdf"}
    b = {"chunk_id": "b", "text": "B", "filename": "b.pdf"}
    c = {"chunk_id": "c", "text": "C", "filename": "c.pdf"}
    r = rrf_fuse([[a,b], [b,c]], k=60)
    assert [x["chunk_id"] for x in r] == ["b", "a", "c"]
    assert r[0]["score"] == pytest.approx(1/62 + 1/61)
    assert r[0]["filename"] == "b.pdf"
