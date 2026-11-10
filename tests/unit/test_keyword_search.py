import pytest
from app.retrieval.keyword_search import BM25Retriever, tokenize

def chunk(i, text):
    return dict(chunk_id=str(i), document_id="doc", filename="a.pdf", page=1, text=text, metadata={"source": "test"})

def test_tokenization_and_search():
    assert "nh\u1ecb_ph\u00e2n" in tokenize("C\u00e2y nh\u1ecb ph\u00e2n")
    docs = [chunk(1, "C\u00e2y nh\u1ecb ph\u00e2n binary tree"), chunk(2, "BFS traversal graph"), chunk(3, "Quick Sort heap")]
    r = BM25Retriever(docs)
    assert r.search("C\u00e2y nh\u1ecb ph\u00e2n")[0]["chunk_id"] == "1"
    assert r.search("BFS")[0]["chunk_id"] == "2"
    assert r.search("Quick Sort")[0]["chunk_id"] == "3"
    assert r.search("binary c\u00e2y nh\u1ecb ph\u00e2n")[0]["metadata"]["source"] == "test"
    assert r.search("") == []
    assert BM25Retriever([]).search("tree") == []

def test_accent_and_false_positives():
    docs = [chunk(1, "m\u00e1"), chunk(2, "m\u00e0"), chunk(3, "heap")]
    assert [x["chunk_id"] for x in BM25Retriever(docs).search("ma")] == []
    assert len(BM25Retriever(docs, accent_insensitive=True).search("ma")) == 2

def test_invalid_and_duplicate():
    with pytest.raises(ValueError): BM25Retriever([chunk(1, "a"), chunk(1, "b")])
    with pytest.raises(ValueError): BM25Retriever([{}])


def test_d_stroke_and_jsonl(tmp_path):
    import json
    data = chunk(1, "\u0111\u1ed3 th\u1ecb graph")
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    index = BM25Retriever.from_jsonl(path, accent_insensitive=True)
    assert index.search("do thi")[0]["chunk_id"] == "1"
    assert "d" not in tokenize("?", accent_insensitive=True)
