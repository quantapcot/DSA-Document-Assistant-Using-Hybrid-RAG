"""BM25 lexical search; index is rebuilt in memory from the supplied corpus."""
from __future__ import annotations
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

logger = logging.getLogger(__name__)
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_TOP_K = 20
_WORD_RE = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)

def tokenize(text: str, *, accent_insensitive: bool = False) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    if accent_insensitive:
        text = "".join(c for c in unicodedata.normalize("NFD", text.replace("\u0111", "d")) if not unicodedata.combining(c))
    segmented = ViTokenizer.tokenize(text)
    return _WORD_RE.findall(segmented)

def normalize_chunk(chunk):
    if hasattr(chunk, "model_dump"):
        chunk = chunk.model_dump()
    elif hasattr(chunk, "dict") and not isinstance(chunk, dict):
        chunk = chunk.dict()
    if not isinstance(chunk, dict):
        raise ValueError("chunk must be a DocumentChunk or dictionary")
    required = ("chunk_id", "document_id", "filename", "page", "text")
    if any(k not in chunk for k in required) or any(not isinstance(chunk[k], str) or not chunk[k].strip() for k in ("chunk_id", "document_id", "filename", "text")) or not isinstance(chunk["page"], int):
        raise ValueError("invalid chunk: required identity, page and nonempty text")
    result = dict(chunk)
    result.setdefault("section", None)
    result.setdefault("metadata", {})
    if not isinstance(result["metadata"], dict):
        raise ValueError("chunk metadata must be a dictionary")
    return result

class BM25Retriever:
    def __init__(self, chunks, *, k1=DEFAULT_K1, b=DEFAULT_B, accent_insensitive=False):
        self.accent_insensitive = accent_insensitive
        self.chunks = []
        self._term_sets = []
        corpus = []
        seen = set()
        for chunk in chunks:
            record = normalize_chunk(chunk)
            if record["chunk_id"] in seen:
                raise ValueError(f"duplicate chunk_id: {record['chunk_id']}")
            seen.add(record["chunk_id"])
            terms = tokenize(record["text"], accent_insensitive=accent_insensitive)
            if not terms:
                raise ValueError("chunk contains no indexable terms")
            self.chunks.append(record)
            corpus.append(terms)
            self._term_sets.append(set(terms))

        self._model = BM25Okapi(corpus, k1=k1, b=b) if corpus else None
        logger.info("BM25 index ready: %d chunks", len(self.chunks))

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        accent_insensitive: bool = False,
    ) -> "BM25Retriever":
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSONL on line {line_no}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object on line {line_no}")
                records.append(record)
        return cls(records, k1=k1, b=b, accent_insensitive=accent_insensitive)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0 or self._model is None:
            return []
        query_terms = tokenize(query, accent_insensitive=self.accent_insensitive)
        if not query_terms:
            return []

        scores = self._model.get_scores(query_terms)
        query_set = set(query_terms)
        matches = [
            i for i, indexed_terms in enumerate(self._term_sets)
            if not query_set.isdisjoint(indexed_terms)
        ]
        # BM25 scores may be zero/negative in tiny collections; filter by
        # lexical overlap instead of scores > 0.
        matches.sort(key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]))

        return [
            {
                **self.chunks[idx],
                "score": float(scores[idx]),
                "rank": position,
                "retrieval_method": "bm25",
            }
            for position, idx in enumerate(matches[:top_k], 1)
        ]
