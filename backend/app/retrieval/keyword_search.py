"""C1: BM25 keyword retrieval — sparse vector từ rank_bm25."""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer đơn giản (hỗ trợ cả tiếng Việt lẫn tiếng Anh)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase + split theo ký tự không phải chữ/số."""
    return re.findall(r"[a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Vocabulary (global, build khi index)
# ---------------------------------------------------------------------------

_vocab: dict[str, int] = {}   # token → index


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Xây dựng vocabulary từ danh sách văn bản."""
    global _vocab
    all_tokens: set[str] = set()
    for t in texts:
        all_tokens.update(_tokenize(t))
    _vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}
    logger.info(f"Vocabulary size: {len(_vocab)}")
    return _vocab


def text_to_sparse_vector(text: str, vocab: dict[str, int] | None = None) -> tuple[list[int], list[float]]:
    """
    Chuyển văn bản thành TF sparse vector (indices + values).
    Dùng với vocab đã build. Trả về (indices, values) chuẩn Qdrant SparseVector.
    """
    v = vocab or _vocab
    if not v:
        raise RuntimeError("Vocab chưa được build. Gọi build_vocab() trước.")
    tokens = _tokenize(text)
    if not tokens:
        return [], []
    counts = Counter(tokens)
    total = sum(counts.values())
    indices = []
    values = []
    for tok, cnt in counts.items():
        if tok in v:
            indices.append(v[tok])
            values.append(cnt / total)   # TF normalized
    return indices, values


def save_vocab(path: str) -> None:
    """Lưu vocab ra file JSON để dùng lại."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_vocab, f, ensure_ascii=False)
    logger.info(f"Vocab saved: {path}")


def load_vocab(path: str) -> dict[str, int]:
    """Load vocab từ file JSON."""
    global _vocab
    import json
    with open(path, encoding="utf-8") as f:
        _vocab = json.load(f)
    logger.info(f"Vocab loaded: {len(_vocab)} tokens from {path}")
    return _vocab
