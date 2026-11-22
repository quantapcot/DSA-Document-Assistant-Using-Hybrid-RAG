
"""Retrieval ranking metrics over explicit relevance judgments."""

from collections.abc import Mapping
from math import log2


def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _require_complete_labels(labels_complete):
    if not isinstance(labels_complete, bool):
        raise TypeError("labels_complete must be a boolean")
    if not labels_complete:
        raise ValueError(
            "Incomplete relevance judgments: "
            "the metric cannot be reported as fully evaluated"
        )


def precision_at_k(retrieved, relevant, k=5):
    """Fraction of the first k ranking positions that are relevant."""
    _validate_k(k)

    relevant_ids = set(relevant)
    found = set(retrieved[:k]) & relevant_ids

    return len(found) / k


def recall_at_k(retrieved, relevant, k=5, *, labels_complete=True):
    """Fraction of all labeled relevant IDs found in Top-K."""
    _validate_k(k)
    _require_complete_labels(labels_complete)

    relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValueError(
            "ground truth must contain at least one relevant chunk"
        )

    found = set(retrieved[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved, relevant, k=None):
    """Reciprocal rank of the first relevant hit.

    k=None preserves compatibility with existing callers.
    """
    if k is not None:
        _validate_k(k)

    relevant_ids = set(relevant)
    results = retrieved if k is None else retrieved[:k]

    for rank, chunk_id in enumerate(results, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mrr_at_k(retrieved_lists, relevant_lists, k=5):
    """Mean Reciprocal Rank at K across multiple queries."""
    _validate_k(k)

    if not retrieved_lists or len(retrieved_lists) != len(relevant_lists):
        raise ValueError(
            "Nonempty query batches with matching lengths are required"
        )

    scores = [
        reciprocal_rank(retrieved, relevant, k=k)
        for retrieved, relevant in zip(retrieved_lists, relevant_lists)
    ]

    return sum(scores) / len(scores)


def ndcg_at_k(retrieved, relevance_grades, k=5, *, labels_complete=True):
    """Normalized Discounted Cumulative Gain at K.

    Grades: 0 = irrelevant, 1 = supporting, 2 = directly relevant.
    Repeated IDs receive gain only at their first occurrence.
    Ranking positions are never shifted to fill duplicates.
    """
    _validate_k(k)
    _require_complete_labels(labels_complete)

    if not isinstance(relevance_grades, Mapping):
        raise TypeError("relevance_grades must be a mapping")

    for grade in relevance_grades.values():
        if isinstance(grade, bool) or grade not in (0, 1, 2):
            raise ValueError("relevance grades must be 0, 1, or 2")

    def gain(grade, rank):
        return (2 ** grade - 1) / log2(rank + 1)

    dcg = 0.0
    seen = set()

    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        grade = relevance_grades.get(chunk_id, 0)
        dcg += gain(grade, rank)

    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]

    idcg = sum(
        gain(grade, rank)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg
