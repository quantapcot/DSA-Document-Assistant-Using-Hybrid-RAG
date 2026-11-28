
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


def recall_at_k(retrieved, relevant, k=5, *, labels_complete=True, zero_positive_policy="raise"):
    """Fraction of all labeled relevant IDs found in Top-K.

    *zero_positive_policy*:
      - ``"raise"`` (default, backward compatible): raise ValueError if *relevant* is empty.
      - ``"none"``: return ``None`` so macro-aggregators exclude undefined denominators explicitly.
    """
    _validate_k(k)
    _require_complete_labels(labels_complete)

    relevant_ids = set(relevant)
    if not relevant_ids:
        if zero_positive_policy == "none":
            return None
        raise ValueError(
            "ground truth must contain at least one relevant chunk"
        )

    found = set(retrieved[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def hit_rate_at_k(retrieved, relevant, k=5):
    """1.0 if at least one relevant chunk appears in Top-K, else 0.0."""
    _validate_k(k)
    relevant_ids = set(relevant)
    return 1.0 if (set(retrieved[:k]) & relevant_ids) else 0.0


def average_precision_at_k(retrieved, relevant, k=5, *, labels_complete=True, zero_positive_policy="raise"):
    """Average Precision at K with cutoff normalization denominator ``min(k, len(relevant_ids))``.

    Duplicate retrieved IDs are ignored after their first occurrence while preserving rank positions.
    """
    _validate_k(k)
    _require_complete_labels(labels_complete)

    relevant_ids = set(relevant)
    if not relevant_ids:
        if zero_positive_policy == "none":
            return None
        raise ValueError("ground truth must contain at least one relevant chunk")

    denom = min(k, len(relevant_ids))
    hits = 0
    precision_sum = 0.0
    seen = set()
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        if chunk_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denom


def f1_at_k(precision, recall):
    """Harmonic mean of per-query Precision@K and Recall@K; returns None if recall is undefined."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def judged_at_k(retrieved, judged_ids, k=5):
    """Return ``(judged_rate, judged_count, actual_returned_count)`` for the first K slots.

    Denominator is ``actual_returned_count = len(retrieved[:k])`` (0.0 if empty) so callers can
    inspect ``actual_returned_count`` alongside ``judged_rate`` without conflating fewer-than-K
    with full coverage.
    """
    _validate_k(k)
    top = list(retrieved[:k])
    returned_count = len(top)
    if returned_count == 0:
        return 0.0, 0, 0
    judged_set = set(judged_ids)
    judged_count = sum(1 for cid in top if cid in judged_set)
    return judged_count / returned_count, judged_count, returned_count


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
