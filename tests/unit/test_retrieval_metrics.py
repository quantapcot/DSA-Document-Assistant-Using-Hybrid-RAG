
import sys
from math import log2
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mrr_at_k,
    ndcg_at_k,
)


# Original baseline test
def test_metrics():
    assert precision_at_k(["a", "b"], {"b", "c"}, 5) == pytest.approx(0.2)
    assert recall_at_k(["a", "b"], {"b", "c"}, 5) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"b", "c"}) == 0.5

    with pytest.raises(ValueError):
        recall_at_k([], [], 5)


# K = 1, 3, 5
@pytest.mark.parametrize(
    "k, expected_precision, expected_recall, expected_rr",
    [
        (1, 0.0, 0.0, 0.0),
        (3, 2 / 3, 1.0, 0.5),
        (5, 2 / 5, 1.0, 0.5),
    ],
)
def test_metrics_at_k(
    k, expected_precision, expected_recall, expected_rr
):
    retrieved = ["x", "b", "c"]
    relevant = {"b", "c"}

    assert precision_at_k(
        retrieved, relevant, k
    ) == pytest.approx(expected_precision)

    assert recall_at_k(
        retrieved, relevant, k
    ) == pytest.approx(expected_recall)

    assert reciprocal_rank(
        retrieved, relevant, k
    ) == pytest.approx(expected_rr)


# Empty retrieval
def test_empty_retrieval():
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert reciprocal_rank([], {"a"}, 3) == 0.0
    assert ndcg_at_k([], {"a": 2}, 3) == 0.0


# Duplicate IDs must not inflate the scores
def test_duplicate_ids():
    retrieved = ["b", "b", "c"]
    relevant = {"b", "c"}

    assert precision_at_k(
        retrieved, relevant, 3
    ) == pytest.approx(2 / 3)

    assert recall_at_k(
        retrieved, relevant, 3
    ) == pytest.approx(1.0)

    expected_ndcg = (
        3 + 1 / log2(4)
    ) / (
        3 + 1 / log2(3)
    )

    assert ndcg_at_k(
        retrieved, {"b": 2, "c": 1}, 3
    ) == pytest.approx(expected_ndcg)


# Mean Reciprocal Rank
def test_mrr_at_k():
    retrieved = [
        ["a", "b"],
        ["c", "a"],
    ]
    relevant = [
        {"b"},
        {"c"},
    ]

    assert mrr_at_k(
        retrieved, relevant, 3
    ) == pytest.approx(0.75)

    assert mrr_at_k(
        retrieved, relevant, 1
    ) == pytest.approx(0.5)


# nDCG checked against manual calculation
def test_ndcg_manual():
    retrieved = ["a", "b", "c"]
    grades = {"b": 2, "c": 1}

    dcg = 3 / log2(3) + 1 / log2(4)
    idcg = 3 / log2(2) + 1 / log2(3)

    expected = dcg / idcg

    assert ndcg_at_k(
        retrieved, grades, 3
    ) == pytest.approx(expected)


# IDCG = 0
def test_ndcg_zero_idcg():
    assert ndcg_at_k(
        ["a", "b"], {"a": 0, "b": 0}, 3
    ) == 0.0

    assert ndcg_at_k(["a"], {}, 3) == 0.0


# Unknown retrieved IDs have zero gain
# when labels are declared complete
def test_ndcg_missing_grade():
    result = ndcg_at_k(
        ["unknown", "a"], {"a": 2}, 3
    )

    expected = (3 / log2(3)) / 3
    assert result == pytest.approx(expected)


# Incomplete judgments must not be silently
# reported as complete
def test_incomplete_labels():
    with pytest.raises(ValueError):
        recall_at_k(
            ["a"], {"a"}, 3, labels_complete=False
        )

    with pytest.raises(ValueError):
        ndcg_at_k(
            ["a"], {"a": 2}, 3, labels_complete=False
        )


# Invalid K
@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3"])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k)

    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k)

    with pytest.raises(ValueError):
        reciprocal_rank(["a"], {"a"}, k)

    with pytest.raises(ValueError):
        mrr_at_k([["a"]], [{"a"}], k)

    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 2}, k)


# Invalid graded relevance
@pytest.mark.parametrize("grade", [-1, 1.5, 3, True])
def test_invalid_grades(grade):
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": grade}, 3)


# Empty / mismatched MRR batches
def test_invalid_mrr_batches():
    with pytest.raises(ValueError):
        mrr_at_k([], [], 3)

    with pytest.raises(ValueError):
        mrr_at_k([["a"]], [], 3)


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 0.0),
        (
            3,
            (3 / log2(3) + 1 / log2(4))
            / (3 + 1 / log2(3)),
        ),
        (
            5,
            (3 / log2(3) + 1 / log2(4))
            / (3 + 1 / log2(3)),
        ),
    ],
)
def test_ndcg_at_required_k(k, expected):
    retrieved = ["x", "b", "c"]
    grades = {"b": 2, "c": 1}

    result = ndcg_at_k(retrieved, grades, k)

    assert result == pytest.approx(expected)
    assert 0.0 <= result <= 1.0
