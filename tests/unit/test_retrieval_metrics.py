import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.metrics.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank

def test_metrics():
    assert precision_at_k(["a", "b"], {"b", "c"}, 5) == pytest.approx(.2)
    assert recall_at_k(["a", "b"], {"b", "c"}, 5) == pytest.approx(.5)
    assert reciprocal_rank(["a", "b"], {"b", "c"}) == .5
    with pytest.raises(ValueError): recall_at_k([], [], 5)
