"""Unit tests for evaluation metrics."""
from rag_notes.evaluate import hit_at_k, reciprocal_rank

def test_hit_at_k():
    ranked = ["docA", "docB", "docC"]
    assert hit_at_k(ranked, "docA", k=1) == 1.0
    assert hit_at_k(ranked, "docB", k=1) == 0.0
    assert hit_at_k(ranked, "docB", k=3) == 1.0

def test_reciprocal_rank():
    ranked = ["docA", "docB", "docC"]
    assert reciprocal_rank(ranked, "docA") == 1.0
    assert reciprocal_rank(ranked, "docB") == 0.5
    assert reciprocal_rank(ranked, "docZ") == 0.0
