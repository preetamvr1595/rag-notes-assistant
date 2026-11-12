"""Unit tests for answer generation and refusal."""
from rag_notes.answer import should_refuse

def test_refusal_logic():
    # Below threshold should refuse
    assert should_refuse(score=0.4, threshold=0.65) is True
    # Above threshold should answer
    assert should_refuse(score=0.8, threshold=0.65) is False
