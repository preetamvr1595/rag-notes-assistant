"""Unit tests for tokenization utilities."""
from rag_notes.tokenize import tokenize

def test_tokenize_simple():
    tokens = tokenize("Hello, World! 123")
    assert "hello" in tokens
    assert "world" in tokens

def test_tokenize_empty():
    assert tokenize("") == []

def test_tokenize_special_characters():
    tokens = tokenize("MinMaxScaler(feature_range=(0, 1))")
    assert "minmaxscaler" in tokens or "minmax" in tokens
