"""Unit tests for retrieval and ranking."""
from rag_notes.retrievers import BM25Retriever

def test_bm25_retriever_initialization():
    corpus = ["numpy arrays are fast", "pandas dataframes for tables"]
    retriever = BM25Retriever(corpus)
    scores = retriever.score("numpy")
    assert scores[0] > scores[1]
