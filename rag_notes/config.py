"""Configuration management for RAG pipeline."""
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    data_dir: str = os.getenv("RAG_DATA_DIR", "data/notes")
    index_path: str = os.getenv("RAG_INDEX_PATH", "models/index.joblib")
    bm25_k1: float = float(os.getenv("RAG_BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("RAG_BM25_B", "0.75"))
    refusal_threshold: float = float(os.getenv("RAG_REFUSAL_THRESHOLD", "0.65"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    backend: str = os.getenv("RAG_BACKEND", "offline")
    model_name: str = os.getenv("RAG_MODEL", "llama3")

DEFAULT_CONFIG = RAGConfig()
