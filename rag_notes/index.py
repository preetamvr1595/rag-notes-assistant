"""Build, save and query the search index."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np
from .chunker import Chunk, build_chunks
from .loader import load_documents
from .retrievers import BM25Retriever, CharNgramRetriever, HybridRetriever, LSARetriever, SentenceTransformerRetriever


class Index:
    def __init__(self, chunks: list[Chunk], retrievers: dict, kind_weights: dict | None = None):
        self.chunks, self.retrievers = chunks, retrievers
        self.kind_weights = kind_weights if kind_weights is not None else {"exercises": 0.6}
        self.hybrid = HybridRetriever([retrievers["bm25"], retrievers["char"]])   # lexical + character n-grams

    @classmethod
    def build(cls, notes_dir: str, max_chars: int = 1200, min_chars: int = 200, use_minilm: bool = False,
              lsa_dims: int = 128, drop_nav: bool = True, kind_weights: dict | None = None, bm25_b: float = 0.75,
              heading_boost: int = 2) -> "Index":
        chunks = build_chunks(load_documents(notes_dir), max_chars, min_chars, drop_nav, heading_boost)
        texts = [c.index_text for c in chunks]
        r = {"bm25": BM25Retriever(b=bm25_b).fit(texts), "char": CharNgramRetriever().fit(texts),
             "lsa": LSARetriever(dims=lsa_dims).fit(texts)}
        idx = cls(chunks, r, kind_weights)
        if use_minilm:
            r["minilm"] = SentenceTransformerRetriever().fit(texts)
            idx.hybrid = HybridRetriever([r["bm25"], r["char"], r["minilm"]])
        return idx

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks": self.chunks, "retrievers": self.retrievers, "kind_weights": self.kind_weights}, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "Index":
        d = joblib.load(path)
        idx = cls(d["chunks"], d["retrievers"], d.get("kind_weights"))
        if "minilm" in d["retrievers"]:
            idx.hybrid = HybridRetriever([d["retrievers"]["bm25"], d["retrievers"]["char"], d["retrievers"]["minilm"]])
        return idx

    def search(self, query: str, k: int = 5, method: str = "hybrid", series: str | None = None):
        r = self.hybrid if method == "hybrid" else self.retrievers[method]
        s = r.scores(query)
        if self.kind_weights:
            s = s * np.array([self.kind_weights.get(c.kind, 1.0) for c in self.chunks])
        if series:
            s = np.where(np.array([c.series == series for c in self.chunks]), s, -np.inf)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top if np.isfinite(s[i])]
