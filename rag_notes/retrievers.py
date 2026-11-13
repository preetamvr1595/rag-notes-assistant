"""Sparse BM25, Character n-gram, and RRF rank fusion retrievers."""
from functools import lru_cache
# Optimized retrieval functions with query caching
"""Sparse BM25, Character n-gram, and RRF rank fusion retrievers."""
"""Three retrievers behind one interface + Reciprocal-Rank-Fusion hybrid."""
from __future__ import annotations
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .tokenize import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=self.k1, b=self.b)
        return self

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)


class LSARetriever:
    """Dense retrieval trained locally: TF-IDF -> SVD (latent semantic analysis). No downloads needed."""
    name = "lsa"

    def __init__(self, dims: int = 128, ngram_range=(1, 2)):
        self.dims, self.ngram_range = dims, ngram_range

    def fit(self, texts: list[str]):
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     ngram_range=self.ngram_range, sublinear_tf=True, min_df=1)
        X = self.tfidf.fit_transform(texts)
        d = min(self.dims, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=d, random_state=42)
        self.M = normalize(self.svd.fit_transform(X))
        return self

    def scores(self, query: str) -> np.ndarray:
        q = normalize(self.svd.transform(self.tfidf.transform([query])))
        return (self.M @ q.T).ravel()


class CharNgramRetriever:
    """TF-IDF over character n-grams: robust to word forms and glued API names (fill ~ fillna, largest ~ nlargest)."""
    name = "char"

    def __init__(self, ngram_range=(3, 5)):
        self.ngram_range = ngram_range

    def fit(self, texts: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True, min_df=2, lowercase=True)
        self.X = self.vec.fit_transform(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return (self.X @ q.T).toarray().ravel()


class SentenceTransformerRetriever:
    """Optional neural embeddings (needs `pip install sentence-transformers` and internet for the first download)."""
    name = "minilm"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model

    def fit(self, texts: list[str]):
        from sentence_transformers import SentenceTransformer     # imported lazily on purpose
        self.model = SentenceTransformer(self.model_name)
        self.M = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)
        return (self.M @ q.T).ravel()


def rrf(rank_lists: list[np.ndarray], weights: list[float] | None = None, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion over score arrays: score = sum_i w_i / (k + rank_i)."""
    weights = weights or [1.0] * len(rank_lists)
    fused = np.zeros_like(rank_lists[0], dtype=float)
    for w, s in zip(weights, rank_lists):
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        fused += w / (k + ranks)
    return fused


class HybridRetriever:
    name = "hybrid"

    def __init__(self, parts: list, weights: list[float] | None = None, k: int = 60):
        self.parts, self.weights, self.k = parts, weights, k

    def scores(self, query: str) -> np.ndarray:
        return rrf([p.scores(query) for p in self.parts], self.weights, self.k)
