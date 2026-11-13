"""Answer synthesis, citation extraction, and out-of-scope refusal gate."""
# Configurable synthesis temperature for LLM backend
"""Answer synthesis, citation extraction, and out-of-scope refusal gate."""
"""Answer generation with citations, a grounding prompt, a refusal gate, and pluggable LLM backends."""
from __future__ import annotations
import json, os, re, urllib.request
from dataclasses import dataclass, field
import numpy as np
from .index import Index
from .tokenize import STOP, tokenize

SYSTEM_PROMPT = (
    "You are a study assistant. Answer ONLY from the numbered context passages below, which come from the student's own notes. "
    "Cite passages like [1] or [2][3] after the sentences they support. Keep code in fenced code blocks. "
    "If the context does not contain the answer, reply exactly: \"I couldn't find that in your notes.\" Never use outside knowledge."
)
REFUSAL = "I couldn't find that in your notes."


@dataclass
class Answer:
    text: str
    sources: list[dict]
    confidence: float
    answered: bool
    backend: str
    hits: list = field(default_factory=list, repr=False)


def _doc_freq(index: Index):
    if not hasattr(index, "_df"):
        from collections import Counter
        df = Counter()
        for c in index.chunks:
            df.update(set(tokenize(c.index_text)))
        index._df = df
    return index._df


def query_coverage(index: Index, query: str, hits, weighted: bool = False) -> float:
    """How much of the question's vocabulary do the top passages cover?
    weighted=True counts each question word by its rarity in the notes. It did NOT beat the plain version on the dev questions
    (see notebook 03), so the default is plain coverage."""
    q = {t for t in tokenize(query) if "." not in t}
    if not q:
        return 0.0
    text_tokens = set()
    for c, _ in hits:
        text_tokens |= set(tokenize(c.index_text))
    if not weighted:
        return len(q & text_tokens) / len(q)
    df, n = _doc_freq(index), len(index.chunks)
    w = {t: np.log((n + 1) / (df.get(t, 0) + 0.5)) for t in q}
    return sum(w[t] for t in q if t in text_tokens) / sum(w.values())


def build_prompt(question: str, hits, max_chars: int = 6000) -> list[dict]:
    parts, used = [], 0
    for i, (c, _) in enumerate(hits, 1):
        block = f"[{i}] ({c.source} › {c.section.split(' > ')[-1]})\n{c.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block); used += len(block)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Context:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"}]


# ----------------------------------------------------------------------------- backends
class ExtractiveBackend:
    """Offline default: no language model. Returns the best-matching note passage(s) with code, with sources."""
    name = "extractive"

    def generate(self, question: str, hits, **_) -> str:
        qt = set(tokenize(question))
        out = []
        for i, (c, _) in enumerate(hits[:2], 1):
            paras = [p for p in re.split(r"\n\s*\n", c.text) if p.strip() and not p.lstrip().startswith("#")]
            def score(p):
                if p.lstrip().startswith("```"):
                    return 0.5 + 0.1 * len(qt & set(tokenize(p)))
                return len(qt & set(tokenize(p))) / (1 + 0.002 * len(p))
            best = sorted(paras, key=score, reverse=True)[:2]
            best.sort(key=paras.index)
            out.append(f"**From your notes — {c.section.split(' > ')[-1]}** [{i}]\n\n" + "\n\n".join(best))
        return "\n\n---\n\n".join(out)


class OpenAICompatBackend:
    """Any OpenAI-compatible chat endpoint. Works with Ollama (http://localhost:11434/v1), vLLM, LM Studio, Groq, OpenAI..."""
    name = "openai-compatible"

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llama3", api_key: str = "ollama", timeout: int = 120):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout

    def generate(self, question: str, hits, **_) -> str:
        body = json.dumps({"model": self.model, "messages": build_prompt(question, hits), "temperature": 0.1}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


def backend_from_env():
    """RAG_BACKEND=extractive (default) | ollama | openai ; see README."""
    kind = os.getenv("RAG_BACKEND", "extractive").lower()
    if kind == "ollama":
        return OpenAICompatBackend(os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), os.getenv("RAG_MODEL", "llama3"))
    if kind == "openai":
        return OpenAICompatBackend(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.getenv("RAG_MODEL", "gpt-4o-mini"),
                                   os.getenv("OPENAI_API_KEY", ""))
    return ExtractiveBackend()


# ----------------------------------------------------------------------------- pipeline
class RAG:
    def __init__(self, index: Index, backend=None, k: int = 5, min_coverage: float = 0.65):
        self.index, self.backend, self.k, self.min_coverage = index, backend or ExtractiveBackend(), k, min_coverage

    def ask(self, question: str, series: str | None = None) -> Answer:
        hits = self.index.search(question, k=self.k, series=series)
        cov = query_coverage(self.index, question, hits) if hits else 0.0
        sources = [{"n": i, "source": c.source, "section": c.section.split(" > ")[-1], "series": c.series, "kind": c.kind}
                   for i, (c, _) in enumerate(hits, 1)]
        if not hits or cov < self.min_coverage:
            return Answer(REFUSAL, sources, cov, False, self.backend.name, hits)
        try:
            text = self.backend.generate(question, hits)
        except Exception as e:      # LLM server down etc. -> fall back to extractive so the app still works
            text = f"_(LLM backend unavailable: {type(e).__name__}. Showing note excerpts instead.)_\n\n" + ExtractiveBackend().generate(question, hits)
            return Answer(text, sources, cov, True, "extractive-fallback", hits)
        return Answer(text, sources, cov, text.strip() != REFUSAL, self.backend.name, hits)
