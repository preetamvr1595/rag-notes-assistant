"""Run the same dev/test retrieval evaluation WITH neural embeddings (MiniLM). Needs internet once.
    pip install sentence-transformers
    python scripts/eval_minilm.py
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from rag_notes.evaluate import load_golden, retrieval_metrics
from rag_notes.index import Index
from rag_notes.retrievers import HybridRetriever

g = load_golden()
idx = Index.build("data/notes", use_minilm=True)
R = idx.retrievers
variants = {"bm25": R["bm25"], "minilm": R["minilm"], "bm25+char": HybridRetriever([R["bm25"], R["char"]]),
            "bm25+minilm": HybridRetriever([R["bm25"], R["minilm"]]),
            "bm25+char+minilm": HybridRetriever([R["bm25"], R["char"], R["minilm"]])}
for split in ("dev", "test"):
    items = [i for i in g["in_scope"] if i["split"] == split]
    rows = {}
    for name, r in variants.items():
        idx.hybrid = r
        m, _ = retrieval_metrics(idx, items, "hybrid"); rows[name] = m
    print(f"\n{split.upper()} ({len(items)} questions)")
    print(pd.DataFrame(rows).T[["hit@1", "hit@3", "hit@5", "hit@10", "MRR"]].round(3))
