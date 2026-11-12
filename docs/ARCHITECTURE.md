# RAG Notes Assistant Architecture

```
User Query
    │
    ▼
[BM25 Retriever] ──┐
                   ├─► [Reciprocal Rank Fusion] ──► [Refusal Gate]
[Char n-grams]   ──┘                                       │
                                                           ├─ Score < 0.65 ──► "Not found in notes"
                                                           │
                                                           └─ Score >= 0.65 ─► [Citation Grounding]
                                                                                    │
                                                                                    ▼
                                                                           Answer with [1], [2] Citations
```
