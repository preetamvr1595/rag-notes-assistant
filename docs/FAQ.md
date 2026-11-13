# Frequently Asked Questions (FAQ)

### Can I run this completely offline without any internet connection?
Yes! The default retrieval pipeline (BM25 + Character n-grams) runs completely locally in pure Python.

### How does the refusal gate work?
It computes a normalized word-overlap and token score between top retrieved passages and the query. If the score falls below threshold 0.65, the assistant safely refuses.
