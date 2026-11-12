"""Evaluation metrics: hit@k, MRR, bootstrap confidence intervals."""
"""Retrieval evaluation: hit@k, MRR, and out-of-scope refusal."""
from __future__ import annotations
import json, re
import numpy as np


def _headings(chunk) -> str:
    return "\n".join(l for l in chunk.text.splitlines() if l.lstrip().startswith("#")) + "\n" + chunk.section


def matches(chunk, targets) -> bool:
    h = _headings(chunk)
    return any(t_src in chunk.source and t_sec.lower() in h.lower() for t_src, t_sec in targets)


def load_golden(path="eval/golden.json"):
    return json.load(open(path))


def validate_targets(index, golden):
    """Every target must exist in the corpus - guards against typos in the hand-written set."""
    bad = []
    for item in golden["in_scope"]:
        for t in item["targets"]:
            if not any(matches(c, [t]) for c in index.chunks):
                bad.append((item["id"], t))
    return bad


def rank_of_first_hit(index, item, method, k=10):
    res = index.search(item["question"], k=k, method=method)
    for r, (c, _) in enumerate(res, 1):
        if matches(c, item["targets"]):
            return r
    return None


def retrieval_metrics(index, items, method, ks=(1, 3, 5, 10)):
    ranks = [rank_of_first_hit(index, it, method, max(ks)) for it in items]
    out = {f"hit@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in ks}
    out["MRR"] = float(np.mean([1 / r if r else 0 for r in ranks]))
    out["n"] = len(items)
    return out, ranks
