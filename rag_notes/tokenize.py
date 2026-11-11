"""Tokenizer that understands code: `np.where`, `groupby`, `read_csv`, `DataFrame` -> useful sub-tokens."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS) - {"not", "no", "never"}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)*")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _stem(w: str) -> str:
    """Very light suffix stripping so 'merging'/'merge'/'merges' match."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks: list[str] = []
    for m in _WORD.finditer(text):
        raw = m.group(0)
        parts = [p for piece in re.split(r"[._]", raw) for p in _CAMEL.findall(piece)]
        low = [p.lower() for p in parts]
        if len(low) > 1:
            toks.append(raw.lower())             # keep the compound: np.where, read_csv
        for p in low:
            if (keep_stop or p not in STOP) and len(p) > 1:
                toks.append(_stem(p))
    return toks
