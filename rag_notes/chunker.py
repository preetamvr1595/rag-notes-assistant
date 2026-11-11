"""Heading-aware Markdown chunker. Code fences are never split mid-line by accident or mistaken for headings."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .loader import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    id: int
    text: str            # what the user/LLM sees
    source: str          # file path
    section: str         # "Title > Heading > Subheading"
    series: str
    kind: str
    n_chars: int = 0
    has_code: bool = False
    index_text: str = field(default="", repr=False)   # text used for indexing (adds the section path)


def _blocks(md: str):
    """Yield ('heading', level, title) | ('code', text) | ('para', text) blocks."""
    lines, i = md.splitlines(), 0
    para: list[str] = []

    def flush():
        if para:
            t = "\n".join(para).strip()
            if t:
                yield ("para", t)
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("```"):
            yield from flush()
            fence, buf = ln, [ln]
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i]); i += 1
            if i < len(lines):
                buf.append(lines[i]); i += 1
            yield ("code", "\n".join(buf))
            continue
        m = HEADING.match(ln)
        if m:
            yield from flush()
            yield ("heading", len(m.group(1)), m.group(2).strip())
        elif not ln.strip():
            yield from flush()
        else:
            para.append(ln)
        i += 1
    yield from flush()


def _split_big(parts: list[str], max_chars: int) -> list[list[str]]:
    groups, cur, size = [], [], 0
    for p in parts:
        if cur and size + len(p) > max_chars:
            groups.append(cur); cur, size = [], 0
        cur.append(p); size += len(p) + 2
    if cur:
        groups.append(cur)
    return groups


def chunk_document(doc: Document, max_chars: int = 1200, min_chars: int = 200) -> list[dict]:
    """Return raw chunk dicts (text, section) for one document."""
    if doc.kind == "exercises":
        min_chars = 0                      # each exercise stays its own chunk
    title, stack, sections = doc.path.rsplit("/", 1)[-1], [], []
    cur_parts: list[str] = []
    cur_path = ""

    def close():
        nonlocal cur_parts
        if cur_parts:
            sections.append((cur_path, cur_parts))
        cur_parts = []

    for b in _blocks(doc.text):
        if b[0] == "heading":
            _, lvl, t = b
            if lvl == 1 and not stack:
                title = re.sub(r"\s*\(.*?\)\s*$", "", t)      # drop '(Preetham's ... Series)' boilerplate
            close()
            stack = [(l, h) for l, h in stack if l < lvl] + [(lvl, t)]
            cur_path = " > ".join(h for _, h in stack)
            cur_parts = [("#" * lvl) + " " + t]
        else:
            cur_parts.append(b[1])
    close()

    # merge tiny neighbouring sections, split oversized ones
    merged: list[tuple[str, list[str]]] = []
    for path, parts in sections:
        size = sum(len(p) for p in parts)
        if merged and min_chars and sum(len(p) for p in merged[-1][1]) < min_chars \
                and sum(len(p) for p in merged[-1][1]) + size <= max_chars:
            merged[-1] = (merged[-1][0], merged[-1][1] + parts)
        else:
            merged.append((path, list(parts)))
    out = []
    for path, parts in merged:
        for g in _split_big(parts, max_chars):
            text = "\n\n".join(g).strip()
            if len(text) >= 25 and (doc.kind == "exercises" or any(not x.lstrip().startswith("#") for x in g)):
                out.append({"text": text, "section": f"{title} > {path}" if path and not path.startswith(title) else (path or title)})
    return out


NAV = re.compile(r"^(\d+[a-z]?\.\s*)?(summary\b|next notebook)|series complete|what you learned", re.I)


def build_chunks(docs: list[Document], max_chars: int = 1200, min_chars: int = 200,
                 drop_nav: bool = True, heading_boost: int = 1) -> list[Chunk]:
    """drop_nav=True removes "Summary" / "Next notebook" chunks: they list every topic, so keyword search over-matches them."""
    chunks: list[Chunk] = []
    for d in docs:
        for c in chunk_document(d, max_chars, min_chars):
            if drop_nav and NAV.search(c["section"].split(" > ")[-1]):
                continue
            header = f"{d.series} | {c['section']}"
            chunks.append(Chunk(id=len(chunks), text=c["text"], source=d.path, section=c["section"],
                                series=d.series, kind=d.kind, n_chars=len(c["text"]),
                                has_code="```" in c["text"], index_text=("\n".join([header] * heading_boost)) + "\n" + c["text"]))
    return chunks
