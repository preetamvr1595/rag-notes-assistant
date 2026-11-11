"""Load documents (.md, .txt, .ipynb, .pdf) from a folder into plain-Markdown documents."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    path: str        # path relative to the notes root, e.g. "numpy/01_numpy_basics.md"
    series: str      # first folder name, e.g. "numpy"
    kind: str        # "notes" | "qa" | "exercises"
    text: str


KIND_BY_SERIES = {"interview-qa": "qa", "exercises": "exercises"}


def _read_ipynb(p: Path) -> str:
    import nbformat
    nb = nbformat.read(p, as_version=4)
    out = []
    for c in nb.cells:
        s = c.source.strip()
        if not s:
            continue
        out.append(s if c.cell_type == "markdown" else f"```python\n{s}\n```")
    return "\n\n".join(out)


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader
    pages = [(pg.extract_text() or "").strip() for pg in PdfReader(str(p)).pages]
    return "\n\n".join(f"## Page {i + 1}\n\n{t}" for i, t in enumerate(pages) if t)


def load_documents(root: str | Path) -> list[Document]:
    root = Path(root)
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".md", ".txt", ".ipynb", ".pdf"}:
            continue
        rel = p.relative_to(root)
        series = rel.parts[0] if len(rel.parts) > 1 else "general"
        text = (_read_ipynb(p) if p.suffix.lower() == ".ipynb"
                else _read_pdf(p) if p.suffix.lower() == ".pdf"
                else p.read_text(encoding="utf-8", errors="ignore"))
        if text.strip():
            docs.append(Document(str(rel).replace("\\", "/"), series, KIND_BY_SERIES.get(series, "notes"), text))
    return docs
