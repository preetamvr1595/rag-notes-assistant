"""Turn .ipynb study notebooks into clean Markdown files (text + code + short text outputs; images dropped)."""
import sys, json, re
from pathlib import Path
import nbformat

MAX_OUT = 500


def nb_to_md(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    parts = []
    for c in nb.cells:
        src = c.source.strip()
        if not src:
            continue
        if c.cell_type == "markdown":
            parts.append(src)
        elif c.cell_type == "code":
            parts.append("```python\n" + src + "\n```")
            outs = []
            for o in c.get("outputs", []):
                if o.output_type == "stream":
                    outs.append(o.text)
                elif o.output_type in ("execute_result", "display_data"):
                    t = o.get("data", {}).get("text/plain", "")
                    if t and not t.startswith("<Figure") and not t.startswith("<Axes"):
                        outs.append(t)
            out = "".join(outs).strip()
            if out:
                out = out[:MAX_OUT] + (" …" if len(out) > MAX_OUT else "")
                parts.append("Output:\n```text\n" + out + "\n```")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.rglob("*.ipynb")):
        out = dst_dir / (p.stem + ".md")
        out.write_text(nb_to_md(p), encoding="utf-8")
        print(out.name, len(out.read_text()) // 1024, "KB")
