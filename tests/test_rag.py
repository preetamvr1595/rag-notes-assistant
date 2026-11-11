import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rag_notes.answer import (REFUSAL, RAG, ExtractiveBackend, OpenAICompatBackend, build_prompt, query_coverage)
from rag_notes.chunker import build_chunks
from rag_notes.index import Index
from rag_notes.loader import Document, load_documents
from rag_notes.tokenize import tokenize

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_splits_code_identifiers_and_keeps_compound():
    t = tokenize("Use np.where and read_csv with DataFrame")
    assert "np.where" in t and "read_csv" in t
    assert "np" in t and "csv" in t and "data" in t and "frame" in t


def test_tokenizer_drops_stop_words_but_keeps_not():
    t = tokenize("how do I do this and not that")
    assert "not" in t and "how" not in t and "the" not in t


# ---------------------------------------------------------------- chunker
MD = """# Demo Notes (Series)

## 1. Intro

Short intro.

```python
# this is a comment, NOT a heading
x = 1
```

## 2. Second topic

Some longer text about merge and join. """ + "word " * 80 + """

## 14. Summary — What You Learned

list of everything
"""


def test_code_comment_is_not_a_heading_and_nav_is_dropped():
    ch = build_chunks([Document("s/demo.md", "s", "notes", MD)], min_chars=0)
    text = "\n".join(c.text for c in ch)
    assert "# this is a comment, NOT a heading" in text      # kept inside the code fence
    assert not any("What You Learned" in c.section for c in ch)
    assert all(c.section.startswith("Demo Notes") for c in ch)  # '(Series)' boilerplate stripped


def test_exercise_questions_without_code_are_kept():
    md = "# 100 things\n\n#### 1. Import the package under the name np\n\n#### 2. Print the version of the package\n"
    ch = build_chunks([Document("exercises/x.md", "exercises", "exercises", md)])
    assert len(ch) == 2


def test_max_chars_respected():
    big = "# T\n\n## A\n\n" + "\n\n".join("para " * 40 for _ in range(30))
    ch = build_chunks([Document("s/b.md", "s", "notes", big)], max_chars=800)
    assert len(ch) > 3 and max(c.n_chars for c in ch) <= 1000


def test_loader_reads_md_and_ipynb(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "n.md").write_text("# x\n\nhello world, this is a note about pandas merge", encoding="utf-8")
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); nb.cells = [nbf.v4.new_markdown_cell("# NB\n\ntext here for the notebook"), nbf.v4.new_code_cell("print(1)")]
    nbf.write(nb, tmp_path / "a" / "b.ipynb")
    docs = load_documents(tmp_path)
    assert {d.path for d in docs} == {"a/n.md", "a/b.ipynb"} and all(d.series == "a" for d in docs)


# ---------------------------------------------------------------- real corpus (integration)
@pytest.fixture(scope="module")
def index():
    return Index.build(str(ROOT / "data" / "notes"))


def test_index_has_expected_size(index):
    assert 400 < len(index.chunks) < 1500
    assert {c.series for c in index.chunks} >= {"numpy", "pandas", "matplotlib", "seaborn", "scikit-learn"}


@pytest.mark.parametrize("q,needle", [
    ("How do I join two tables on a common column like in SQL?", "merge()"),
    ("What is the difference between ravel and flatten?", "flatten()"),
    ("How do I choose the number of clusters for k-means?", "Elbow"),
    ("How do I draw two y-axes with different scales on one chart?", "Twin Axes"),
])
def test_known_questions_find_the_right_section_in_top5(index, q, needle):
    hits = index.search(q, k=5)
    assert any(needle.lower() in (c.section + c.text[:200]).lower() for c, _ in hits)


def test_series_filter(index):
    hits = index.search("heatmap", k=5, series="seaborn")
    assert hits and all(c.series == "seaborn" for c, _ in hits)


def test_out_of_scope_question_is_refused(index):
    ans = RAG(index).ask("How do I deploy pods on Kubernetes?")
    assert ans.answered is False and ans.text == REFUSAL


def test_in_scope_question_is_answered_with_sources(index):
    ans = RAG(index).ask("How do I join two tables on a common column like in SQL?")
    assert ans.answered and ans.sources and "merge" in ans.text.lower()


def test_save_and_load_roundtrip(index, tmp_path):
    p = tmp_path / "i.joblib"; index.save(str(p)); idx2 = Index.load(str(p))
    q = "difference between loc and iloc"
    assert [c.id for c, _ in index.search(q)] == [c.id for c, _ in idx2.search(q)]


# ---------------------------------------------------------------- prompt + LLM backend
def test_prompt_numbers_sources_and_forbids_outside_knowledge(index):
    hits = index.search("what does np.where do", k=3)
    msgs = build_prompt("what does np.where do", hits)
    assert msgs[0]["role"] == "system" and REFUSAL in msgs[0]["content"] and "ONLY" in msgs[0]["content"]
    assert "[1]" in msgs[1]["content"] and "[3]" in msgs[1]["content"] and "Question: what does np.where do" in msgs[1]["content"]


class _Handler(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen = {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        out = json.dumps({"choices": [{"message": {"content": "Use pd.merge [1]."}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_openai_compatible_backend_against_a_local_fake_server(index):
    srv = HTTPServer(("127.0.0.1", 0), _Handler); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = OpenAICompatBackend(f"http://127.0.0.1:{srv.server_port}/v1", "llama3", "k")
        ans = RAG(index, be).ask("How do I join two tables on a common column like in SQL?")
        assert ans.text == "Use pd.merge [1]." and ans.backend == "openai-compatible"
        assert _Handler.seen["path"] == "/v1/chat/completions" and _Handler.seen["auth"] == "Bearer k"
        assert _Handler.seen["body"]["model"] == "llama3" and _Handler.seen["body"]["messages"][0]["role"] == "system"
    finally:
        srv.shutdown()


def test_backend_down_falls_back_to_extractive(index):
    be = OpenAICompatBackend("http://127.0.0.1:9/v1", "llama3", timeout=2)
    ans = RAG(index, be).ask("How do I join two tables on a common column like in SQL?")
    assert ans.answered and ans.backend == "extractive-fallback" and "unavailable" in ans.text
