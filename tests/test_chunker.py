"""Unit tests for heading-aware chunker."""
from rag_notes.chunker import chunk_markdown

def test_chunk_markdown_heading():
    md = "# Heading 1\nContent under 1.\n## Heading 2\nContent under 2."
    chunks = chunk_markdown(md, series="test", filename="test.md")
    assert len(chunks) >= 1
