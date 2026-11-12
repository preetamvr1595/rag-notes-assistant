"""Unit tests for file loaders."""
import os
from rag_notes.loader import load_notes

def test_load_notes_directory():
    if os.path.exists("data/notes"):
        docs = load_notes("data/notes")
        assert len(docs) > 0
