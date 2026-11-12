"""Data structures and domain models for RAG Notes."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class Document:
    """Represents a source raw document loaded from disk."""
    doc_id: str
    series: str
    filename: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Passage:
    """Represents a chunked passage ready for indexing."""
    passage_id: str
    doc_id: str
    series: str
    file_title: str
    heading_path: str
    text: str
    is_code: bool = False
    start_char: int = 0
    end_char: int = 0
