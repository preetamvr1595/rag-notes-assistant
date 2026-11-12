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

@dataclass
class SearchResult:
    """Represents a retrieved passage with relevance score."""
    passage: Passage
    score: float
    rank: int
    retriever_name: str

@dataclass
class Citation:
    """Citation reference linked to an answer."""
    citation_id: int
    source_file: str
    section: str
    snippet: str

@dataclass
class AnswerResult:
    """Complete synthesized answer response."""
    query: str
    answer: str
    refused: bool
    confidence_score: float
    citations: List[Citation] = field(default_factory=list)
    top_passages: List[SearchResult] = field(default_factory=list)
