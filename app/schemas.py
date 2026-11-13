"""Pydantic request and response schemas for REST API."""
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to answer from study notes")
    top_k: int = Field(default=5, ge=1, le=20)
    strictness: float = Field(default=0.65, ge=0.0, le=1.0)

class CitationSchema(BaseModel):
    source_file: str
    section: str
    snippet: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    confidence_score: float
    citations: List[CitationSchema]
