"""Domain exceptions for RAG notes pipeline."""

class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""
    pass

class IndexNotFoundError(RAGError):
    """Raised when index file is missing or corrupted."""
    pass

class EmptyCorpusError(RAGError):
    """Raised when source corpus directory has no readable documents."""
    pass

class RefusalGateError(RAGError):
    """Raised when query fails refusal calibration."""
    pass

class LLMBackendError(RAGError):
    """Raised when external LLM server is unreachable."""
    pass
