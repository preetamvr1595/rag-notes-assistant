"""FastAPI application entrypoint for RAG Notes Assistant."""
from fastapi import FastAPI, HTTPException
import time
from typing import Dict, Any
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(
    title="RAG Notes Assistant API",
    version="0.3.0",
    description="REST API for offline retrieval and grounded note question answering."
)

START_TIME = time.time()

@app.get("/")
def root():
    return {"message": "RAG Notes Assistant API is running", "docs": "/docs"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "rag-notes-assistant"}

@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "status": "operational"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    # Simulated grounded answering logic for API
    return QueryResponse(
        query=req.query,
        answer="Retrieved from study notes index.",
        refused=False,
        confidence_score=0.88,
        citations=[]
    )
