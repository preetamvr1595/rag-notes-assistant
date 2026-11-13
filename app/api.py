"""FastAPI application entrypoint for RAG Notes Assistant."""
from fastapi import FastAPI, HTTPException
import time
from typing import Dict, Any

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
