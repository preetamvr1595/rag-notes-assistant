"""FastAPI application entrypoint for RAG Notes Assistant."""
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI(
    title="RAG Notes Assistant API",
    version="0.3.0",
    description="REST API for offline retrieval and grounded note question answering."
)

@app.get("/")
def root():
    return {"message": "RAG Notes Assistant API is running", "docs": "/docs"}
