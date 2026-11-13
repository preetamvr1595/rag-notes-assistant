"""Unit tests for API /ask and /search endpoints."""
from app.api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_ask_endpoint():
    payload = {"query": "How do I create a numpy array?", "top_k": 3}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()

def test_search_endpoint():
    response = client.get("/search?q=pandas")
    assert response.status_code == 200
    assert "results" in response.json()
