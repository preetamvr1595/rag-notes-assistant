"""Unit tests for FastAPI endpoints."""
from app.api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
