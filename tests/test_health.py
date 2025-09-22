"""
Tests for health check endpoint.
"""

from fastapi.testclient import TestClient

from olive.api import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint returns correct response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "repo": "olive"}


def test_health_check_response_structure():
    """Test health check endpoint response structure."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, dict)
    assert "ok" in data
    assert "repo" in data
    assert data["ok"] is True
    assert data["repo"] == "olive"