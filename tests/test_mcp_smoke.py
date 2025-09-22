"""
MCP (Model Context Protocol) smoke tests for Olive service.
"""

from fastapi.testclient import TestClient

from olive.api import app

client = TestClient(app)


def test_mcp_invoke_get_status():
    """Test MCP getStatus verb returns correct response."""
    response = client.post(
        "/mcp/invoke",
        json={
            "verb": "getStatus",
            "args": {}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "success" in data
    assert "data" in data
    assert data["success"] is True
    
    # Check data content
    assert "ok" in data["data"]
    assert "agent" in data["data"]
    assert data["data"]["ok"] is True
    assert data["data"]["agent"] == "olive"


def test_mcp_invoke_list_incentives():
    """Test MCP listIncentives verb returns correct response."""
    response = client.post(
        "/mcp/invoke",
        json={
            "verb": "listIncentives",
            "args": {}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "success" in data
    assert "data" in data
    assert data["success"] is True
    
    # Check data content
    assert "incentives" in data["data"]
    assert "count" in data["data"]
    assert "total_active" in data["data"]
    assert "categories" in data["data"]
    
    # Check incentives structure
    incentives = data["data"]["incentives"]
    assert isinstance(incentives, list)
    assert len(incentives) > 0
    
    # Check first incentive structure
    first_incentive = incentives[0]
    required_fields = ["id", "name", "description", "type", "value", "currency", "eligibility", "status"]
    for field in required_fields:
        assert field in first_incentive
    
    # Check counts match
    assert data["data"]["count"] == len(incentives)
    assert data["data"]["total_active"] >= 0
    
    # Check categories
    assert isinstance(data["data"]["categories"], list)
    assert len(data["data"]["categories"]) > 0


def test_mcp_invoke_unsupported_verb():
    """Test MCP invoke with unsupported verb returns 400."""
    response = client.post(
        "/mcp/invoke",
        json={
            "verb": "unsupportedVerb",
            "args": {}
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Unsupported verb" in data["detail"]


def test_mcp_invoke_missing_verb():
    """Test MCP invoke with missing verb returns 422."""
    response = client.post(
        "/mcp/invoke",
        json={
            "args": {}
        }
    )
    
    assert response.status_code == 422


def test_mcp_invoke_response_schema_consistency():
    """Test that MCP responses have consistent schema."""
    verbs = ["getStatus", "listIncentives"]
    
    for verb in verbs:
        response = client.post(
            "/mcp/invoke",
            json={
                "verb": verb,
                "args": {}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All MCP responses should have these top-level fields
        assert "success" in data
        assert "data" in data
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"], dict)
