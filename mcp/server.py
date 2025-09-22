"""
MCP (Model Context Protocol) server implementation for Olive service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class MCPRequest(BaseModel):
    """MCP request model."""

    verb: str
    args: dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP response model."""

    success: bool = True
    data: dict[str, Any] = {}


# Create MCP router
mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


@mcp_router.post("/invoke", response_model=MCPResponse)
async def mcp_invoke(request: MCPRequest) -> MCPResponse:
    """
    MCP protocol endpoint for Olive service operations.

    Args:
        request: MCP request containing verb and arguments

    Returns:
        MCP response with operation result

    Raises:
        HTTPException: If verb is not supported
    """
    verb = request.verb

    if verb == "getStatus":
        return await get_status()
    elif verb == "listIncentives":
        return await list_incentives()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported verb: {verb}. Supported verbs: getStatus, listIncentives",
        )


async def get_status() -> MCPResponse:
    """
    Get the current status of the Olive agent.

    Returns:
        MCP response with agent status
    """
    return MCPResponse(success=True, data={"ok": True, "agent": "olive"})


async def list_incentives() -> MCPResponse:
    """
    List available incentives and rewards in the OCN ecosystem.

    Returns:
        MCP response with stub incentives data
    """
    # Stub incentives data for Olive service
    incentives = [
        {
            "id": "early_adopter_bonus",
            "name": "Early Adopter Bonus",
            "description": "Special reward for early participants in the OCN ecosystem",
            "type": "cashback",
            "value": "5%",
            "currency": "USD",
            "eligibility": "new_users",
            "status": "active",
            "expires_at": "2024-12-31T23:59:59Z",
        },
        {
            "id": "volume_discount",
            "name": "Volume Discount",
            "description": "Reduced fees for high-volume transactions",
            "type": "fee_reduction",
            "value": "0.1%",
            "currency": "USD",
            "eligibility": "volume_tier_2",
            "status": "active",
            "expires_at": None,
        },
        {
            "id": "referral_reward",
            "name": "Referral Reward",
            "description": "Earn rewards for referring new merchants to OCN",
            "type": "cashback",
            "value": "$50",
            "currency": "USD",
            "eligibility": "verified_merchants",
            "status": "active",
            "expires_at": "2024-12-31T23:59:59Z",
        },
        {
            "id": "loyalty_points",
            "name": "Loyalty Points",
            "description": "Accumulate points for every transaction processed",
            "type": "points",
            "value": "1 point per $10",
            "currency": "points",
            "eligibility": "all_users",
            "status": "active",
            "expires_at": None,
        },
    ]

    return MCPResponse(
        success=True,
        data={
            "incentives": incentives,
            "count": len(incentives),
            "total_active": len([i for i in incentives if i["status"] == "active"]),
            "categories": list({i["type"] for i in incentives}),
        },
    )
