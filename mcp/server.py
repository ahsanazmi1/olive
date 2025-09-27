"""
MCP (Model Context Protocol) server implementation for Olive service.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import policy modules
from src.olive.policies.models import PolicyDSL, get_policy_storage
from src.olive.policies.loader import get_policy_loader
from src.olive.policies.evaluator import get_policy_evaluator, PolicyEvaluationRequest

logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """MCP request model."""

    verb: str
    args: dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP response model."""

    success: bool = True
    data: dict[str, Any] = {}


class PolicyRequest(BaseModel):
    """Policy request model."""
    
    policy_data: dict[str, Any]


class PolicyEvaluationMCPRequest(BaseModel):
    """Policy evaluation MCP request model."""
    
    merchant_id: str
    transaction_amount: float
    rail_candidates: List[Dict[str, Any]]
    current_scores: Dict[str, float]
    trace_id: str
    channel: str = "online"
    currency: str = "USD"


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
    elif verb == "setPolicy":
        return await set_policy(request.args)
    elif verb == "getPolicy":
        return await get_policy(request.args)
    elif verb == "listPolicies":
        return await list_policies(request.args)
    elif verb == "evaluatePolicies":
        return await evaluate_policies(request.args)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported verb: {verb}. Supported verbs: getStatus, listIncentives, setPolicy, getPolicy, listPolicies, evaluatePolicies",
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
    incentives: list[dict[str, Any]] = [
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


async def set_policy(args: Dict[str, Any]) -> MCPResponse:
    """
    Set a policy using the Policy DSL.
    
    Args:
        args: Policy arguments containing policy_data
        
    Returns:
        MCP response with policy creation result
    """
    try:
        if "policy_data" not in args:
            raise ValueError("Missing required field: policy_data")
        
        policy_data = args["policy_data"]
        
        # Load policy from data
        loader = get_policy_loader()
        policy = loader.load_policy_from_dict(policy_data)
        
        # Store policy
        storage = get_policy_storage()
        storage.store_policy(policy)
        
        logger.info(f"Policy set successfully: {policy.policy_id}")
        
        return MCPResponse(
            success=True,
            data={
                "policy_id": policy.policy_id,
                "policy_name": policy.policy_name,
                "message": f"Policy '{policy.policy_name}' created successfully",
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting policy: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to set policy"
            }
        )


async def get_policy(args: Dict[str, Any]) -> MCPResponse:
    """
    Get a policy by ID.
    
    Args:
        args: Policy arguments containing policy_id
        
    Returns:
        MCP response with policy data
    """
    try:
        if "policy_id" not in args:
            raise ValueError("Missing required field: policy_id")
        
        policy_id = args["policy_id"]
        
        # Get policy from storage
        storage = get_policy_storage()
        policy = storage.get_policy(policy_id)
        
        if not policy:
            return MCPResponse(
                success=False,
                data={
                    "error": f"Policy not found: {policy_id}",
                    "message": "Policy does not exist"
                }
            )
        
        return MCPResponse(
            success=True,
            data={
                "policy": policy.dict(),
                "message": f"Policy '{policy.policy_name}' retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting policy: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to get policy"
            }
        )


async def list_policies(args: Dict[str, Any]) -> MCPResponse:
    """
    List all policies, optionally filtered by merchant.
    
    Args:
        args: Optional arguments containing merchant_id
        
    Returns:
        MCP response with list of policies
    """
    try:
        storage = get_policy_storage()
        
        # Get merchant_id filter if provided
        merchant_id = args.get("merchant_id")
        
        if merchant_id:
            policies = storage.get_merchant_policies(merchant_id)
        else:
            policies = storage.list_policies()
        
        # Convert policies to dictionaries
        policy_dicts = [policy.dict() for policy in policies]
        
        return MCPResponse(
            success=True,
            data={
                "policies": policy_dicts,
                "count": len(policies),
                "merchant_filter": merchant_id,
                "message": f"Retrieved {len(policies)} policy(ies)"
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing policies: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to list policies"
            }
        )


async def evaluate_policies(args: Dict[str, Any]) -> MCPResponse:
    """
    Evaluate policies for a given transaction context.
    
    Args:
        args: Policy evaluation arguments
        
    Returns:
        MCP response with policy evaluation results
    """
    try:
        # Validate required fields
        required_fields = ["merchant_id", "transaction_amount", "rail_candidates", "current_scores", "trace_id"]
        for field in required_fields:
            if field not in args:
                raise ValueError(f"Missing required field: {field}")
        
        # Create evaluation request
        eval_request = PolicyEvaluationRequest(
            merchant_id=args["merchant_id"],
            transaction_amount=args["transaction_amount"],
            rail_candidates=args["rail_candidates"],
            current_scores=args["current_scores"],
            trace_id=args["trace_id"],
            channel=args.get("channel", "online"),
            currency=args.get("currency", "USD"),
        )
        
        # Evaluate policies
        evaluator = get_policy_evaluator()
        evaluation_response = evaluator.evaluate_policies(eval_request)
        
        # Convert adjustments to dictionaries
        adjustment_dicts = [adj.dict() for adj in evaluation_response.adjustments]
        
        return MCPResponse(
            success=True,
            data={
                "adjustments": adjustment_dicts,
                "updated_scores": evaluation_response.updated_scores,
                "applied_policies": evaluation_response.applied_policies,
                "ignored_policies": evaluation_response.ignored_policies,
                "winner_rail": evaluation_response.winner_rail,
                "policy_impact": evaluation_response.policy_impact,
                "trace_id": evaluation_response.trace_id,
                "message": "Policy evaluation completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error evaluating policies: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to evaluate policies"
            }
        )
