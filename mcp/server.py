"""
MCP (Model Context Protocol) server implementation for Olive service.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import policy modules
from src.olive.policies.models import (
    PolicyDSL, 
    get_policy_storage,
    SetPolicyRequest,
    GetPolicyRequest,
    EvaluatePoliciesRequest,
    PolicyEnforcementRequest
)
from src.olive.policies.loader import get_policy_loader
from src.olive.policies.evaluator import get_policy_evaluator, PolicyEvaluationRequest
from src.olive.policies.enhanced_evaluator import get_enhanced_evaluator

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


# Enhanced MCP Verb Handlers for Phase 4

async def handle_set_policy(request: SetPolicyRequest) -> MCPResponse:
    """Handle setPolicy MCP verb."""
    try:
        storage = get_policy_storage()
        
        # Check if policy already exists
        existing_policy = storage.get_policy(request.policy.policy_id)
        if existing_policy and not request.overwrite:
            return MCPResponse(
                success=False,
                data={
                    "error": f"Policy {request.policy.policy_id} already exists",
                    "message": "Use overwrite=true to replace existing policy"
                }
            )
        
        # Store the policy
        storage.set_policy(request.policy)
        
        logger.info(f"Policy {request.policy.policy_id} set successfully")
        
        return MCPResponse(
            success=True,
            data={
                "policy_id": request.policy.policy_id,
                "policy_name": request.policy.policy_name,
                "merchant_id": request.policy.merchant_id,
                "enforcement_mode": request.policy.enforcement_mode,
                "message": "Policy set successfully"
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


async def handle_get_policy(request: GetPolicyRequest) -> MCPResponse:
    """Handle getPolicy MCP verb."""
    try:
        storage = get_policy_storage()
        
        if request.policy_id:
            # Get specific policy
            policy = storage.get_policy(request.policy_id)
            if not policy:
                return MCPResponse(
                    success=False,
                    data={
                        "error": f"Policy {request.policy_id} not found",
                        "message": "Policy does not exist"
                    }
                )
            
            policies = [policy.dict()]
        elif request.merchant_id:
            # Get policies for merchant
            policies = storage.get_policies_by_merchant(request.merchant_id)
            if request.enabled_only:
                policies = [p for p in policies if p.enabled]
            policies = [p.dict() for p in policies]
        else:
            # Get all policies
            all_policies = storage.get_all_policies()
            if request.enabled_only:
                all_policies = [p for p in all_policies if p.enabled]
            policies = [p.dict() for p in all_policies]
        
        return MCPResponse(
            success=True,
            data={
                "policies": policies,
                "count": len(policies),
                "message": f"Retrieved {len(policies)} policies"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting policies: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to get policies"
            }
        )


async def handle_evaluate_policies_enhanced(request: EvaluatePoliciesRequest) -> MCPResponse:
    """Handle evaluatePolicies MCP verb with enhanced functionality."""
    try:
        enhanced_evaluator = get_enhanced_evaluator()
        
        # Evaluate policies with enhanced evaluator
        response = enhanced_evaluator.evaluate_policies_enhanced(request.evaluation_request)
        
        # Convert adjustments to dictionaries
        adjustment_dicts = []
        for adjustment in response.adjustments:
            adjustment_dicts.append({
                "policy_id": adjustment.policy_id,
                "adjustment_type": adjustment.adjustment_type,
                "rail_type": adjustment.rail_type,
                "adjustment_value": adjustment.adjustment_value,
                "description": adjustment.description
            })
        
        return MCPResponse(
            success=True,
            data={
                "adjustments": adjustment_dicts,
                "updated_scores": response.updated_scores,
                "evaluation_metadata": response.evaluation_metadata,
                "message": "Enhanced policy evaluation completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced policy evaluation: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to evaluate policies with enhanced evaluator"
            }
        )


async def handle_enforce_policies(request: PolicyEnforcementRequest) -> MCPResponse:
    """Handle policy enforcement during negotiation."""
    try:
        enhanced_evaluator = get_enhanced_evaluator()
        
        # Enforce policies during negotiation
        enforcement_results = enhanced_evaluator.enforce_policies_during_negotiation(request)
        
        return MCPResponse(
            success=True,
            data={
                "enforcement_results": enforcement_results,
                "message": "Policy enforcement completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error enforcing policies: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": "Failed to enforce policies during negotiation"
            }
        )


# Enhanced MCP verb dispatcher
async def handle_mcp_request_enhanced(request: MCPRequest) -> MCPResponse:
    """Handle MCP requests with enhanced verb support."""
    
    verb_handlers = {
        "setPolicy": handle_set_policy,
        "getPolicy": handle_get_policy,
        "evaluatePolicies": handle_evaluate_policies_enhanced,
        "enforcePolicies": handle_enforce_policies,
    }
    
    handler = verb_handlers.get(request.verb)
    if not handler:
        return MCPResponse(
            success=False,
            data={
                "error": f"Unknown verb: {request.verb}",
                "supported_verbs": list(verb_handlers.keys()),
                "message": "Verb not supported"
            }
        )
    
    try:
        # Parse request args based on verb
        if request.verb == "setPolicy":
            policy_request = SetPolicyRequest(**request.args)
            return await handler(policy_request)
        elif request.verb == "getPolicy":
            get_request = GetPolicyRequest(**request.args)
            return await handler(get_request)
        elif request.verb == "evaluatePolicies":
            eval_request = EvaluatePoliciesRequest(**request.args)
            return await handler(eval_request)
        elif request.verb == "enforcePolicies":
            enforce_request = PolicyEnforcementRequest(**request.args)
            return await handler(enforce_request)
        else:
            return await handler(request.args)
            
    except Exception as e:
        logger.error(f"Error handling MCP request {request.verb}: {e}")
        return MCPResponse(
            success=False,
            data={
                "error": str(e),
                "message": f"Failed to handle {request.verb} request"
            }
        )
