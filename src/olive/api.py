"""
FastAPI application for Olive service.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# Add project root to Python path for MCP imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server import mcp_router  # noqa: E402
from src.olive.policies.models import PolicyDSL, get_policy_storage  # noqa: E402
from src.olive.policies.loader import get_policy_loader  # noqa: E402
from src.olive.policies.evaluator import get_policy_evaluator, PolicyEvaluationRequest  # noqa: E402
from src.olive.policies.events import emit_policy_applied_event  # noqa: E402
from src.olive.policies.explainer import generate_policy_explanation  # noqa: E402

# Create FastAPI application
app = FastAPI(
    title="Olive Service",
    description="Olive service for the Open Checkout Network (OCN)",
    version="0.1.0",
    contact={
        "name": "OCN Team",
        "email": "team@ocn.ai",
        "url": "https://github.com/ahsanazmi1/olive",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Include MCP router
app.include_router(mcp_router)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "olive"}


# Policy DSL API Endpoints

class PolicyCreateRequest(BaseModel):
    """Request model for creating a policy."""
    policy_data: Dict[str, Any]


class PolicyEvaluationAPIRequest(BaseModel):
    """Request model for policy evaluation."""
    merchant_id: str
    transaction_amount: float
    rail_candidates: List[Dict[str, Any]]
    current_scores: Dict[str, float]
    trace_id: str
    channel: str = "online"
    currency: str = "USD"


@app.post("/policies", response_model=Dict[str, Any])
async def create_policy(request: PolicyCreateRequest) -> Dict[str, Any]:
    """
    Create a new policy using the Policy DSL.
    
    Args:
        request: Policy creation request
        
    Returns:
        Policy creation result
    """
    try:
        loader = get_policy_loader()
        policy = loader.load_policy_from_dict(request.policy_data)
        
        storage = get_policy_storage()
        storage.store_policy(policy)
        
        return {
            "success": True,
            "policy_id": policy.policy_id,
            "policy_name": policy.policy_name,
            "message": f"Policy '{policy.policy_name}' created successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create policy: {str(e)}"
        )


@app.get("/policies/{policy_id}", response_model=Dict[str, Any])
async def get_policy(policy_id: str) -> Dict[str, Any]:
    """
    Get a policy by ID.
    
    Args:
        policy_id: Policy identifier
        
    Returns:
        Policy data
    """
    storage = get_policy_storage()
    policy = storage.get_policy(policy_id)
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy not found: {policy_id}"
        )
    
    return {
        "success": True,
        "policy": policy.dict(),
        "message": f"Policy '{policy.policy_name}' retrieved successfully"
    }


@app.get("/policies", response_model=Dict[str, Any])
async def list_policies(merchant_id: str = None) -> Dict[str, Any]:
    """
    List all policies, optionally filtered by merchant.
    
    Args:
        merchant_id: Optional merchant filter
        
    Returns:
        List of policies
    """
    storage = get_policy_storage()
    
    if merchant_id:
        policies = storage.get_merchant_policies(merchant_id)
    else:
        policies = storage.list_policies()
    
    policy_dicts = [policy.dict() for policy in policies]
    
    return {
        "success": True,
        "policies": policy_dicts,
        "count": len(policies),
        "merchant_filter": merchant_id,
        "message": f"Retrieved {len(policies)} policy(ies)"
    }


@app.post("/policies/evaluate", response_model=Dict[str, Any])
async def evaluate_policies(request: PolicyEvaluationAPIRequest) -> Dict[str, Any]:
    """
    Evaluate policies for a given transaction context.
    
    Args:
        request: Policy evaluation request
        
    Returns:
        Policy evaluation results
    """
    try:
        # Create evaluation request
        eval_request = PolicyEvaluationRequest(
            merchant_id=request.merchant_id,
            transaction_amount=request.transaction_amount,
            rail_candidates=request.rail_candidates,
            current_scores=request.current_scores,
            trace_id=request.trace_id,
            channel=request.channel,
            currency=request.currency,
        )
        
        # Evaluate policies
        evaluator = get_policy_evaluator()
        evaluation_response = evaluator.evaluate_policies(eval_request)
        
        # Generate explanation
        explanation = generate_policy_explanation(
            applied_policies=evaluation_response.applied_policies,
            adjustments=evaluation_response.adjustments,
            before_scores=request.current_scores,
            after_scores=evaluation_response.updated_scores,
            winner_rail=evaluation_response.winner_rail or "unknown"
        )
        
        # Emit CloudEvent
        await emit_policy_applied_event(
            evaluation_response=evaluation_response,
            merchant_id=request.merchant_id,
            transaction_amount=request.transaction_amount,
            before_scores=request.current_scores,
            llm_explanation=explanation,
            evaluation_duration_ms=0.0  # Would be calculated in real implementation
        )
        
        # Convert adjustments to dictionaries
        adjustment_dicts = [adj.dict() for adj in evaluation_response.adjustments]
        
        return {
            "success": True,
            "adjustments": adjustment_dicts,
            "updated_scores": evaluation_response.updated_scores,
            "applied_policies": evaluation_response.applied_policies,
            "ignored_policies": evaluation_response.ignored_policies,
            "winner_rail": evaluation_response.winner_rail,
            "policy_impact": evaluation_response.policy_impact,
            "explanation": explanation,
            "trace_id": evaluation_response.trace_id,
            "message": "Policy evaluation completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to evaluate policies: {str(e)}"
        )


def main() -> None:
    """Main entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "olive.api:app",
        host="127.0.0.1",  # Use localhost for development security
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
