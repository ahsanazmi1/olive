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

# Import ML-enhanced evaluator
try:
    from src.olive.ml_enhanced_evaluator import get_ml_enhanced_evaluator
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML models not available: {e}")
    ML_AVAILABLE = False

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


@app.get("/incentives", response_model=Dict[str, Any])
async def get_incentives(
    merchant_id: str = "demo_merchant_001",
    transaction_amount: float = 410.40,
    channel: str = "online"
) -> Dict[str, Any]:
    """
    Get available incentives and rewards for a transaction.
    
    Args:
        merchant_id: Merchant identifier
        transaction_amount: Transaction amount
        channel: Transaction channel
        
    Returns:
        Available incentives and rewards
    """
    # Calculate dynamic incentive values based on transaction amount
    base_cashback_rate = 0.02  # 2% base cashback
    category_bonus_rate = 0.03  # 3% for clothing/fashion
    volume_threshold = 300.0
    volume_bonus_rate = 0.01 if transaction_amount >= volume_threshold else 0.0
    
    total_cashback_rate = base_cashback_rate + category_bonus_rate + volume_bonus_rate
    cashback_value = transaction_amount * total_cashback_rate
    
    incentives = [
        {
            "id": "base_cashback",
            "name": "Base Cashback Reward",
            "description": f"Standard {base_cashback_rate*100:.0f}% cashback on all purchases",
            "type": "cashback",
            "value": f"{base_cashback_rate*100:.0f}%",
            "amount": transaction_amount * base_cashback_rate,
            "currency": "USD",
            "eligibility": "all_users",
            "status": "active",
            "expires_at": None
        },
        {
            "id": "fashion_category_bonus",
            "name": "Fashion Category Bonus",
            "description": f"Extra {category_bonus_rate*100:.0f}% cashback on clothing and fashion purchases",
            "type": "cashback",
            "value": f"{category_bonus_rate*100:.0f}%",
            "amount": transaction_amount * category_bonus_rate,
            "currency": "USD",
            "eligibility": "clothing_category",
            "status": "active",
            "expires_at": "2024-12-31T23:59:59Z"
        },
        {
            "id": "volume_discount",
            "name": "Volume Bonus",
            "description": f"Additional {volume_bonus_rate*100:.0f}% cashback for purchases over ${volume_threshold}",
            "type": "cashback",
            "value": f"{volume_bonus_rate*100:.0f}%" if volume_bonus_rate > 0 else "0%",
            "amount": transaction_amount * volume_bonus_rate,
            "currency": "USD",
            "eligibility": f"purchases_over_{volume_threshold}",
            "status": "active" if volume_bonus_rate > 0 else "inactive",
            "expires_at": None
        },
        {
            "id": "loyalty_points",
            "name": "Loyalty Points",
            "description": "Earn loyalty points for every dollar spent",
            "type": "points",
            "value": f"{int(transaction_amount)} points",
            "amount": transaction_amount,
            "currency": "points",
            "eligibility": "all_users",
            "status": "active",
            "expires_at": None
        },
        {
            "id": "early_adopter_bonus",
            "name": "Early Adopter Bonus",
            "description": "Special bonus for early participants in the OCN ecosystem",
            "type": "bonus",
            "value": "$5.00",
            "amount": 5.0,
            "currency": "USD",
            "eligibility": "new_users",
            "status": "active",
            "expires_at": "2024-12-31T23:59:59Z"
        }
    ]
    
    # Filter active incentives
    active_incentives = [inc for inc in incentives if inc["status"] == "active"]
    
    return {
        "success": True,
        "data": {
            "incentives": active_incentives,
            "count": len(active_incentives),
            "total_active": len(active_incentives),
            "total_value": sum(inc["amount"] for inc in active_incentives if inc["type"] == "cashback"),
            "categories": list(set(inc["type"] for inc in active_incentives)),
            "summary": {
                "total_cashback_rate": f"{total_cashback_rate*100:.1f}%",
                "total_cashback_value": cashback_value,
                "loyalty_points": int(transaction_amount),
                "bonus_value": 5.0 if any(inc["id"] == "early_adopter_bonus" for inc in active_incentives) else 0.0
            }
        }
    }


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
        
        # Evaluate policies (ML-enhanced if available)
        if ML_AVAILABLE:
            evaluator = get_ml_enhanced_evaluator()
        else:
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


@app.get("/ml/status", response_model=Dict[str, Any])
async def get_ml_status():
    """Get ML model status and configuration."""
    if not ML_AVAILABLE:
        return {
            "ml_enabled": False,
            "error": "ML models not available"
        }
    
    try:
        from src.olive.ml.policy_optimization import get_policy_optimizer
        from src.olive.ml.incentive_effectiveness import get_effectiveness_predictor
        
        policy_optimizer = get_policy_optimizer()
        effectiveness_predictor = get_effectiveness_predictor()
        
        return {
            "ml_enabled": True,
            "ml_weight": 0.7,  # From MLEnhancedPolicyEvaluator
            "models": {
                "policy_optimization": {
                    "loaded": policy_optimizer.is_loaded,
                    "model_type": policy_optimizer.metadata.get("model_type", "unknown"),
                    "version": policy_optimizer.metadata.get("version", "unknown"),
                    "training_date": policy_optimizer.metadata.get("trained_on", "unknown"),
                    "features": len(policy_optimizer.feature_names) if policy_optimizer.feature_names else 0
                },
                "incentive_effectiveness": {
                    "loaded": effectiveness_predictor.is_loaded,
                    "model_type": effectiveness_predictor.metadata.get("model_type", "unknown"),
                    "version": effectiveness_predictor.metadata.get("version", "unknown"),
                    "training_date": effectiveness_predictor.metadata.get("trained_on", "unknown"),
                    "features": len(effectiveness_predictor.feature_names) if effectiveness_predictor.feature_names else 0
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML status: {str(e)}",
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
