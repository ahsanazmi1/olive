"""
Policy DSL models for Olive service.

This module defines the Policy DSL structure supporting:
- prefer_rail: <ACH|debit|credit>
- loyalty_rebate_pct: number
- early_pay_discount_bps: number
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class RailType(str, Enum):
    """Supported rail types for policy preferences."""
    ACH = "ACH"
    DEBIT = "debit"
    CREDIT = "credit"
    BNPL = "bnpl"
    STABLECOIN = "stablecoin"
    PREPAID = "prepaid"


class PolicyConditionType(str, Enum):
    """Types of policy conditions."""
    AMOUNT_RANGE = "amount_range"
    TIME_WINDOW = "time_window"
    MERCHANT_CATEGORY = "merchant_category"
    CUSTOMER_SEGMENT = "customer_segment"
    PAYMENT_FREQUENCY = "payment_frequency"
    RISK_LEVEL = "risk_level"


class PolicyActionType(str, Enum):
    """Types of policy actions."""
    RAIL_PREFERENCE = "rail_preference"
    REBATE_APPLICATION = "rebate_application"
    DISCOUNT_APPLICATION = "discount_application"
    LOYALTY_BOOST = "loyalty_boost"
    TAX_VALIDATION = "tax_validation"
    EARLY_PAY_INCENTIVE = "early_pay_incentive"


class PolicyCondition(BaseModel):
    """Policy condition for conditional policy application."""
    
    condition_type: PolicyConditionType = Field(..., description="Type of condition")
    field: str = Field(..., description="Field to evaluate")
    operator: str = Field(..., description="Comparison operator (eq, gt, lt, gte, lte, in, between)")
    value: Any = Field(..., description="Value to compare against")
    description: str = Field(..., description="Human-readable condition description")


class PolicyAction(BaseModel):
    """Policy action to be applied when conditions are met."""
    
    action_type: PolicyActionType = Field(..., description="Type of action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    weight: float = Field(1.0, description="Action weight/multiplier", ge=0.0, le=10.0)
    description: str = Field(..., description="Human-readable action description")


class PolicyDSL(BaseModel):
    """Policy DSL model for Olive service."""
    
    # Policy identification
    policy_id: str = Field(..., description="Unique policy identifier")
    policy_name: str = Field(..., description="Human-readable policy name")
    description: str = Field(..., description="Policy description")
    
    # Policy rules (legacy fields for backward compatibility)
    prefer_rail: Optional[RailType] = Field(None, description="Preferred rail type")
    loyalty_rebate_pct: float = Field(0.0, description="Loyalty rebate percentage", ge=0.0, le=100.0)
    early_pay_discount_bps: float = Field(0.0, description="Early payment discount in basis points", ge=0.0, le=10000.0)
    
    # Enhanced policy rules for Phase 4
    conditions: List[PolicyCondition] = Field(default_factory=list, description="Policy conditions")
    actions: List[PolicyAction] = Field(default_factory=list, description="Policy actions")
    
    # Merchant routing rules
    rebate_rules: Dict[str, float] = Field(default_factory=dict, description="Rail-specific rebate percentages")
    early_pay_rules: Dict[str, float] = Field(default_factory=dict, description="Rail-specific early pay discounts")
    loyalty_incentives: Dict[str, float] = Field(default_factory=dict, description="Loyalty program incentives")
    tax_validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Tax validation requirements")
    
    # Policy enforcement settings
    enforcement_mode: str = Field("advisory", description="Enforcement mode: advisory, mandatory, override")
    override_threshold: float = Field(0.0, description="Threshold for policy override", ge=0.0, le=1.0)
    
    # Policy metadata
    merchant_id: Optional[str] = Field(None, description="Merchant this policy applies to")
    effective_from: datetime = Field(default_factory=datetime.now, description="Policy effective date")
    effective_until: Optional[datetime] = Field(None, description="Policy expiration date")
    priority: int = Field(1, description="Policy priority (higher = more important)", ge=1, le=10)
    
    # Policy status
    enabled: bool = Field(True, description="Whether policy is active")
    
    @validator('loyalty_rebate_pct')
    def validate_rebate_percentage(cls, v):
        """Validate loyalty rebate percentage."""
        if v < 0 or v > 100:
            raise ValueError("Loyalty rebate percentage must be between 0 and 100")
        return v
    
    @validator('early_pay_discount_bps')
    def validate_discount_bps(cls, v):
        """Validate early payment discount in basis points."""
        if v < 0 or v > 10000:
            raise ValueError("Early payment discount must be between 0 and 10000 basis points")
        return v


class PolicyAdjustment(BaseModel):
    """Normalized policy adjustments for scoring systems."""
    
    # Cost adjustments
    cost_bps_delta: float = Field(0.0, description="Cost adjustment in basis points", ge=-10000.0, le=10000.0)
    
    # Reward adjustments
    reward_bonus_delta: float = Field(0.0, description="Reward bonus adjustment", ge=-1.0, le=1.0)
    
    # Rail preference adjustments
    rail_preference_boost: float = Field(0.0, description="Rail preference boost score", ge=0.0, le=1.0)
    
    # Policy metadata
    policy_id: str = Field(..., description="Policy that generated this adjustment")
    adjustment_reason: str = Field(..., description="Human-readable reason for adjustment")
    
    # Applied conditions
    conditions_met: List[str] = Field(default_factory=list, description="Conditions that triggered this adjustment")
    
    @validator('cost_bps_delta')
    def validate_cost_delta(cls, v):
        """Validate cost adjustment range."""
        if v < -10000 or v > 10000:
            raise ValueError("Cost adjustment must be between -10000 and 10000 basis points")
        return v
    
    @validator('reward_bonus_delta')
    def validate_reward_delta(cls, v):
        """Validate reward adjustment range."""
        if v < -1.0 or v > 1.0:
            raise ValueError("Reward adjustment must be between -1.0 and 1.0")
        return v


class PolicyEvaluationRequest(BaseModel):
    """Request for policy evaluation."""
    
    # Transaction context
    transaction_amount: float = Field(..., description="Transaction amount", gt=0.0)
    merchant_id: str = Field(..., description="Merchant identifier")
    
    # Rail candidates
    rail_candidates: List[Dict[str, Any]] = Field(..., description="Available rail options with scores")
    
    # Current scores
    current_scores: Dict[str, float] = Field(..., description="Current rail scores")
    
    # Additional context
    channel: str = Field("online", description="Transaction channel")
    currency: str = Field("USD", description="Transaction currency")
    trace_id: str = Field(..., description="Transaction trace ID")


class PolicyEvaluationResponse(BaseModel):
    """Response from policy evaluation."""
    
    # Policy adjustments
    adjustments: List[PolicyAdjustment] = Field(..., description="Applied policy adjustments")
    
    # Updated scores
    updated_scores: Dict[str, float] = Field(..., description="Rail scores after policy adjustments")
    
    # Policy metadata
    applied_policies: List[str] = Field(..., description="Policies that were applied")
    ignored_policies: List[str] = Field(default_factory=list, description="Policies that were ignored")
    
    # Evaluation metadata
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    trace_id: str = Field(..., description="Transaction trace ID")
    
    # Summary
    winner_rail: Optional[str] = Field(None, description="Winning rail after policy application")
    policy_impact: str = Field(..., description="Summary of policy impact")


class PolicyAppliedEvent(BaseModel):
    """CloudEvent data for policy application."""
    
    # Event metadata
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field("ocn.olive.policy_applied.v1", description="CloudEvent type")
    source: str = Field("olive", description="Event source")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Transaction context
    trace_id: str = Field(..., description="Transaction trace ID")
    merchant_id: str = Field(..., description="Merchant identifier")
    transaction_amount: float = Field(..., description="Transaction amount")
    
    # Policy evaluation results
    applied_policies: List[str] = Field(..., description="Policies that were applied")
    before_scores: Dict[str, float] = Field(..., description="Rail scores before policy application")
    after_scores: Dict[str, float] = Field(..., description="Rail scores after policy application")
    
    # Policy adjustments
    adjustments: List[PolicyAdjustment] = Field(..., description="Applied policy adjustments")
    
    # Results
    winner_rail: str = Field(..., description="Winning rail after policy application")
    policy_impact_summary: str = Field(..., description="Summary of policy impact")
    llm_explanation: str = Field(..., description="LLM-generated explanation of policy application")
    
    # Metadata
    evaluation_duration_ms: float = Field(..., description="Policy evaluation duration in milliseconds")
    policy_count: int = Field(..., description="Number of policies evaluated")


class PolicyStorage:
    """Simple in-memory policy storage for demo purposes."""
    
    def __init__(self):
        self._policies: Dict[str, PolicyDSL] = {}
        self._merchant_policies: Dict[str, List[str]] = {}  # merchant_id -> policy_ids
    
    def store_policy(self, policy: PolicyDSL) -> None:
        """Store a policy."""
        self._policies[policy.policy_id] = policy
        
        # Update merchant policy mapping
        if policy.merchant_id:
            if policy.merchant_id not in self._merchant_policies:
                self._merchant_policies[policy.merchant_id] = []
            if policy.policy_id not in self._merchant_policies[policy.merchant_id]:
                self._merchant_policies[policy.merchant_id].append(policy.policy_id)
    
    def get_policy(self, policy_id: str) -> Optional[PolicyDSL]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)
    
    def get_merchant_policies(self, merchant_id: str) -> List[PolicyDSL]:
        """Get all policies for a merchant."""
        policy_ids = self._merchant_policies.get(merchant_id, [])
        return [self._policies[pid] for pid in policy_ids if pid in self._policies]
    
    def list_policies(self) -> List[PolicyDSL]:
        """List all stored policies."""
        return list(self._policies.values())
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        if policy_id in self._policies:
            policy = self._policies[policy_id]
            del self._policies[policy_id]
            
            # Update merchant policy mapping
            if policy.merchant_id and policy.merchant_id in self._merchant_policies:
                if policy_id in self._merchant_policies[policy.merchant_id]:
                    self._merchant_policies[policy.merchant_id].remove(policy_id)
            
            return True
        return False


# Global policy storage instance
_policy_storage: Optional[PolicyStorage] = None


def get_policy_storage() -> PolicyStorage:
    """Get the global policy storage instance."""
    global _policy_storage
    if _policy_storage is None:
        _policy_storage = PolicyStorage()
    return _policy_storage


# Enhanced MCP Request Models for Phase 4

class SetPolicyRequest(BaseModel):
    """Request model for setting a policy."""
    
    policy: PolicyDSL = Field(..., description="Policy to set")
    overwrite: bool = Field(False, description="Whether to overwrite existing policy")


class GetPolicyRequest(BaseModel):
    """Request model for getting a policy."""
    
    policy_id: Optional[str] = Field(None, description="Specific policy ID to get")
    merchant_id: Optional[str] = Field(None, description="Merchant ID to get policies for")
    enabled_only: bool = Field(True, description="Only return enabled policies")


class EvaluatePoliciesRequest(BaseModel):
    """Request model for evaluating policies."""
    
    evaluation_request: PolicyEvaluationRequest = Field(..., description="Policy evaluation request")
    include_explanations: bool = Field(True, description="Include policy explanations in response")
    enforce_mode: str = Field("advisory", description="Enforcement mode: advisory, mandatory, override")


class PolicyEnforcementRequest(BaseModel):
    """Request model for policy enforcement during negotiation."""
    
    merchant_id: str = Field(..., description="Merchant identifier")
    negotiation_context: Dict[str, Any] = Field(..., description="Negotiation context from Orca/Opal")
    trace_id: str = Field(..., description="Transaction trace ID")
    enforcement_mode: str = Field("advisory", description="Enforcement mode")

