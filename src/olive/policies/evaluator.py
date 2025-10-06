"""
Policy evaluation engine for Olive service.

This module provides policy evaluation logic that outputs normalized adjustments
for Orca/Opal scoring systems.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    PolicyAdjustment,
    PolicyDSL,
    PolicyEvaluationRequest,
    PolicyEvaluationResponse,
    get_policy_storage,
)

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Policy evaluation engine."""
    
    def __init__(self):
        self.storage = get_policy_storage()
    
    def evaluate_policies(
        self, 
        request: PolicyEvaluationRequest
    ) -> PolicyEvaluationResponse:
        """
        Evaluate applicable policies and return adjustments.
        
        Args:
            request: Policy evaluation request with transaction context
            
        Returns:
            Policy evaluation response with adjustments and updated scores
        """
        start_time = datetime.now()
        
        logger.info(
            f"Evaluating policies for merchant {request.merchant_id}",
            extra={
                "merchant_id": request.merchant_id,
                "transaction_amount": request.transaction_amount,
                "trace_id": request.trace_id,
            }
        )
        
        # Get applicable policies for the merchant
        applicable_policies = self._get_applicable_policies(request)
        
        # Evaluate each policy
        adjustments = []
        applied_policies = []
        ignored_policies = []
        
        for policy in applicable_policies:
            try:
                adjustment = self._evaluate_single_policy(policy, request)
                if adjustment:
                    adjustments.append(adjustment)
                    applied_policies.append(policy.policy_id)
                    logger.debug(
                        f"Applied policy {policy.policy_id}: {adjustment.adjustment_reason}",
                        extra={
                            "policy_id": policy.policy_id,
                            "adjustment": adjustment.dict(),
                        }
                    )
                else:
                    ignored_policies.append(policy.policy_id)
                    logger.debug(f"Ignored policy {policy.policy_id} - no applicable conditions")
            except Exception as e:
                logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
                ignored_policies.append(policy.policy_id)
        
        # Apply adjustments to scores
        updated_scores = self._apply_adjustments(request.current_scores, adjustments)
        
        # Determine winner
        winner_rail = self._determine_winner(updated_scores)
        
        # Generate policy impact summary
        policy_impact = self._generate_policy_impact_summary(adjustments, applied_policies, ignored_policies)
        
        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            f"Policy evaluation completed",
            extra={
                "trace_id": request.trace_id,
                "applied_policies": len(applied_policies),
                "ignored_policies": len(ignored_policies),
                "winner_rail": winner_rail,
                "evaluation_time_ms": evaluation_time,
            }
        )
        
        return PolicyEvaluationResponse(
            adjustments=adjustments,
            updated_scores=updated_scores,
            applied_policies=applied_policies,
            ignored_policies=ignored_policies,
            trace_id=request.trace_id,
            winner_rail=winner_rail,
            policy_impact=policy_impact,
        )
    
    def _get_applicable_policies(self, request: PolicyEvaluationRequest) -> List[PolicyDSL]:
        """Get policies applicable to the current request."""
        # Get merchant-specific policies
        merchant_policies = self.storage.get_merchant_policies(request.merchant_id)
        
        # Filter by effective dates and enabled status
        applicable = []
        now = datetime.now()
        
        for policy in merchant_policies:
            if not policy.enabled:
                continue
            
            # Check effective date
            if policy.effective_from > now:
                continue
            
            if policy.effective_until and policy.effective_until < now:
                continue
            
            applicable.append(policy)
        
        # Sort by priority (higher priority first)
        applicable.sort(key=lambda p: p.priority, reverse=True)
        
        return applicable
    
    def _evaluate_single_policy(
        self, 
        policy: PolicyDSL, 
        request: PolicyEvaluationRequest
    ) -> Optional[PolicyAdjustment]:
        """
        Evaluate a single policy and return adjustment if applicable.
        
        Args:
            policy: Policy to evaluate
            request: Policy evaluation request
            
        Returns:
            Policy adjustment if policy applies, None otherwise
        """
        conditions_met = []
        
        # Check rail preference
        rail_preference_boost = 0.0
        if policy.prefer_rail:
            # Check if preferred rail is available
            available_rails = [rail.get("rail_type") for rail in request.rail_candidates]
            if policy.prefer_rail.value in available_rails:
                rail_preference_boost = 0.2  # 20% boost for preferred rail
                conditions_met.append(f"preferred_rail_{policy.prefer_rail.value}")
        
        # Check loyalty rebate
        reward_bonus_delta = 0.0
        if policy.loyalty_rebate_pct > 0:
            # Convert percentage to decimal bonus
            reward_bonus_delta = policy.loyalty_rebate_pct / 100.0
            conditions_met.append(f"loyalty_rebate_{policy.loyalty_rebate_pct}%")
        
        # Check early payment discount
        cost_bps_delta = 0.0
        if policy.early_pay_discount_bps > 0:
            # Negative delta means cost reduction
            cost_bps_delta = -policy.early_pay_discount_bps
            conditions_met.append(f"early_pay_discount_{policy.early_pay_discount_bps}bps")
        
        # Only create adjustment if any conditions are met
        if conditions_met:
            adjustment_reason = self._generate_adjustment_reason(policy, conditions_met)
            
            return PolicyAdjustment(
                cost_bps_delta=cost_bps_delta,
                reward_bonus_delta=reward_bonus_delta,
                rail_preference_boost=rail_preference_boost,
                policy_id=policy.policy_id,
                adjustment_reason=adjustment_reason,
                conditions_met=conditions_met,
            )
        
        return None
    
    def _generate_adjustment_reason(self, policy: PolicyDSL, conditions_met: List[str]) -> str:
        """Generate human-readable adjustment reason."""
        reasons = []
        
        if "preferred_rail_" in " ".join(conditions_met):
            rail = policy.prefer_rail.value if policy.prefer_rail else "unknown"
            reasons.append(f"Policy favored {rail} due to rail preference")
        
        if "loyalty_rebate_" in " ".join(conditions_met):
            reasons.append(f"Applied {policy.loyalty_rebate_pct}% loyalty rebate")
        
        if "early_pay_discount_" in " ".join(conditions_met):
            reasons.append(f"Applied {policy.early_pay_discount_bps}bps early payment discount")
        
        return "; ".join(reasons)
    
    def _apply_adjustments(
        self, 
        original_scores: Dict[str, float], 
        adjustments: List[PolicyAdjustment]
    ) -> Dict[str, float]:
        """
        Apply policy adjustments to original scores.
        
        Args:
            original_scores: Original rail scores
            adjustments: Policy adjustments to apply
            
        Returns:
            Updated scores after applying adjustments
        """
        updated_scores = original_scores.copy()
        
        for adjustment in adjustments:
            # Apply rail preference boost
            if adjustment.rail_preference_boost > 0:
                # Boost all scores proportionally, but more for preferred rails
                for rail, score in updated_scores.items():
                    if "preferred_rail_" in " ".join(adjustment.conditions_met):
                        # Find which rail was preferred
                        preferred_rail = None
                        for condition in adjustment.conditions_met:
                            if condition.startswith("preferred_rail_"):
                                preferred_rail = condition.replace("preferred_rail_", "")
                                break
                        
                        if preferred_rail and rail.lower() == preferred_rail.lower():
                            # Boost preferred rail
                            updated_scores[rail] = min(1.0, score + adjustment.rail_preference_boost)
                        else:
                            # Slight penalty for non-preferred rails
                            updated_scores[rail] = max(0.0, score - (adjustment.rail_preference_boost * 0.1))
            
            # Apply cost adjustments (affects cost-sensitive rails more)
            if adjustment.cost_bps_delta != 0:
                # Cost adjustments affect ACH and debit more than credit
                for rail, score in updated_scores.items():
                    if rail.lower() in ["ach", "debit"]:
                        # Cost adjustments have more impact on cost-sensitive rails
                        cost_impact = adjustment.cost_bps_delta / 10000.0  # Convert bps to decimal
                        updated_scores[rail] = max(0.0, min(1.0, score + cost_impact))
            
            # Apply reward adjustments (affects reward-sensitive rails more)
            if adjustment.reward_bonus_delta != 0:
                # Reward adjustments affect credit cards more
                for rail, score in updated_scores.items():
                    if rail.lower() == "credit":
                        updated_scores[rail] = max(0.0, min(1.0, score + adjustment.reward_bonus_delta))
        
        return updated_scores
    
    def _determine_winner(self, scores: Dict[str, float]) -> Optional[str]:
        """Determine the winning rail based on scores."""
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _generate_policy_impact_summary(
        self, 
        adjustments: List[PolicyAdjustment], 
        applied_policies: List[str], 
        ignored_policies: List[str]
    ) -> str:
        """Generate a summary of policy impact."""
        if not adjustments:
            return "No policies applied - using default scoring"
        
        impact_parts = []
        
        if applied_policies:
            impact_parts.append(f"Applied {len(applied_policies)} policy(ies)")
        
        if ignored_policies:
            impact_parts.append(f"Ignored {len(ignored_policies)} policy(ies)")
        
        # Add specific adjustment details
        for adjustment in adjustments:
            if adjustment.cost_bps_delta != 0:
                impact_parts.append(f"Cost adjustment: {adjustment.cost_bps_delta:+.0f}bps")
            
            if adjustment.reward_bonus_delta != 0:
                impact_parts.append(f"Reward adjustment: {adjustment.reward_bonus_delta:+.1%}")
            
            if adjustment.rail_preference_boost > 0:
                impact_parts.append(f"Rail preference boost: {adjustment.rail_preference_boost:.1%}")
        
        return "; ".join(impact_parts)


# Global policy evaluator instance
_policy_evaluator: Optional[PolicyEvaluator] = None


def get_policy_evaluator() -> PolicyEvaluator:
    """Get the global policy evaluator instance."""
    global _policy_evaluator
    if _policy_evaluator is None:
        _policy_evaluator = PolicyEvaluator()
    return _policy_evaluator

