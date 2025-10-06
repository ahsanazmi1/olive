"""
Enhanced Policy Evaluator for Olive Phase 4

This module provides advanced policy evaluation with support for:
- Conditional policy application
- Merchant routing rules (rebates, early-pay, loyalty incentives)
- Tax validation
- Policy enforcement during negotiation
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    PolicyDSL,
    PolicyCondition,
    PolicyAction,
    PolicyAdjustment,
    PolicyEvaluationRequest,
    PolicyEvaluationResponse,
    PolicyEnforcementRequest,
    PolicyConditionType,
    PolicyActionType,
    get_policy_storage
)

logger = logging.getLogger(__name__)


class EnhancedPolicyEvaluator:
    """Enhanced policy evaluator with advanced DSL support."""
    
    def __init__(self):
        self.storage = get_policy_storage()
        self.condition_evaluators = {
            PolicyConditionType.AMOUNT_RANGE: self._evaluate_amount_range,
            PolicyConditionType.TIME_WINDOW: self._evaluate_time_window,
            PolicyConditionType.MERCHANT_CATEGORY: self._evaluate_merchant_category,
            PolicyConditionType.CUSTOMER_SEGMENT: self._evaluate_customer_segment,
            PolicyConditionType.PAYMENT_FREQUENCY: self._evaluate_payment_frequency,
            PolicyConditionType.RISK_LEVEL: self._evaluate_risk_level,
        }
        
        self.action_appliers = {
            PolicyActionType.RAIL_PREFERENCE: self._apply_rail_preference,
            PolicyActionType.REBATE_APPLICATION: self._apply_rebate_application,
            PolicyActionType.DISCOUNT_APPLICATION: self._apply_discount_application,
            PolicyActionType.LOYALTY_BOOST: self._apply_loyalty_boost,
            PolicyActionType.TAX_VALIDATION: self._apply_tax_validation,
            PolicyActionType.EARLY_PAY_INCENTIVE: self._apply_early_pay_incentive,
        }

    def evaluate_policies_enhanced(
        self, 
        request: PolicyEvaluationRequest
    ) -> PolicyEvaluationResponse:
        """
        Evaluate policies with enhanced DSL support.
        
        Args:
            request: Policy evaluation request
            
        Returns:
            Enhanced policy evaluation response
        """
        start_time = datetime.now()
        
        logger.info(
            f"Evaluating enhanced policies for merchant {request.merchant_id}",
            extra={
                "merchant_id": request.merchant_id,
                "transaction_amount": request.transaction_amount,
                "trace_id": request.trace_id,
            }
        )
        
        # Get applicable policies
        applicable_policies = self._get_applicable_policies(request)
        
        # Evaluate each policy
        adjustments = []
        updated_scores = request.current_scores.copy()
        
        for policy in applicable_policies:
            policy_adjustments = self._evaluate_policy(request, policy)
            adjustments.extend(policy_adjustments)
            
            # Apply adjustments to scores
            for adjustment in policy_adjustments:
                updated_scores = self._apply_adjustment_to_scores(updated_scores, adjustment)
        
        # Create response
        response = PolicyEvaluationResponse(
            adjustments=adjustments,
            updated_scores=updated_scores,
            evaluation_metadata={
                "total_policies_evaluated": len(applicable_policies),
                "total_adjustments": len(adjustments),
                "evaluation_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "trace_id": request.trace_id,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Enhanced policy evaluation completed in {(datetime.now() - start_time).total_seconds() * 1000:.2f}ms")
        
        return response

    def enforce_policies_during_negotiation(
        self, 
        request: PolicyEnforcementRequest
    ) -> Dict[str, Any]:
        """
        Enforce policies during Orca/Opal negotiation.
        
        Args:
            request: Policy enforcement request
            
        Returns:
            Enforcement results with overrides and constraints
        """
        logger.info(f"Enforcing policies during negotiation for merchant {request.merchant_id}")
        
        # Get merchant policies
        merchant_policies = self.storage.get_policies_by_merchant(request.merchant_id)
        
        enforcement_results = {
            "enforcement_mode": request.enforcement_mode,
            "policy_constraints": [],
            "score_overrides": {},
            "rail_restrictions": [],
            "enforcement_actions": [],
            "trace_id": request.trace_id,
            "enforcement_timestamp": datetime.now().isoformat()
        }
        
        # Apply each policy based on enforcement mode
        for policy in merchant_policies:
            if not policy.enabled:
                continue
                
            # Check if policy applies to this negotiation context
            if self._policy_applies_to_context(policy, request.negotiation_context):
                enforcement_result = self._enforce_policy(policy, request.negotiation_context)
                
                if enforcement_result:
                    enforcement_results["policy_constraints"].append(enforcement_result)
                    
                    # Apply score overrides if in mandatory mode
                    if request.enforcement_mode == "mandatory":
                        enforcement_results["score_overrides"].update(enforcement_result.get("score_overrides", {}))
                    
                    # Add rail restrictions
                    if "rail_restrictions" in enforcement_result:
                        enforcement_results["rail_restrictions"].extend(enforcement_result["rail_restrictions"])
                    
                    # Add enforcement actions
                    enforcement_results["enforcement_actions"].append({
                        "policy_id": policy.policy_id,
                        "policy_name": policy.policy_name,
                        "action": enforcement_result.get("action", "constraint_applied"),
                        "description": enforcement_result.get("description", "")
                    })
        
        logger.info(f"Policy enforcement completed with {len(enforcement_results['policy_constraints'])} constraints")
        
        return enforcement_results

    def _get_applicable_policies(self, request: PolicyEvaluationRequest) -> List[PolicyDSL]:
        """Get policies that apply to the given request."""
        all_policies = self.storage.get_policies_by_merchant(request.merchant_id)
        
        applicable = []
        for policy in all_policies:
            if not policy.enabled:
                continue
                
            # Check if policy is effective
            now = datetime.now()
            if policy.effective_from > now or (policy.effective_until and policy.effective_until < now):
                continue
            
            # Check if policy conditions are met
            if self._policy_conditions_met(policy, request):
                applicable.append(policy)
        
        # Sort by priority (higher priority first)
        applicable.sort(key=lambda p: p.priority, reverse=True)
        
        return applicable

    def _policy_conditions_met(self, policy: PolicyDSL, request: PolicyEvaluationRequest) -> bool:
        """Check if all policy conditions are met."""
        if not policy.conditions:
            return True  # No conditions means always applicable
        
        for condition in policy.conditions:
            if not self._evaluate_condition(condition, request):
                return False
        
        return True

    def _evaluate_condition(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate a single policy condition."""
        evaluator = self.condition_evaluators.get(condition.condition_type)
        if not evaluator:
            logger.warning(f"No evaluator for condition type: {condition.condition_type}")
            return False
        
        try:
            return evaluator(condition, request)
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.condition_type}: {e}")
            return False

    def _evaluate_amount_range(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate amount range condition."""
        amount = request.transaction_amount
        
        if condition.operator == "between":
            min_val, max_val = condition.value
            return min_val <= amount <= max_val
        elif condition.operator == "gte":
            return amount >= condition.value
        elif condition.operator == "lte":
            return amount <= condition.value
        elif condition.operator == "gt":
            return amount > condition.value
        elif condition.operator == "lt":
            return amount < condition.value
        elif condition.operator == "eq":
            return amount == condition.value
        
        return False

    def _evaluate_time_window(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate time window condition."""
        # This would typically check against transaction timestamp or business hours
        # For now, we'll implement a simple time-of-day check
        now = datetime.now()
        current_hour = now.hour
        
        if condition.operator == "between":
            start_hour, end_hour = condition.value
            return start_hour <= current_hour <= end_hour
        
        return True  # Default to true for time conditions

    def _evaluate_merchant_category(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate merchant category condition."""
        # This would check against MCC or merchant category
        # For now, we'll return True as a placeholder
        return True

    def _evaluate_customer_segment(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate customer segment condition."""
        # This would check against customer segmentation data
        # For now, we'll return True as a placeholder
        return True

    def _evaluate_payment_frequency(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate payment frequency condition."""
        # This would check against historical payment frequency
        # For now, we'll return True as a placeholder
        return True

    def _evaluate_risk_level(self, condition: PolicyCondition, request: PolicyEvaluationRequest) -> bool:
        """Evaluate risk level condition."""
        # This would check against risk assessment data
        # For now, we'll return True as a placeholder
        return True

    def _evaluate_policy(self, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Evaluate a single policy and return adjustments."""
        adjustments = []
        
        # Apply legacy policy rules for backward compatibility
        if policy.prefer_rail:
            adjustment = PolicyAdjustment(
                policy_id=policy.policy_id,
                adjustment_type="rail_preference",
                rail_type=policy.prefer_rail.value,
                adjustment_value=policy.priority * 0.1,  # Convert priority to adjustment
                description=f"Rail preference from policy: {policy.policy_name}"
            )
            adjustments.append(adjustment)
        
        # Apply enhanced policy actions
        for action in policy.actions:
            action_applier = self.action_appliers.get(action.action_type)
            if action_applier:
                try:
                    action_adjustments = action_applier(action, request, policy)
                    adjustments.extend(action_adjustments)
                except Exception as e:
                    logger.error(f"Error applying action {action.action_type}: {e}")
        
        # Apply merchant routing rules
        routing_adjustments = self._apply_merchant_routing_rules(policy, request)
        adjustments.extend(routing_adjustments)
        
        return adjustments

    def _apply_merchant_routing_rules(self, policy: PolicyDSL, request: PolicyEvaluationRequest) -> List[PolicyAdjustment]:
        """Apply merchant routing rules (rebates, early-pay, loyalty incentives)."""
        adjustments = []
        
        # Apply rebate rules
        for rail_type, rebate_pct in policy.rebate_rules.items():
            adjustment = PolicyAdjustment(
                policy_id=policy.policy_id,
                adjustment_type="rebate_application",
                rail_type=rail_type,
                adjustment_value=rebate_pct,
                description=f"Rebate rule: {rebate_pct}% for {rail_type}"
            )
            adjustments.append(adjustment)
        
        # Apply early pay rules
        for rail_type, discount_bps in policy.early_pay_rules.items():
            adjustment = PolicyAdjustment(
                policy_id=policy.policy_id,
                adjustment_type="early_pay_incentive",
                rail_type=rail_type,
                adjustment_value=discount_bps / 10000.0,  # Convert basis points to decimal
                description=f"Early pay discount: {discount_bps}bps for {rail_type}"
            )
            adjustments.append(adjustment)
        
        # Apply loyalty incentives
        for loyalty_type, incentive_value in policy.loyalty_incentives.items():
            adjustment = PolicyAdjustment(
                policy_id=policy.policy_id,
                adjustment_type="loyalty_boost",
                rail_type="all",  # Loyalty incentives typically apply to all rails
                adjustment_value=incentive_value,
                description=f"Loyalty incentive: {incentive_value} for {loyalty_type}"
            )
            adjustments.append(adjustment)
        
        return adjustments

    def _apply_rail_preference(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply rail preference action."""
        rail_type = action.parameters.get("rail_type", "ACH")
        boost_value = action.parameters.get("boost_value", 0.1) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="rail_preference",
            rail_type=rail_type,
            adjustment_value=boost_value,
            description=f"Rail preference: {rail_type} (boost: {boost_value:.3f})"
        )
        
        return [adjustment]

    def _apply_rebate_application(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply rebate application action."""
        rail_type = action.parameters.get("rail_type", "all")
        rebate_pct = action.parameters.get("rebate_pct", 0.0) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="rebate_application",
            rail_type=rail_type,
            adjustment_value=rebate_pct,
            description=f"Rebate application: {rebate_pct:.2f}% for {rail_type}"
        )
        
        return [adjustment]

    def _apply_discount_application(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply discount application action."""
        rail_type = action.parameters.get("rail_type", "all")
        discount_bps = action.parameters.get("discount_bps", 0.0) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="discount_application",
            rail_type=rail_type,
            adjustment_value=discount_bps / 10000.0,
            description=f"Discount application: {discount_bps}bps for {rail_type}"
        )
        
        return [adjustment]

    def _apply_loyalty_boost(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply loyalty boost action."""
        loyalty_type = action.parameters.get("loyalty_type", "general")
        boost_value = action.parameters.get("boost_value", 0.1) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="loyalty_boost",
            rail_type="all",
            adjustment_value=boost_value,
            description=f"Loyalty boost: {boost_value:.3f} for {loyalty_type}"
        )
        
        return [adjustment]

    def _apply_tax_validation(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply tax validation action."""
        validation_type = action.parameters.get("validation_type", "standard")
        validation_weight = action.parameters.get("validation_weight", 0.05) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="tax_validation",
            rail_type="all",
            adjustment_value=validation_weight,
            description=f"Tax validation: {validation_type} (weight: {validation_weight:.3f})"
        )
        
        return [adjustment]

    def _apply_early_pay_incentive(self, action: PolicyAction, request: PolicyEvaluationRequest, policy: PolicyDSL) -> List[PolicyAdjustment]:
        """Apply early pay incentive action."""
        rail_type = action.parameters.get("rail_type", "all")
        incentive_bps = action.parameters.get("incentive_bps", 0.0) * action.weight
        
        adjustment = PolicyAdjustment(
            policy_id=policy.policy_id,
            adjustment_type="early_pay_incentive",
            rail_type=rail_type,
            adjustment_value=incentive_bps / 10000.0,
            description=f"Early pay incentive: {incentive_bps}bps for {rail_type}"
        )
        
        return [adjustment]

    def _apply_adjustment_to_scores(self, scores: Dict[str, float], adjustment: PolicyAdjustment) -> Dict[str, float]:
        """Apply a policy adjustment to rail scores."""
        updated_scores = scores.copy()
        
        if adjustment.rail_type == "all":
            # Apply to all rails
            for rail_type in updated_scores:
                updated_scores[rail_type] += adjustment.adjustment_value
        elif adjustment.rail_type in updated_scores:
            # Apply to specific rail
            updated_scores[adjustment.rail_type] += adjustment.adjustment_value
        
        return updated_scores

    def _policy_applies_to_context(self, policy: PolicyDSL, context: Dict[str, Any]) -> bool:
        """Check if policy applies to negotiation context."""
        # This would check policy conditions against negotiation context
        # For now, we'll return True as a placeholder
        return True

    def _enforce_policy(self, policy: PolicyDSL, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enforce a single policy during negotiation."""
        if policy.enforcement_mode == "advisory":
            return {
                "action": "advisory_constraint",
                "description": f"Advisory: {policy.description}",
                "policy_id": policy.policy_id
            }
        elif policy.enforcement_mode == "mandatory":
            return {
                "action": "mandatory_constraint",
                "description": f"Mandatory: {policy.description}",
                "score_overrides": self._generate_score_overrides(policy),
                "rail_restrictions": self._generate_rail_restrictions(policy),
                "policy_id": policy.policy_id
            }
        elif policy.enforcement_mode == "override":
            return {
                "action": "override_constraint",
                "description": f"Override: {policy.description}",
                "score_overrides": self._generate_score_overrides(policy),
                "policy_id": policy.policy_id
            }
        
        return None

    def _generate_score_overrides(self, policy: PolicyDSL) -> Dict[str, float]:
        """Generate score overrides for mandatory/override policies."""
        overrides = {}
        
        if policy.prefer_rail:
            # Boost preferred rail significantly
            overrides[policy.prefer_rail.value] = 1.0
        
        return overrides

    def _generate_rail_restrictions(self, policy: PolicyDSL) -> List[str]:
        """Generate rail restrictions for mandatory policies."""
        restrictions = []
        
        # If policy has specific rail preferences, restrict others
        if policy.prefer_rail:
            all_rails = ["ACH", "debit", "credit", "bnpl", "stablecoin", "prepaid"]
            restrictions = [rail for rail in all_rails if rail != policy.prefer_rail.value]
        
        return restrictions


# Global enhanced evaluator instance
_enhanced_evaluator: Optional[EnhancedPolicyEvaluator] = None


def get_enhanced_evaluator() -> EnhancedPolicyEvaluator:
    """Get the global enhanced policy evaluator instance."""
    global _enhanced_evaluator
    if _enhanced_evaluator is None:
        _enhanced_evaluator = EnhancedPolicyEvaluator()
    return _enhanced_evaluator








