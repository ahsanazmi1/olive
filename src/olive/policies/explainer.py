"""
LLM explanation generator for policy applications.

This module provides explanations for policy decisions using a simple
deterministic approach (no external LLM dependency for demo purposes).
"""

import logging
from typing import Dict, List, Optional

from .models import PolicyAdjustment, PolicyDSL

logger = logging.getLogger(__name__)


class PolicyExplainer:
    """Policy explanation generator."""
    
    def __init__(self):
        self.source = "olive_policy_explainer"
    
    def explain_policy_application(
        self,
        applied_policies: List[str],
        adjustments: List[PolicyAdjustment],
        before_scores: Dict[str, float],
        after_scores: Dict[str, float],
        winner_rail: str
    ) -> str:
        """
        Generate an explanation for policy application.
        
        Args:
            applied_policies: List of applied policy IDs
            adjustments: Policy adjustments made
            before_scores: Rail scores before policy application
            after_scores: Rail scores after policy application
            winner_rail: Winning rail after policy application
            
        Returns:
            Human-readable explanation of policy application
        """
        if not applied_policies:
            return "No policies applied - using default scoring without policy adjustments."
        
        explanation_parts = []
        
        # Explain which policies were applied
        explanation_parts.append(f"Applied {len(applied_policies)} policy(ies): {', '.join(applied_policies)}")
        
        # Explain specific adjustments
        for adjustment in adjustments:
            if adjustment.cost_bps_delta != 0:
                if adjustment.cost_bps_delta < 0:
                    explanation_parts.append(f"Reduced costs by {abs(adjustment.cost_bps_delta):.0f} basis points")
                else:
                    explanation_parts.append(f"Increased costs by {adjustment.cost_bps_delta:.0f} basis points")
            
            if adjustment.reward_bonus_delta != 0:
                if adjustment.reward_bonus_delta > 0:
                    explanation_parts.append(f"Increased rewards by {adjustment.reward_bonus_delta:.1%}")
                else:
                    explanation_parts.append(f"Decreased rewards by {abs(adjustment.reward_bonus_delta):.1%}")
            
            if adjustment.rail_preference_boost > 0:
                explanation_parts.append(f"Applied rail preference boost of {adjustment.rail_preference_boost:.1%}")
        
        # Explain the impact on rail selection
        rail_changes = self._analyze_rail_changes(before_scores, after_scores)
        
        if rail_changes:
            explanation_parts.append(f"Policy adjustments changed rail rankings: {rail_changes}")
        
        # Explain the final winner
        explanation_parts.append(f"Policy favored {winner_rail} as the optimal rail choice")
        
        return ". ".join(explanation_parts) + "."
    
    def _analyze_rail_changes(
        self, 
        before_scores: Dict[str, float], 
        after_scores: Dict[str, float]
    ) -> str:
        """Analyze how rail rankings changed due to policy application."""
        if not before_scores or not after_scores:
            return ""
        
        # Sort rails by score (before and after)
        before_ranking = sorted(before_scores.items(), key=lambda x: x[1], reverse=True)
        after_ranking = sorted(after_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Check if rankings changed
        before_order = [rail for rail, _ in before_ranking]
        after_order = [rail for rail, _ in after_ranking]
        
        if before_order == after_order:
            return "no ranking changes"
        
        # Find the changes
        changes = []
        for i, (before_rail, after_rail) in enumerate(zip(before_order, after_order)):
            if before_rail != after_rail:
                changes.append(f"{before_rail}→{after_rail}")
        
        if changes:
            return "; ".join(changes)
        else:
            return "minor score adjustments"
    
    def explain_policy_reason(
        self,
        policy: PolicyDSL,
        adjustment: PolicyAdjustment
    ) -> str:
        """
        Explain the reasoning behind a specific policy adjustment.
        
        Args:
            policy: The policy that was applied
            adjustment: The adjustment made by the policy
            
        Returns:
            Explanation of the policy reasoning
        """
        reason_parts = []
        
        # Explain policy purpose
        if policy.prefer_rail:
            reason_parts.append(f"Policy {policy.policy_name} prefers {policy.prefer_rail.value} rail")
        
        if policy.loyalty_rebate_pct > 0:
            reason_parts.append(f"offers {policy.loyalty_rebate_pct}% loyalty rebate")
        
        if policy.early_pay_discount_bps > 0:
            reason_parts.append(f"provides {policy.early_pay_discount_bps}bps early payment discount")
        
        # Explain the specific adjustment
        if adjustment.cost_bps_delta < 0:
            reason_parts.append(f"reducing costs by {abs(adjustment.cost_bps_delta):.0f} basis points")
        elif adjustment.cost_bps_delta > 0:
            reason_parts.append(f"increasing costs by {adjustment.cost_bps_delta:.0f} basis points")
        
        if adjustment.reward_bonus_delta > 0:
            reason_parts.append(f"boosting rewards by {adjustment.reward_bonus_delta:.1%}")
        elif adjustment.reward_bonus_delta < 0:
            reason_parts.append(f"reducing rewards by {abs(adjustment.reward_bonus_delta):.1%}")
        
        if adjustment.rail_preference_boost > 0:
            reason_parts.append(f"giving preference boost of {adjustment.rail_preference_boost:.1%}")
        
        return " ".join(reason_parts) + "."


# Global policy explainer instance
_policy_explainer: Optional[PolicyExplainer] = None


def get_policy_explainer() -> PolicyExplainer:
    """Get the global policy explainer instance."""
    global _policy_explainer
    if _policy_explainer is None:
        _policy_explainer = PolicyExplainer()
    return _policy_explainer


def generate_policy_explanation(
    applied_policies: List[str],
    adjustments: List[PolicyAdjustment],
    before_scores: Dict[str, float],
    after_scores: Dict[str, float],
    winner_rail: str
) -> str:
    """
    Generate an explanation for policy application.
    
    Args:
        applied_policies: List of applied policy IDs
        adjustments: Policy adjustments made
        before_scores: Rail scores before policy application
        after_scores: Rail scores after policy application
        winner_rail: Winning rail after policy application
        
    Returns:
        Human-readable explanation of policy application
    """
    explainer = get_policy_explainer()
    return explainer.explain_policy_application(
        applied_policies=applied_policies,
        adjustments=adjustments,
        before_scores=before_scores,
        after_scores=after_scores,
        winner_rail=winner_rail
    )
