"""
CloudEvent emission for policy application events.

This module handles emission of ocn.olive.policy_applied.v1 events.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import PolicyAdjustment, PolicyAppliedEvent, PolicyEvaluationResponse

logger = logging.getLogger(__name__)


class PolicyEventEmitter:
    """CloudEvent emitter for policy application events."""
    
    def __init__(self):
        self.source = "olive"
        self.event_type = "ocn.olive.policy_applied.v1"
    
    def emit_policy_applied_event(
        self,
        evaluation_response: PolicyEvaluationResponse,
        merchant_id: str,
        transaction_amount: float,
        before_scores: Dict[str, float],
        llm_explanation: str,
        evaluation_duration_ms: float
    ) -> PolicyAppliedEvent:
        """
        Emit a policy applied CloudEvent.
        
        Args:
            evaluation_response: Policy evaluation response
            merchant_id: Merchant identifier
            transaction_amount: Transaction amount
            before_scores: Rail scores before policy application
            llm_explanation: LLM-generated explanation
            evaluation_duration_ms: Evaluation duration in milliseconds
            
        Returns:
            PolicyAppliedEvent for emission
        """
        event_id = str(uuid.uuid4())
        
        # Generate policy impact summary
        policy_impact_summary = self._generate_policy_impact_summary(
            evaluation_response.applied_policies,
            evaluation_response.ignored_policies,
            evaluation_response.adjustments
        )
        
        event = PolicyAppliedEvent(
            event_id=event_id,
            event_type=self.event_type,
            source=self.source,
            timestamp=datetime.now(),
            trace_id=evaluation_response.trace_id,
            merchant_id=merchant_id,
            transaction_amount=transaction_amount,
            applied_policies=evaluation_response.applied_policies,
            before_scores=before_scores,
            after_scores=evaluation_response.updated_scores,
            adjustments=evaluation_response.adjustments,
            winner_rail=evaluation_response.winner_rail or "unknown",
            policy_impact_summary=policy_impact_summary,
            llm_explanation=llm_explanation,
            evaluation_duration_ms=evaluation_duration_ms,
            policy_count=len(evaluation_response.applied_policies) + len(evaluation_response.ignored_policies),
        )
        
        # Log the event
        logger.info(
            f"Policy applied event emitted",
            extra={
                "event_id": event_id,
                "trace_id": evaluation_response.trace_id,
                "merchant_id": merchant_id,
                "applied_policies": len(evaluation_response.applied_policies),
                "winner_rail": evaluation_response.winner_rail,
            }
        )
        
        return event
    
    def _generate_policy_impact_summary(
        self,
        applied_policies: List[str],
        ignored_policies: List[str],
        adjustments: List[PolicyAdjustment]
    ) -> str:
        """Generate a summary of policy impact."""
        if not applied_policies:
            return "No policies applied - using default scoring"
        
        summary_parts = []
        
        # Count policies
        summary_parts.append(f"Applied {len(applied_policies)} policy(ies)")
        
        if ignored_policies:
            summary_parts.append(f"Ignored {len(ignored_policies)} policy(ies)")
        
        # Add specific impacts
        cost_adjustments = [adj for adj in adjustments if adj.cost_bps_delta != 0]
        reward_adjustments = [adj for adj in adjustments if adj.reward_bonus_delta != 0]
        rail_adjustments = [adj for adj in adjustments if adj.rail_preference_boost > 0]
        
        if cost_adjustments:
            total_cost_delta = sum(adj.cost_bps_delta for adj in cost_adjustments)
            summary_parts.append(f"Total cost adjustment: {total_cost_delta:+.0f}bps")
        
        if reward_adjustments:
            total_reward_delta = sum(adj.reward_bonus_delta for adj in reward_adjustments)
            summary_parts.append(f"Total reward adjustment: {total_reward_delta:+.1%}")
        
        if rail_adjustments:
            summary_parts.append(f"Rail preference boosts applied")
        
        return "; ".join(summary_parts)
    
    def format_cloud_event(self, event: PolicyAppliedEvent) -> Dict[str, Any]:
        """
        Format the event as a CloudEvent.
        
        Args:
            event: Policy applied event
            
        Returns:
            CloudEvent formatted dictionary
        """
        return {
            "specversion": "1.0",
            "type": event.event_type,
            "source": event.source,
            "id": event.event_id,
            "time": event.timestamp.isoformat() + "Z",
            "datacontenttype": "application/json",
            "data": event.dict(),
        }


# Global event emitter instance
_event_emitter: Optional[PolicyEventEmitter] = None


def get_policy_event_emitter() -> PolicyEventEmitter:
    """Get the global policy event emitter instance."""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = PolicyEventEmitter()
    return _event_emitter


async def emit_policy_applied_event(
    evaluation_response: PolicyEvaluationResponse,
    merchant_id: str,
    transaction_amount: float,
    before_scores: Dict[str, float],
    llm_explanation: str,
    evaluation_duration_ms: float
) -> PolicyAppliedEvent:
    """
    Emit a policy applied CloudEvent.
    
    Args:
        evaluation_response: Policy evaluation response
        merchant_id: Merchant identifier
        transaction_amount: Transaction amount
        before_scores: Rail scores before policy application
        llm_explanation: LLM-generated explanation
        evaluation_duration_ms: Evaluation duration in milliseconds
        
    Returns:
        PolicyAppliedEvent that was emitted
    """
    emitter = get_policy_event_emitter()
    return emitter.emit_policy_applied_event(
        evaluation_response=evaluation_response,
        merchant_id=merchant_id,
        transaction_amount=transaction_amount,
        before_scores=before_scores,
        llm_explanation=llm_explanation,
        evaluation_duration_ms=evaluation_duration_ms,
    )
