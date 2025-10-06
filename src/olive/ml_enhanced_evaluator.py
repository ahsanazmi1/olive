import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .policies.models import PolicyAdjustment, PolicyDSL, PolicyEvaluationRequest, PolicyEvaluationResponse, get_policy_storage
from .policies.evaluator import PolicyEvaluator
from .ml.policy_optimization import get_policy_optimizer, PolicyOptimizationFeatures
from .ml.incentive_effectiveness import get_effectiveness_predictor, IncentiveEffectivenessFeatures

logger = logging.getLogger(__name__)


class MLEnhancedPolicyEvaluator:
    """ML-enhanced policy evaluator with intelligent optimization."""

    def __init__(self, ml_weight: float = 0.7, use_ml: bool = True):
        self.ml_weight = ml_weight
        self.use_ml = use_ml
        self.traditional_evaluator = PolicyEvaluator()
        self.policy_optimizer = get_policy_optimizer()
        self.effectiveness_predictor = get_effectiveness_predictor()

    def evaluate_policies(
        self, 
        request: PolicyEvaluationRequest
    ) -> PolicyEvaluationResponse:
        """
        Evaluate policies with ML-enhanced optimization.
        
        Args:
            request: Policy evaluation request with transaction context
            
        Returns:
            ML-enhanced policy evaluation response
        """
        # First, run traditional policy evaluation
        traditional_response = self.traditional_evaluator.evaluate_policies(request)

        if not self.use_ml:
            return traditional_response

        # Apply ML enhancements
        ml_enhanced_response = self._apply_ml_enhancements(request, traditional_response)

        return ml_enhanced_response

    def _apply_ml_enhancements(
        self, 
        request: PolicyEvaluationRequest, 
        traditional_response: PolicyEvaluationResponse
    ) -> PolicyEvaluationResponse:
        """Apply ML enhancements to traditional policy evaluation."""
        try:
            # Get ML-optimized policy parameters
            optimization_features = self._extract_optimization_features(request)
            optimization_result = self.policy_optimizer.optimize_policy(optimization_features)

            # Predict incentive effectiveness
            effectiveness_features = self._extract_effectiveness_features(request)
            effectiveness_result = self.effectiveness_predictor.predict_effectiveness(effectiveness_features)

            # Create ML-enhanced adjustments
            ml_adjustments = self._create_ml_adjustments(
                traditional_response.adjustments,
                optimization_result,
                effectiveness_result
            )

            # Apply ML-enhanced adjustments to scores
            ml_updated_scores = self._apply_ml_adjustments(
                request.current_scores, 
                ml_adjustments,
                optimization_result,
                effectiveness_result
            )

            # Determine ML-enhanced winner
            ml_winner_rail = self._determine_ml_winner(ml_updated_scores)

            # Generate ML-enhanced policy impact
            ml_policy_impact = self._generate_ml_policy_impact(
                traditional_response.policy_impact,
                optimization_result,
                effectiveness_result,
                ml_adjustments
            )

            # Create ML-enhanced response
            return PolicyEvaluationResponse(
                adjustments=ml_adjustments,
                updated_scores=ml_updated_scores,
                applied_policies=traditional_response.applied_policies + ["ml_optimization", "ml_effectiveness"],
                ignored_policies=traditional_response.ignored_policies,
                trace_id=request.trace_id,
                winner_rail=ml_winner_rail,
                policy_impact=ml_policy_impact,
            )

        except Exception as e:
            logger.error(f"ML enhancement failed, falling back to traditional evaluation: {e}")
            return traditional_response

    def _extract_optimization_features(self, request: PolicyEvaluationRequest) -> PolicyOptimizationFeatures:
        """Extract features for policy optimization."""
        # Extract customer and transaction context
        customer_segment = "regular"  # Default, would come from customer data in real implementation
        transaction_frequency = 2.5  # Default, would come from customer data
        avg_transaction_amount = request.transaction_amount * 0.8  # Estimate
        loyalty_tier = 3  # Default, would come from customer data
        preferred_rail = request.rail_candidates[0].get("rail_type", "ACH") if request.rail_candidates else "ACH"
        
        # Extract timing information
        now = datetime.now()
        time_of_day = now.hour
        day_of_week = now.weekday()
        
        # Default values for ML features (would come from real data sources)
        seasonal_factor = 1.0
        competitive_pressure = 0.5
        cost_sensitivity = 0.5
        reward_preference = 0.5
        payment_method_age_days = 365
        account_age_days = 730
        previous_policy_success_rate = 0.7
        merchant_volume_tier = 3

        return PolicyOptimizationFeatures(
            transaction_amount=request.transaction_amount,
            merchant_id=request.merchant_id,
            customer_segment=customer_segment,
            transaction_frequency=transaction_frequency,
            avg_transaction_amount=avg_transaction_amount,
            loyalty_tier=loyalty_tier,
            preferred_rail=preferred_rail,
            channel=request.channel,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            seasonal_factor=seasonal_factor,
            competitive_pressure=competitive_pressure,
            cost_sensitivity=cost_sensitivity,
            reward_preference=reward_preference,
            payment_method_age_days=payment_method_age_days,
            account_age_days=account_age_days,
            previous_policy_success_rate=previous_policy_success_rate,
            merchant_volume_tier=merchant_volume_tier
        )

    def _extract_effectiveness_features(self, request: PolicyEvaluationRequest) -> IncentiveEffectivenessFeatures:
        """Extract features for incentive effectiveness prediction."""
        # Extract customer context
        customer_age_days = 730  # Default, would come from customer data
        customer_segment = "regular"  # Default
        loyalty_tier = 3  # Default
        transaction_frequency = 2.5  # Default
        avg_transaction_amount = request.transaction_amount * 0.8  # Estimate
        
        # Extract incentive context
        incentive_type = "cashback"  # Default
        incentive_value = request.transaction_amount * 0.02  # 2% cashback
        incentive_percentage = 2.0
        
        # Extract merchant and timing context
        merchant_category = "retail"  # Default
        now = datetime.now()
        time_of_day = now.hour
        day_of_week = now.weekday()
        
        # Default values for ML features
        seasonal_factor = 1.0
        competitive_pressure = 0.5
        customer_engagement_score = 0.5
        previous_incentive_response_rate = 0.7
        merchant_trust_score = 0.8
        economic_indicator = 0.6

        return IncentiveEffectivenessFeatures(
            transaction_amount=request.transaction_amount,
            customer_age_days=customer_age_days,
            customer_segment=customer_segment,
            loyalty_tier=loyalty_tier,
            transaction_frequency=transaction_frequency,
            avg_transaction_amount=avg_transaction_amount,
            incentive_type=incentive_type,
            incentive_value=incentive_value,
            incentive_percentage=incentive_percentage,
            merchant_category=merchant_category,
            channel=request.channel,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            seasonal_factor=seasonal_factor,
            competitive_pressure=competitive_pressure,
            customer_engagement_score=customer_engagement_score,
            previous_incentive_response_rate=previous_incentive_response_rate,
            merchant_trust_score=merchant_trust_score,
            economic_indicator=economic_indicator
        )

    def _create_ml_adjustments(
        self, 
        traditional_adjustments: List[PolicyAdjustment],
        optimization_result,
        effectiveness_result
    ) -> List[PolicyAdjustment]:
        """Create ML-enhanced policy adjustments."""
        ml_adjustments = traditional_adjustments.copy()

        # Create ML optimization adjustment
        ml_optimization_adjustment = PolicyAdjustment(
            cost_bps_delta=-optimization_result.optimal_early_pay_discount_bps,
            reward_bonus_delta=optimization_result.optimal_loyalty_rebate_pct / 100.0,
            rail_preference_boost=0.15 if effectiveness_result.effectiveness_score > 0.7 else 0.05,
            policy_id="ml_optimization",
            adjustment_reason=f"ML optimization: {optimization_result.optimal_loyalty_rebate_pct:.1f}% rebate, {optimization_result.optimal_early_pay_discount_bps:.0f}bps discount, prefer {optimization_result.optimal_rail_preference}",
            conditions_met=[
                f"ml_optimization_confidence_{optimization_result.confidence_score:.2f}",
                f"effectiveness_score_{effectiveness_result.effectiveness_score:.2f}",
                f"expected_conversion_boost_{optimization_result.expected_conversion_boost:.2f}"
            ]
        )
        ml_adjustments.append(ml_optimization_adjustment)

        # Create ML effectiveness adjustment
        ml_effectiveness_adjustment = PolicyAdjustment(
            cost_bps_delta=0.0,
            reward_bonus_delta=effectiveness_result.effectiveness_score * 0.1,  # Boost based on effectiveness
            rail_preference_boost=0.1 if effectiveness_result.conversion_probability > 0.8 else 0.0,
            policy_id="ml_effectiveness",
            adjustment_reason=f"ML effectiveness prediction: {effectiveness_result.effectiveness_score:.2f} effectiveness, {effectiveness_result.conversion_probability:.2f} conversion probability",
            conditions_met=[
                f"effectiveness_confidence_{effectiveness_result.confidence_score:.2f}",
                f"conversion_probability_{effectiveness_result.conversion_probability:.2f}",
                f"engagement_likelihood_{effectiveness_result.engagement_likelihood:.2f}"
            ]
        )
        ml_adjustments.append(ml_effectiveness_adjustment)

        return ml_adjustments

    def _apply_ml_adjustments(
        self, 
        original_scores: Dict[str, float], 
        ml_adjustments: List[PolicyAdjustment],
        optimization_result,
        effectiveness_result
    ) -> Dict[str, float]:
        """Apply ML-enhanced adjustments to original scores."""
        updated_scores = original_scores.copy()

        # Apply traditional adjustments
        for adjustment in ml_adjustments:
            # Apply rail preference boost
            if adjustment.rail_preference_boost > 0:
                for rail, score in updated_scores.items():
                    if "ml_optimization" in adjustment.policy_id:
                        # ML optimization boost for optimal rail
                        if rail.lower() == optimization_result.optimal_rail_preference.lower():
                            updated_scores[rail] = min(1.0, score + adjustment.rail_preference_boost)
                        else:
                            # Slight penalty for non-optimal rails
                            updated_scores[rail] = max(0.0, score - (adjustment.rail_preference_boost * 0.2))
                    else:
                        # Standard rail preference boost
                        if "preferred_rail_" in " ".join(adjustment.conditions_met):
                            preferred_rail = None
                            for condition in adjustment.conditions_met:
                                if condition.startswith("preferred_rail_"):
                                    preferred_rail = condition.replace("preferred_rail_", "")
                                    break
                            
                            if preferred_rail and rail.lower() == preferred_rail.lower():
                                updated_scores[rail] = min(1.0, score + adjustment.rail_preference_boost)
                            else:
                                updated_scores[rail] = max(0.0, score - (adjustment.rail_preference_boost * 0.1))
            
            # Apply cost adjustments
            if adjustment.cost_bps_delta != 0:
                for rail, score in updated_scores.items():
                    if rail.lower() in ["ach", "debit"]:
                        cost_impact = adjustment.cost_bps_delta / 10000.0
                        updated_scores[rail] = max(0.0, min(1.0, score + cost_impact))
            
            # Apply reward adjustments
            if adjustment.reward_bonus_delta != 0:
                for rail, score in updated_scores.items():
                    if rail.lower() == "credit":
                        updated_scores[rail] = max(0.0, min(1.0, score + adjustment.reward_bonus_delta))

        # Apply ML-specific enhancements
        # Boost scores based on ML predictions
        ml_boost_factor = (optimization_result.confidence_score + effectiveness_result.confidence_score) / 2.0
        
        for rail, score in updated_scores.items():
            # Apply confidence-based boost
            confidence_boost = ml_boost_factor * 0.05  # Up to 5% boost based on ML confidence
            updated_scores[rail] = min(1.0, score + confidence_boost)

        return updated_scores

    def _determine_ml_winner(self, scores: Dict[str, float]) -> Optional[str]:
        """Determine the ML-enhanced winning rail."""
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def _generate_ml_policy_impact(
        self, 
        traditional_impact: str,
        optimization_result,
        effectiveness_result,
        ml_adjustments: List[PolicyAdjustment]
    ) -> str:
        """Generate ML-enhanced policy impact summary."""
        ml_impact_parts = [traditional_impact]
        
        # Add ML optimization impact
        ml_impact_parts.append(f"ML optimization: {optimization_result.optimal_loyalty_rebate_pct:.1f}% rebate, {optimization_result.optimal_early_pay_discount_bps:.0f}bps discount")
        ml_impact_parts.append(f"Expected conversion boost: {optimization_result.expected_conversion_boost:.1%}")
        ml_impact_parts.append(f"Expected revenue impact: ${optimization_result.expected_revenue_impact:.2f}")
        
        # Add ML effectiveness impact
        ml_impact_parts.append(f"Incentive effectiveness: {effectiveness_result.effectiveness_score:.2f}")
        ml_impact_parts.append(f"Conversion probability: {effectiveness_result.conversion_probability:.2f}")
        ml_impact_parts.append(f"Engagement likelihood: {effectiveness_result.engagement_likelihood:.2f}")
        
        # Add ML confidence
        avg_confidence = (optimization_result.confidence_score + effectiveness_result.confidence_score) / 2.0
        ml_impact_parts.append(f"ML confidence: {avg_confidence:.2f}")
        
        return "; ".join(ml_impact_parts)


# Global ML-enhanced evaluator instance
_ml_evaluator: Optional[MLEnhancedPolicyEvaluator] = None


def get_ml_enhanced_evaluator() -> MLEnhancedPolicyEvaluator:
    """Get the global ML-enhanced policy evaluator instance."""
    global _ml_evaluator
    if _ml_evaluator is None:
        _ml_evaluator = MLEnhancedPolicyEvaluator()
    return _ml_evaluator
