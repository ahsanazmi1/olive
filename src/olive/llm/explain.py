"""
LLM explanation generation for Olive loyalty decisions.

Provides Azure OpenAI integration for generating human-readable explanations 
of loyalty policy applications and incentive recommendations.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LoyaltyExplanationRequest(BaseModel):
    """Request for loyalty explanation generation."""
    
    applied_policies: List[str] = Field(..., description="List of applied policy IDs")
    policy_adjustments: List[Dict[str, Any]] = Field(..., description="Policy adjustments made")
    before_scores: Dict[str, float] = Field(..., description="Rail scores before policy application")
    after_scores: Dict[str, float] = Field(..., description="Rail scores after policy application")
    winner_rail: str = Field(..., description="Winning rail after policy application")
    customer_context: Dict[str, Any] = Field(..., description="Customer context and features")
    loyalty_offers: List[Dict[str, Any]] = Field(..., description="Available loyalty offers")
    recommended_offer: Optional[str] = Field(None, description="Recommended loyalty offer")
    transaction_context: Dict[str, Any] = Field(..., description="Transaction context")


class LoyaltyExplanationResponse(BaseModel):
    """Response from loyalty explanation generation."""
    
    explanation: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., description="Explanation confidence", ge=0.0, le=1.0)
    key_factors: List[str] = Field(..., description="Key factors in decision")
    policy_impact: Dict[str, Any] = Field(..., description="Policy impact analysis")
    loyalty_benefits: Dict[str, Any] = Field(..., description="Loyalty benefits breakdown")
    recommendations: List[str] = Field(..., description="Recommendations for customer")
    model_provenance: Dict[str, str] = Field(..., description="Model provenance information")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    tokens_used: int = Field(..., description="Tokens used in generation")


class OliveLLMExplainer:
    """Azure OpenAI-based loyalty explanation generator."""
    
    def __init__(self):
        self.client = None
        self.is_available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available")
            return
        
        # Get configuration from environment
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "olive-llm")
        
        if endpoint and api_key:
            try:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                self.is_available = True
                logger.info(f"✅ Azure OpenAI client initialized with deployment: {deployment}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
                self.is_available = False
        else:
            logger.warning("Azure OpenAI not configured - missing endpoint or API key")
    
    def is_configured(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return self.is_available and self.client is not None
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "status": "configured" if self.is_configured() else "not_configured",
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "olive-llm"),
            "api_key": "***" if os.getenv("AZURE_OPENAI_API_KEY") else None,
        }
    
    def explain_loyalty_decision(self, request: LoyaltyExplanationRequest) -> Optional[LoyaltyExplanationResponse]:
        """Generate explanation for loyalty decision."""
        if not self.is_configured():
            logger.warning("Azure OpenAI not configured, using fallback explanation")
            return self._generate_fallback_explanation(request)
        
        start_time = datetime.now()
        
        try:
            # Build the prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_explanation_prompt(request)
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "olive-llm"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse the response
            explanation_data = self._parse_explanation_response(raw_response)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LoyaltyExplanationResponse(
                explanation=explanation_data.get("explanation", raw_response),
                confidence=explanation_data.get("confidence", 0.8),
                key_factors=explanation_data.get("key_factors", []),
                policy_impact=explanation_data.get("policy_impact", {}),
                loyalty_benefits=explanation_data.get("loyalty_benefits", {}),
                recommendations=explanation_data.get("recommendations", []),
                model_provenance={
                    "model_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "olive-llm"),
                    "provider": "azure_openai",
                    "status": "active",
                    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                },
                processing_time_ms=processing_time,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
        except Exception as e:
            logger.error(f"Error generating loyalty explanation: {e}")
            return self._generate_fallback_explanation(request)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for loyalty explanation generation."""
        return """You are an expert loyalty program analyst for Olive, the Open Loyalty Agent.
Your task is to generate clear, helpful explanations for loyalty policy applications and incentive recommendations.

CRITICAL REQUIREMENTS:
1. Explain WHY specific loyalty policies were applied
2. Highlight the impact on payment rail selection
3. Explain loyalty benefits and rewards in consumer-friendly terms
4. Provide actionable recommendations for maximizing loyalty value
5. Use clear, engaging language that builds customer excitement
6. Be specific about percentages, amounts, and benefits

JSON SCHEMA FOR RESPONSE:
{
  "explanation": "string - Clear explanation of loyalty policy application (max 200 words)",
  "confidence": "number - Confidence score between 0.0 and 1.0",
  "key_factors": ["string"] - List of 3-5 most important factors,
  "policy_impact": {
    "rail_changes": ["string"],
    "cost_savings": "string",
    "reward_boost": "string"
  },
  "loyalty_benefits": {
    "immediate_benefits": ["string"],
    "long_term_value": "string",
    "exclusive_offers": ["string"]
  },
  "recommendations": ["string"] - List of actionable recommendations
}

Always respond with valid JSON matching this schema."""
    
    def _build_explanation_prompt(self, request: LoyaltyExplanationRequest) -> str:
        """Build the user prompt for explanation generation."""
        return f"""Generate a clear explanation for this loyalty policy application:

APPLIED POLICIES: {', '.join(request.applied_policies)}
WINNER RAIL: {request.winner_rail}

POLICY ADJUSTMENTS:
{json.dumps(request.policy_adjustments, indent=2)}

RAIL SCORE CHANGES:
Before: {json.dumps(request.before_scores, indent=2)}
After: {json.dumps(request.after_scores, indent=2)}

CUSTOMER CONTEXT:
- Loyalty Tier: {request.customer_context.get('loyalty_tier', 'Standard')}
- Purchase Frequency: {request.customer_context.get('purchase_frequency', 'Unknown')}
- Average Order Value: ${request.customer_context.get('average_order_value', 0):.2f}
- Total Spent: ${request.customer_context.get('total_spent', 0):.2f}

AVAILABLE LOYALTY OFFERS:
{json.dumps(request.loyalty_offers, indent=2)}

RECOMMENDED OFFER: {request.recommended_offer or 'None'}

TRANSACTION CONTEXT:
- Amount: ${request.transaction_context.get('amount', 0):.2f}
- Category: {request.transaction_context.get('category', 'General')}
- Channel: {request.transaction_context.get('channel', 'Online')}

Please explain the loyalty policy application and its benefits for the customer."""
    
    def _parse_explanation_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse the LLM response."""
        try:
            # Try to extract JSON from the response
            if "```json" in raw_response:
                json_start = raw_response.find("```json") + 7
                json_end = raw_response.find("```", json_start)
                json_str = raw_response[json_start:json_end].strip()
            elif "{" in raw_response and "}" in raw_response:
                json_start = raw_response.find("{")
                json_end = raw_response.rfind("}") + 1
                json_str = raw_response[json_start:json_end]
            else:
                # Fallback to plain text
                return {
                    "explanation": raw_response,
                    "confidence": 0.7,
                    "key_factors": ["LLM-generated explanation"],
                    "policy_impact": {
                        "rail_changes": ["Unable to parse"],
                        "cost_savings": "Unable to parse",
                        "reward_boost": "Unable to parse"
                    },
                    "loyalty_benefits": {
                        "immediate_benefits": ["Unable to parse"],
                        "long_term_value": "Unable to parse",
                        "exclusive_offers": ["Unable to parse"]
                    },
                    "recommendations": ["Unable to parse"]
                }
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            # Fallback to plain text
            return {
                "explanation": raw_response,
                "confidence": 0.7,
                "key_factors": ["LLM-generated explanation"],
                "policy_impact": {
                    "rail_changes": ["Unable to parse"],
                    "cost_savings": "Unable to parse",
                    "reward_boost": "Unable to parse"
                },
                "loyalty_benefits": {
                    "immediate_benefits": ["Unable to parse"],
                    "long_term_value": "Unable to parse",
                    "exclusive_offers": ["Unable to parse"]
                },
                "recommendations": ["Unable to parse"]
            }
    
    def _generate_fallback_explanation(self, request: LoyaltyExplanationRequest) -> LoyaltyExplanationResponse:
        """Generate fallback explanation when LLM is not available."""
        applied_policies = request.applied_policies
        winner_rail = request.winner_rail
        
        # Generate contextual explanation
        if applied_policies:
            explanation = f"Applied {len(applied_policies)} loyalty policy(ies): {', '.join(applied_policies)}. "
            explanation += f"Policy adjustments favored {winner_rail} rail for optimal loyalty benefits."
        else:
            explanation = f"No specific loyalty policies applied. {winner_rail} rail selected based on standard scoring."
        
        # Add loyalty benefits
        if request.recommended_offer:
            explanation += f" Recommended loyalty offer: {request.recommended_offer}."
        
        # Analyze rail changes
        rail_changes = []
        for rail, before_score in request.before_scores.items():
            after_score = request.after_scores.get(rail, before_score)
            if abs(after_score - before_score) > 0.05:
                if after_score > before_score:
                    rail_changes.append(f"{rail} boosted by {after_score - before_score:.2f}")
                else:
                    rail_changes.append(f"{rail} reduced by {before_score - after_score:.2f}")
        
        # Key factors
        key_factors = ["loyalty_tier", "purchase_history", "policy_application"]
        if request.recommended_offer:
            key_factors.append("exclusive_offer")
        
        # Policy impact
        policy_impact = {
            "rail_changes": rail_changes if rail_changes else ["minimal changes"],
            "cost_savings": "Policy applied for cost optimization",
            "reward_boost": "Loyalty rewards enhanced"
        }
        
        # Loyalty benefits
        loyalty_benefits = {
            "immediate_benefits": ["enhanced_rewards", "cost_savings"],
            "long_term_value": "Improved loyalty tier progression",
            "exclusive_offers": [request.recommended_offer] if request.recommended_offer else []
        }
        
        # Recommendations
        recommendations = [
            "Continue using recommended payment methods",
            "Take advantage of loyalty offers",
            "Monitor loyalty tier progression"
        ]
        
        return LoyaltyExplanationResponse(
            explanation=explanation,
            confidence=0.6,
            key_factors=key_factors,
            policy_impact=policy_impact,
            loyalty_benefits=loyalty_benefits,
            recommendations=recommendations,
            model_provenance={
                "model_name": "fallback",
                "provider": "olive_deterministic",
                "status": "fallback_mode",
                "message": "Azure OpenAI not available, using deterministic explanation"
            },
            processing_time_ms=0,
            tokens_used=0
        )


# Global explainer instance
_explainer: Optional[OliveLLMExplainer] = None


def get_loyalty_explainer() -> OliveLLMExplainer:
    """Get the global loyalty explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = OliveLLMExplainer()
    return _explainer


def explain_loyalty_decision_llm(
    applied_policies: List[str],
    policy_adjustments: List[Dict[str, Any]],
    before_scores: Dict[str, float],
    after_scores: Dict[str, float],
    winner_rail: str,
    customer_context: Dict[str, Any],
    loyalty_offers: List[Dict[str, Any]],
    transaction_context: Dict[str, Any],
    recommended_offer: Optional[str] = None
) -> Optional[LoyaltyExplanationResponse]:
    """
    Generate LLM explanation for loyalty decision.
    
    Args:
        applied_policies: List of applied policy IDs
        policy_adjustments: Policy adjustments made
        before_scores: Rail scores before policy application
        after_scores: Rail scores after policy application
        winner_rail: Winning rail after policy application
        customer_context: Customer context and features
        loyalty_offers: Available loyalty offers
        transaction_context: Transaction context
        recommended_offer: Recommended loyalty offer
        
    Returns:
        LoyaltyExplanationResponse with explanation
    """
    request = LoyaltyExplanationRequest(
        applied_policies=applied_policies,
        policy_adjustments=policy_adjustments,
        before_scores=before_scores,
        after_scores=after_scores,
        winner_rail=winner_rail,
        customer_context=customer_context,
        loyalty_offers=loyalty_offers,
        recommended_offer=recommended_offer,
        transaction_context=transaction_context
    )
    
    explainer = get_loyalty_explainer()
    return explainer.explain_loyalty_decision(request)


def is_loyalty_llm_configured() -> bool:
    """Check if loyalty LLM explanation service is configured."""
    explainer = get_loyalty_explainer()
    return explainer.is_configured()

