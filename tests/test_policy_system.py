"""
Tests for Olive policy system.

This module tests the Policy DSL, MCP verbs, policy evaluation,
and CloudEvent emission functionality.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from src.olive.policies.models import (
    PolicyDSL,
    PolicyAdjustment,
    PolicyEvaluationRequest,
    get_policy_storage,
    RailType,
)
from src.olive.policies.evaluator import get_policy_evaluator
from src.olive.policies.loader import get_policy_loader
from src.olive.policies.explainer import get_policy_explainer
from src.olive.policies.events import get_policy_event_emitter


class TestPolicyDSL:
    """Test Policy DSL models and validation."""
    
    def test_policy_dsl_creation(self):
        """Test basic policy DSL creation."""
        policy = PolicyDSL(
            policy_id="test_policy_001",
            policy_name="Test Policy",
            description="A test policy",
            prefer_rail=RailType.DEBIT,
            loyalty_rebate_pct=5.0,
            early_pay_discount_bps=100.0,
            merchant_id="merchant_001"
        )
        
        assert policy.policy_id == "test_policy_001"
        assert policy.policy_name == "Test Policy"
        assert policy.prefer_rail == RailType.DEBIT
        assert policy.loyalty_rebate_pct == 5.0
        assert policy.early_pay_discount_bps == 100.0
        assert policy.merchant_id == "merchant_001"
        assert policy.enabled is True
        assert policy.priority == 1
    
    def test_policy_dsl_validation(self):
        """Test policy DSL validation rules."""
        # Test invalid loyalty rebate percentage
        with pytest.raises(ValueError, match="Loyalty rebate percentage must be between 0 and 100"):
            PolicyDSL(
                policy_id="test_policy_002",
                policy_name="Invalid Policy",
                loyalty_rebate_pct=150.0
            )
        
        # Test invalid early pay discount
        with pytest.raises(ValueError, match="Early payment discount must be between 0 and 10000 basis points"):
            PolicyDSL(
                policy_id="test_policy_003",
                policy_name="Invalid Policy",
                early_pay_discount_bps=15000.0
            )
    
    def test_policy_adjustment_creation(self):
        """Test policy adjustment creation."""
        adjustment = PolicyAdjustment(
            cost_bps_delta=-100.0,
            reward_bonus_delta=0.05,
            rail_preference_boost=0.2,
            policy_id="test_policy_001",
            adjustment_reason="Test adjustment",
            conditions_met=["loyalty_rebate_5%", "early_pay_discount_100bps"]
        )
        
        assert adjustment.cost_bps_delta == -100.0
        assert adjustment.reward_bonus_delta == 0.05
        assert adjustment.rail_preference_boost == 0.2
        assert adjustment.policy_id == "test_policy_001"
        assert len(adjustment.conditions_met) == 2


class TestPolicyLoader:
    """Test policy loading from YAML and JSON."""
    
    def test_load_policy_from_dict(self):
        """Test loading policy from dictionary."""
        loader = get_policy_loader()
        
        policy_data = {
            "policy_id": "yaml_policy_001",
            "policy_name": "YAML Test Policy",
            "description": "Policy loaded from YAML",
            "prefer_rail": "DEBIT",
            "loyalty_rebate_pct": 3.0,
            "early_pay_discount_bps": 50.0,
            "merchant_id": "merchant_001",
            "priority": 2
        }
        
        policy = loader.load_policy_from_dict(policy_data)
        
        assert policy.policy_id == "yaml_policy_001"
        assert policy.prefer_rail == RailType.DEBIT
        assert policy.loyalty_rebate_pct == 3.0
        assert policy.early_pay_discount_bps == 50.0
        assert policy.priority == 2
    
    def test_load_policies_from_yaml(self):
        """Test loading policies from YAML content."""
        loader = get_policy_loader()
        
        yaml_content = """
        - policy_id: yaml_policy_001
          policy_name: YAML Policy 1
          description: First YAML policy
          prefer_rail: ACH
          loyalty_rebate_pct: 2.0
          merchant_id: merchant_001
        - policy_id: yaml_policy_002
          policy_name: YAML Policy 2
          description: Second YAML policy
          prefer_rail: CREDIT
          early_pay_discount_bps: 75.0
          merchant_id: merchant_002
        """
        
        policies = loader.load_policies_from_yaml(yaml_content)
        
        assert len(policies) == 2
        assert policies[0].policy_id == "yaml_policy_001"
        assert policies[0].prefer_rail == RailType.ACH
        assert policies[1].policy_id == "yaml_policy_002"
        assert policies[1].prefer_rail == RailType.CREDIT
    
    def test_load_policies_from_json(self):
        """Test loading policies from JSON content."""
        loader = get_policy_loader()
        
        json_content = """
        [
            {
                "policy_id": "json_policy_001",
                "policy_name": "JSON Policy 1",
                "description": "First JSON policy",
                "prefer_rail": "DEBIT",
                "loyalty_rebate_pct": 4.0,
                "merchant_id": "merchant_001"
            },
            {
                "policy_id": "json_policy_002",
                "policy_name": "JSON Policy 2",
                "description": "Second JSON policy",
                "prefer_rail": "ACH",
                "early_pay_discount_bps": 200.0,
                "merchant_id": "merchant_002"
            }
        ]
        """
        
        policies = loader.load_policies_from_json(json_content)
        
        assert len(policies) == 2
        assert policies[0].policy_id == "json_policy_001"
        assert policies[0].prefer_rail == RailType.DEBIT
        assert policies[1].policy_id == "json_policy_002"
        assert policies[1].prefer_rail == RailType.ACH
    
    def test_malformed_policy_validation(self):
        """Test validation of malformed policy data."""
        loader = get_policy_loader()
        
        # Missing required field
        with pytest.raises(ValueError, match="Missing required field: policy_id"):
            loader.load_policy_from_dict({
                "policy_name": "Missing ID Policy"
            })
        
        # Invalid YAML
        with pytest.raises(ValueError, match="Invalid YAML format"):
            loader.load_policies_from_yaml("invalid: yaml: content: [")
        
        # Invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON format"):
            loader.load_policies_from_json('{"invalid": json}')


class TestPolicyEvaluation:
    """Test policy evaluation functionality."""
    
    def setup_method(self):
        """Set up test policies."""
        storage = get_policy_storage()
        
        # Clear existing policies
        for policy in storage.list_policies():
            storage.delete_policy(policy.policy_id)
        
        # Create test policies
        policy1 = PolicyDSL(
            policy_id="test_policy_001",
            policy_name="Debit Preference Policy",
            description="Prefers debit rail",
            prefer_rail=RailType.DEBIT,
            loyalty_rebate_pct=2.0,
            merchant_id="merchant_001",
            priority=1
        )
        
        policy2 = PolicyDSL(
            policy_id="test_policy_002",
            policy_name="Early Pay Discount Policy",
            description="Provides early payment discount",
            early_pay_discount_bps=150.0,
            merchant_id="merchant_001",
            priority=2
        )
        
        policy3 = PolicyDSL(
            policy_id="test_policy_003",
            policy_name="ACH Preference Policy",
            description="Prefers ACH rail",
            prefer_rail=RailType.ACH,
            merchant_id="merchant_002",
            priority=1
        )
        
        storage.store_policy(policy1)
        storage.store_policy(policy2)
        storage.store_policy(policy3)
    
    def test_policy_evaluation_basic(self):
        """Test basic policy evaluation."""
        evaluator = get_policy_evaluator()
        
        request = PolicyEvaluationRequest(
            merchant_id="merchant_001",
            transaction_amount=1000.0,
            rail_candidates=[
                {"rail_type": "ACH"},
                {"rail_type": "debit"},
                {"rail_type": "credit"}
            ],
            current_scores={
                "ACH": 0.6,
                "debit": 0.7,
                "credit": 0.8
            },
            trace_id="test_trace_001"
        )
        
        response = evaluator.evaluate_policies(request)
        
        assert len(response.applied_policies) > 0
        assert len(response.adjustments) > 0
        assert response.trace_id == "test_trace_001"
        assert response.winner_rail is not None
    
    def test_policy_flips_winner(self):
        """Test that policy application can flip the winning rail."""
        evaluator = get_policy_evaluator()
        
        # Create request where credit is initially winning
        request = PolicyEvaluationRequest(
            merchant_id="merchant_001",
            transaction_amount=1000.0,
            rail_candidates=[
                {"rail_type": "ACH"},
                {"rail_type": "debit"},
                {"rail_type": "credit"}
            ],
            current_scores={
                "ACH": 0.5,
                "debit": 0.6,  # Should be boosted by debit preference policy
                "credit": 0.7  # Initially winning
            },
            trace_id="test_trace_002"
        )
        
        response = evaluator.evaluate_policies(request)
        
        # Check that debit policy was applied (looking for test_policy_001 which prefers debit)
        debit_policies = [pid for pid in response.applied_policies if "test_policy_001" in pid]
        assert len(debit_policies) > 0
        
        # Check that debit might now be winning due to policy boost
        assert "debit" in response.updated_scores
        assert response.updated_scores["debit"] > 0.6  # Should be boosted
    
    def test_policy_noop(self):
        """Test policy evaluation when no policies apply."""
        evaluator = get_policy_evaluator()
        
        # Create request for merchant with no policies
        request = PolicyEvaluationRequest(
            merchant_id="merchant_999",  # No policies for this merchant
            transaction_amount=1000.0,
            rail_candidates=[
                {"rail_type": "ACH"},
                {"rail_type": "credit"}
            ],
            current_scores={
                "ACH": 0.6,
                "credit": 0.7
            },
            trace_id="test_trace_003"
        )
        
        response = evaluator.evaluate_policies(request)
        
        # Should have no applied policies
        assert len(response.applied_policies) == 0
        assert len(response.adjustments) == 0
        assert response.winner_rail == "credit"  # Original winner unchanged
        assert response.policy_impact == "No policies applied - using default scoring"
    
    def test_policy_priority_ordering(self):
        """Test that policies are applied in priority order."""
        evaluator = get_policy_evaluator()
        
        request = PolicyEvaluationRequest(
            merchant_id="merchant_001",
            transaction_amount=1000.0,
            rail_candidates=[
                {"rail_type": "ACH"},
                {"rail_type": "debit"}
            ],
            current_scores={
                "ACH": 0.5,
                "debit": 0.5  # Tie before policies
            },
            trace_id="test_trace_004"
        )
        
        response = evaluator.evaluate_policies(request)
        
        # Should have multiple policies applied
        assert len(response.applied_policies) >= 2
        
        # Higher priority policy should be listed first
        # (policies are sorted by priority in descending order)
        assert response.applied_policies[0] == "test_policy_002"  # priority=2
        assert response.applied_policies[1] == "test_policy_001"  # priority=1


class TestPolicyExplainer:
    """Test policy explanation functionality."""
    
    def test_explain_policy_application(self):
        """Test policy application explanation."""
        explainer = get_policy_explainer()
        
        adjustments = [
            PolicyAdjustment(
                cost_bps_delta=-100.0,
                reward_bonus_delta=0.02,
                rail_preference_boost=0.2,
                policy_id="test_policy_001",
                adjustment_reason="Applied debit preference and early pay discount",
                conditions_met=["preferred_rail_debit", "early_pay_discount_100bps"]
            )
        ]
        
        explanation = explainer.explain_policy_application(
            applied_policies=["test_policy_001"],
            adjustments=adjustments,
            before_scores={"ACH": 0.6, "debit": 0.5, "credit": 0.7},
            after_scores={"ACH": 0.6, "debit": 0.8, "credit": 0.7},
            winner_rail="debit"
        )
        
        assert "Applied 1 policy(ies)" in explanation
        assert "test_policy_001" in explanation
        assert "Reduced costs by 100 basis points" in explanation
        assert "Increased rewards by 2.0%" in explanation
        assert "Policy favored debit" in explanation
    
    def test_explain_policy_reason(self):
        """Test specific policy reason explanation."""
        explainer = get_policy_explainer()
        
        policy = PolicyDSL(
            policy_id="test_policy_001",
            policy_name="Test Policy",
            description="Test policy",
            prefer_rail=RailType.DEBIT,
            loyalty_rebate_pct=3.0,
            early_pay_discount_bps=150.0
        )
        
        adjustment = PolicyAdjustment(
            cost_bps_delta=-150.0,
            reward_bonus_delta=0.03,
            rail_preference_boost=0.2,
            policy_id="test_policy_001",
            adjustment_reason="Test adjustment",
            conditions_met=["preferred_rail_debit", "loyalty_rebate_3%", "early_pay_discount_150bps"]
        )
        
        reason = explainer.explain_policy_reason(policy, adjustment)
        
        assert "Test Policy prefers debit rail" in reason
        assert "offers 3.0% loyalty rebate" in reason
        assert "provides 150.0bps early payment discount" in reason


class TestPolicyEventEmission:
    """Test CloudEvent emission for policy applications."""
    
    def test_emit_policy_applied_event(self):
        """Test policy applied event emission."""
        emitter = get_policy_event_emitter()
        
        from src.olive.policies.evaluator import PolicyEvaluationResponse
        
        evaluation_response = PolicyEvaluationResponse(
            adjustments=[
                PolicyAdjustment(
                    cost_bps_delta=-100.0,
                    reward_bonus_delta=0.02,
                    rail_preference_boost=0.2,
                    policy_id="test_policy_001",
                    adjustment_reason="Test adjustment",
                    conditions_met=["preferred_rail_debit"]
                )
            ],
            updated_scores={"ACH": 0.6, "debit": 0.8, "credit": 0.7},
            applied_policies=["test_policy_001"],
            ignored_policies=[],
            trace_id="test_trace_001",
            winner_rail="debit",
            policy_impact="Applied debit preference policy"
        )
        
        event = emitter.emit_policy_applied_event(
            evaluation_response=evaluation_response,
            merchant_id="merchant_001",
            transaction_amount=1000.0,
            before_scores={"ACH": 0.6, "debit": 0.6, "credit": 0.7},
            llm_explanation="Policy favored debit due to rebate structure",
            evaluation_duration_ms=15.5
        )
        
        assert event.event_type == "ocn.olive.policy_applied.v1"
        assert event.source == "olive"
        assert event.trace_id == "test_trace_001"
        assert event.merchant_id == "merchant_001"
        assert len(event.applied_policies) == 1
        assert event.winner_rail == "debit"
        assert "Policy favored debit" in event.llm_explanation
        assert event.evaluation_duration_ms == 15.5
    
    def test_format_cloud_event(self):
        """Test CloudEvent formatting."""
        emitter = get_policy_event_emitter()
        
        from src.olive.policies.evaluator import PolicyEvaluationResponse
        
        evaluation_response = PolicyEvaluationResponse(
            adjustments=[],
            updated_scores={"ACH": 0.6, "debit": 0.7},
            applied_policies=[],
            ignored_policies=[],
            trace_id="test_trace_002",
            winner_rail="debit",
            policy_impact="No policies applied"
        )
        
        event = emitter.emit_policy_applied_event(
            evaluation_response=evaluation_response,
            merchant_id="merchant_001",
            transaction_amount=1000.0,
            before_scores={"ACH": 0.6, "debit": 0.7},
            llm_explanation="No policies applied",
            evaluation_duration_ms=5.0
        )
        
        cloud_event = emitter.format_cloud_event(event)
        
        assert cloud_event["specversion"] == "1.0"
        assert cloud_event["type"] == "ocn.olive.policy_applied.v1"
        assert cloud_event["source"] == "olive"
        assert cloud_event["datacontenttype"] == "application/json"
        assert "data" in cloud_event


if __name__ == "__main__":
    pytest.main([__file__])
