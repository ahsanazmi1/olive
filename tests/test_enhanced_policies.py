"""
Unit tests for Olive Phase 4 Enhanced Policy functionality.

Tests policy DSL, MCP verbs, enforcement scenarios, and policy promotion.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path

from src.olive.policies.models import (
    PolicyDSL,
    PolicyCondition,
    PolicyAction,
    PolicyConditionType,
    PolicyActionType,
    RailType,
    SetPolicyRequest,
    GetPolicyRequest,
    EvaluatePoliciesRequest,
    PolicyEnforcementRequest,
    PolicyEvaluationRequest,
    get_policy_storage
)
from src.olive.policies.enhanced_evaluator import EnhancedPolicyEvaluator


class TestEnhancedPolicyDSL:
    """Test the enhanced Policy DSL functionality."""

    @pytest.fixture
    def sample_policy_dsl(self):
        """Create a sample enhanced policy DSL."""
        return PolicyDSL(
            policy_id="enhanced_policy_001",
            policy_name="Enhanced Merchant Routing Policy",
            description="Advanced policy with conditions and actions",
            prefer_rail=RailType.ACH,
            loyalty_rebate_pct=2.5,
            early_pay_discount_bps=100.0,
            
            # Enhanced policy rules
            conditions=[
                PolicyCondition(
                    condition_type=PolicyConditionType.AMOUNT_RANGE,
                    field="transaction_amount",
                    operator="gte",
                    value=100.0,
                    description="Applies to transactions >= $100"
                ),
                PolicyCondition(
                    condition_type=PolicyConditionType.TIME_WINDOW,
                    field="transaction_time",
                    operator="between",
                    value=[9, 17],  # 9 AM to 5 PM
                    description="Applies during business hours"
                )
            ],
            
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.RAIL_PREFERENCE,
                    parameters={"rail_type": "ACH", "boost_value": 0.2},
                    weight=1.5,
                    description="Strong preference for ACH rail"
                ),
                PolicyAction(
                    action_type=PolicyActionType.REBATE_APPLICATION,
                    parameters={"rail_type": "ACH", "rebate_pct": 1.5},
                    weight=1.0,
                    description="1.5% rebate for ACH transactions"
                )
            ],
            
            # Merchant routing rules
            rebate_rules={
                "ACH": 2.0,
                "debit": 1.0,
                "credit": 0.5
            },
            early_pay_rules={
                "ACH": 150.0,  # 150 basis points
                "debit": 100.0
            },
            loyalty_incentives={
                "premium_customer": 1.2,
                "frequent_buyer": 0.8
            },
            tax_validation_rules={
                "required": True,
                "threshold": 100.0,
                "jurisdictions": ["US", "CA"]
            },
            
            # Policy enforcement
            enforcement_mode="mandatory",
            override_threshold=0.8,
            
            # Metadata
            merchant_id="merchant_001",
            priority=3,
            enabled=True
        )

    def test_policy_dsl_creation(self, sample_policy_dsl):
        """Test enhanced policy DSL creation."""
        assert sample_policy_dsl.policy_id == "enhanced_policy_001"
        assert sample_policy_dsl.prefer_rail == RailType.ACH
        assert len(sample_policy_dsl.conditions) == 2
        assert len(sample_policy_dsl.actions) == 2
        assert sample_policy_dsl.enforcement_mode == "mandatory"

    def test_policy_conditions_validation(self, sample_policy_dsl):
        """Test policy conditions validation."""
        conditions = sample_policy_dsl.conditions
        
        # Test amount range condition
        amount_condition = conditions[0]
        assert amount_condition.condition_type == PolicyConditionType.AMOUNT_RANGE
        assert amount_condition.operator == "gte"
        assert amount_condition.value == 100.0
        
        # Test time window condition
        time_condition = conditions[1]
        assert time_condition.condition_type == PolicyConditionType.TIME_WINDOW
        assert time_condition.operator == "between"
        assert time_condition.value == [9, 17]

    def test_policy_actions_validation(self, sample_policy_dsl):
        """Test policy actions validation."""
        actions = sample_policy_dsl.actions
        
        # Test rail preference action
        rail_action = actions[0]
        assert rail_action.action_type == PolicyActionType.RAIL_PREFERENCE
        assert rail_action.parameters["rail_type"] == "ACH"
        assert rail_action.parameters["boost_value"] == 0.2
        assert rail_action.weight == 1.5
        
        # Test rebate application action
        rebate_action = actions[1]
        assert rebate_action.action_type == PolicyActionType.REBATE_APPLICATION
        assert rebate_action.parameters["rail_type"] == "ACH"
        assert rebate_action.parameters["rebate_pct"] == 1.5

    def test_merchant_routing_rules(self, sample_policy_dsl):
        """Test merchant routing rules."""
        # Test rebate rules
        assert sample_policy_dsl.rebate_rules["ACH"] == 2.0
        assert sample_policy_dsl.rebate_rules["debit"] == 1.0
        assert sample_policy_dsl.rebate_rules["credit"] == 0.5
        
        # Test early pay rules
        assert sample_policy_dsl.early_pay_rules["ACH"] == 150.0
        assert sample_policy_dsl.early_pay_rules["debit"] == 100.0
        
        # Test loyalty incentives
        assert sample_policy_dsl.loyalty_incentives["premium_customer"] == 1.2
        assert sample_policy_dsl.loyalty_incentives["frequent_buyer"] == 0.8
        
        # Test tax validation rules
        assert sample_policy_dsl.tax_validation_rules["required"] is True
        assert sample_policy_dsl.tax_validation_rules["threshold"] == 100.0
        assert "US" in sample_policy_dsl.tax_validation_rules["jurisdictions"]


class TestEnhancedPolicyEvaluator:
    """Test the enhanced policy evaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create an enhanced policy evaluator."""
        return EnhancedPolicyEvaluator()

    @pytest.fixture
    def sample_evaluation_request(self):
        """Create a sample policy evaluation request."""
        return PolicyEvaluationRequest(
            transaction_amount=150.0,
            merchant_id="merchant_001",
            rail_candidates=[
                {"rail_type": "ACH", "score": 0.7},
                {"rail_type": "debit", "score": 0.6},
                {"rail_type": "credit", "score": 0.5}
            ],
            current_scores={
                "ACH": 0.7,
                "debit": 0.6,
                "credit": 0.5
            },
            trace_id="test_trace_123"
        )

    def test_condition_evaluation_amount_range(self, evaluator, sample_evaluation_request):
        """Test amount range condition evaluation."""
        condition = PolicyCondition(
            condition_type=PolicyConditionType.AMOUNT_RANGE,
            field="transaction_amount",
            operator="gte",
            value=100.0,
            description="Test condition"
        )
        
        # Should pass for amount >= 100
        assert evaluator._evaluate_condition(condition, sample_evaluation_request) is True
        
        # Test with lower amount
        sample_evaluation_request.transaction_amount = 50.0
        assert evaluator._evaluate_condition(condition, sample_evaluation_request) is False

    def test_condition_evaluation_between_operator(self, evaluator, sample_evaluation_request):
        """Test between operator for amount range."""
        condition = PolicyCondition(
            condition_type=PolicyConditionType.AMOUNT_RANGE,
            field="transaction_amount",
            operator="between",
            value=[100.0, 500.0],
            description="Amount between $100-$500"
        )
        
        # Should pass for amount in range
        assert evaluator._evaluate_condition(condition, sample_evaluation_request) is True
        
        # Test with amount below range
        sample_evaluation_request.transaction_amount = 50.0
        assert evaluator._evaluate_condition(condition, sample_evaluation_request) is False
        
        # Test with amount above range
        sample_evaluation_request.transaction_amount = 600.0
        assert evaluator._evaluate_condition(condition, sample_evaluation_request) is False

    def test_policy_conditions_met(self, evaluator, sample_evaluation_request, sample_policy_dsl):
        """Test policy conditions evaluation."""
        # Test with conditions that should be met
        assert evaluator._policy_conditions_met(sample_policy_dsl, sample_evaluation_request) is True
        
        # Test with amount below threshold
        sample_evaluation_request.transaction_amount = 50.0
        assert evaluator._policy_conditions_met(sample_policy_dsl, sample_evaluation_request) is False

    def test_merchant_routing_rules_application(self, evaluator, sample_evaluation_request, sample_policy_dsl):
        """Test merchant routing rules application."""
        adjustments = evaluator._apply_merchant_routing_rules(sample_policy_dsl, sample_evaluation_request)
        
        # Should have adjustments for rebate rules, early pay rules, and loyalty incentives
        assert len(adjustments) >= 3
        
        # Check for rebate adjustments
        rebate_adjustments = [adj for adj in adjustments if adj.adjustment_type == "rebate_application"]
        assert len(rebate_adjustments) == 3  # ACH, debit, credit
        
        # Check for early pay adjustments
        early_pay_adjustments = [adj for adj in adjustments if adj.adjustment_type == "early_pay_incentive"]
        assert len(early_pay_adjustments) == 2  # ACH, debit
        
        # Check for loyalty boost adjustments
        loyalty_adjustments = [adj for adj in adjustments if adj.adjustment_type == "loyalty_boost"]
        assert len(loyalty_adjustments) == 2  # premium_customer, frequent_buyer

    def test_policy_evaluation_response(self, evaluator, sample_evaluation_request, sample_policy_dsl):
        """Test complete policy evaluation response."""
        # Mock storage to return our sample policy
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[sample_policy_dsl]):
            response = evaluator.evaluate_policies_enhanced(sample_evaluation_request)
        
        assert len(response.adjustments) > 0
        assert "ACH" in response.updated_scores
        assert "debit" in response.updated_scores
        assert "credit" in response.updated_scores
        
        # Scores should be updated (likely increased due to policy adjustments)
        assert response.updated_scores["ACH"] >= sample_evaluation_request.current_scores["ACH"]

    def test_policy_enforcement_during_negotiation(self, evaluator):
        """Test policy enforcement during negotiation."""
        enforcement_request = PolicyEnforcementRequest(
            merchant_id="merchant_001",
            negotiation_context={
                "transaction_amount": 150.0,
                "customer_segment": "premium",
                "rail_options": ["ACH", "debit", "credit"]
            },
            trace_id="enforcement_trace_123",
            enforcement_mode="mandatory"
        )
        
        # Mock storage to return sample policies
        sample_policy = PolicyDSL(
            policy_id="enforcement_policy",
            policy_name="Enforcement Test Policy",
            description="Test policy for enforcement",
            prefer_rail=RailType.ACH,
            enforcement_mode="mandatory",
            merchant_id="merchant_001",
            enabled=True
        )
        
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[sample_policy]):
            results = evaluator.enforce_policies_during_negotiation(enforcement_request)
        
        assert results["enforcement_mode"] == "mandatory"
        assert "policy_constraints" in results
        assert "score_overrides" in results
        assert "rail_restrictions" in results
        assert "enforcement_actions" in results


class TestMCPVerbs:
    """Test MCP verb functionality."""

    @pytest.fixture
    def sample_set_policy_request(self, sample_policy_dsl):
        """Create a sample set policy request."""
        return SetPolicyRequest(
            policy=sample_policy_dsl,
            overwrite=False
        )

    @pytest.fixture
    def sample_get_policy_request(self):
        """Create a sample get policy request."""
        return GetPolicyRequest(
            merchant_id="merchant_001",
            enabled_only=True
        )

    def test_set_policy_request_creation(self, sample_set_policy_request, sample_policy_dsl):
        """Test set policy request creation."""
        assert sample_set_policy_request.policy == sample_policy_dsl
        assert sample_set_policy_request.overwrite is False

    def test_get_policy_request_creation(self, sample_get_policy_request):
        """Test get policy request creation."""
        assert sample_get_policy_request.merchant_id == "merchant_001"
        assert sample_get_policy_request.enabled_only is True

    def test_evaluate_policies_request_creation(self, sample_evaluation_request):
        """Test evaluate policies request creation."""
        request = EvaluatePoliciesRequest(
            evaluation_request=sample_evaluation_request,
            include_explanations=True,
            enforce_mode="advisory"
        )
        
        assert request.evaluation_request == sample_evaluation_request
        assert request.include_explanations is True
        assert request.enforce_mode == "advisory"

    def test_policy_enforcement_request_creation(self):
        """Test policy enforcement request creation."""
        request = PolicyEnforcementRequest(
            merchant_id="merchant_001",
            negotiation_context={"test": "context"},
            trace_id="enforcement_trace",
            enforcement_mode="mandatory"
        )
        
        assert request.merchant_id == "merchant_001"
        assert request.negotiation_context == {"test": "context"}
        assert request.trace_id == "enforcement_trace"
        assert request.enforcement_mode == "mandatory"


class TestPolicyOverrideScenarios:
    """Test scenarios where policies override cheapest path."""

    @pytest.fixture
    def evaluator(self):
        """Create an enhanced policy evaluator."""
        return EnhancedPolicyEvaluator()

    def test_policy_overrides_cheapest_path(self, evaluator):
        """Test that policies can override the cheapest path."""
        # Create a policy that strongly prefers ACH over cheaper options
        override_policy = PolicyDSL(
            policy_id="override_policy",
            policy_name="Override Cheapest Path Policy",
            description="Policy that overrides cheapest path",
            prefer_rail=RailType.ACH,
            enforcement_mode="mandatory",
            override_threshold=0.5,
            merchant_id="merchant_001",
            priority=10,  # High priority
            enabled=True,
            
            # Strong rebate for ACH to make it attractive despite higher base cost
            rebate_rules={"ACH": 5.0},  # 5% rebate
            early_pay_rules={"ACH": 300.0},  # 3% early pay discount
            
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.RAIL_PREFERENCE,
                    parameters={"rail_type": "ACH", "boost_value": 0.5},
                    weight=2.0,  # Strong weight
                    description="Strong ACH preference"
                )
            ]
        )
        
        # Create evaluation request with ACH being more expensive initially
        evaluation_request = PolicyEvaluationRequest(
            transaction_amount=200.0,
            merchant_id="merchant_001",
            rail_candidates=[
                {"rail_type": "ACH", "base_cost": 2.0, "score": 0.4},  # Higher cost, lower initial score
                {"rail_type": "debit", "base_cost": 1.0, "score": 0.8},  # Lower cost, higher initial score
                {"rail_type": "credit", "base_cost": 1.5, "score": 0.6}
            ],
            current_scores={
                "ACH": 0.4,
                "debit": 0.8,  # Cheapest path
                "credit": 0.6
            },
            trace_id="override_test_trace"
        )
        
        # Mock storage to return our override policy
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[override_policy]):
            response = evaluator.evaluate_policies_enhanced(evaluation_request)
        
        # ACH should now have the highest score due to policy adjustments
        assert response.updated_scores["ACH"] > response.updated_scores["debit"]
        assert response.updated_scores["ACH"] > response.updated_scores["credit"]
        
        # Should have multiple adjustments for ACH
        ach_adjustments = [adj for adj in response.adjustments if adj.rail_type == "ACH"]
        assert len(ach_adjustments) >= 3  # Rebate, early pay, rail preference

    def test_policy_promotion_via_learning_loop(self, evaluator):
        """Test policy promotion via learning loop integration."""
        # Create a policy that gets promoted based on learning loop feedback
        learning_policy = PolicyDSL(
            policy_id="learning_promoted_policy",
            policy_name="Learning-Promoted Policy",
            description="Policy promoted by Weave learning loop",
            prefer_rail=RailType.ACH,
            enforcement_mode="advisory",
            merchant_id="merchant_001",
            priority=5,
            enabled=True,
            
            # Dynamic adjustments based on learning feedback
            loyalty_incentives={
                "high_value_customer": 1.5,  # Increased for learning-promoted customers
                "frequent_buyer": 1.2
            },
            
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.LOYALTY_BOOST,
                    parameters={"loyalty_type": "learning_promoted", "boost_value": 0.3},
                    weight=1.5,  # Higher weight due to learning promotion
                    description="Boost for learning-promoted segments"
                )
            ]
        )
        
        evaluation_request = PolicyEvaluationRequest(
            transaction_amount=300.0,
            merchant_id="merchant_001",
            rail_candidates=[
                {"rail_type": "ACH", "score": 0.6},
                {"rail_type": "debit", "score": 0.7},
                {"rail_type": "credit", "score": 0.5}
            ],
            current_scores={
                "ACH": 0.6,
                "debit": 0.7,
                "credit": 0.5
            },
            trace_id="learning_promotion_trace"
        )
        
        # Mock storage to return learning-promoted policy
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[learning_policy]):
            response = evaluator.evaluate_policies_enhanced(evaluation_request)
        
        # Should have learning-related adjustments
        learning_adjustments = [
            adj for adj in response.adjustments 
            if "learning" in adj.description.lower() or adj.adjustment_type == "loyalty_boost"
        ]
        assert len(learning_adjustments) > 0
        
        # ACH should benefit from learning promotions
        assert response.updated_scores["ACH"] > evaluation_request.current_scores["ACH"]

    def test_deterministic_policy_scoring(self, evaluator):
        """Test deterministic policy scoring for same inputs."""
        policy = PolicyDSL(
            policy_id="deterministic_policy",
            policy_name="Deterministic Policy",
            description="Policy for deterministic testing",
            prefer_rail=RailType.ACH,
            merchant_id="merchant_001",
            enabled=True,
            rebate_rules={"ACH": 2.0},
            early_pay_rules={"ACH": 100.0}
        )
        
        evaluation_request = PolicyEvaluationRequest(
            transaction_amount=150.0,
            merchant_id="merchant_001",
            rail_candidates=[{"rail_type": "ACH", "score": 0.5}],
            current_scores={"ACH": 0.5},
            trace_id="deterministic_test"
        )
        
        # Run evaluation multiple times with same inputs
        results = []
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[policy]):
            for _ in range(5):
                response = evaluator.evaluate_policies_enhanced(evaluation_request)
                results.append(response.updated_scores["ACH"])
        
        # All results should be identical (deterministic)
        assert all(score == results[0] for score in results)

    def test_explainability_diffs_between_policies(self, evaluator):
        """Test explainability differences between different policies."""
        # Create two different policies
        policy_1 = PolicyDSL(
            policy_id="cashback_policy",
            policy_name="Cashback-Focused Policy",
            description="Policy focused on cashback rewards",
            merchant_id="merchant_001",
            enabled=True,
            rebate_rules={"ACH": 3.0, "credit": 2.0},
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.REBATE_APPLICATION,
                    parameters={"rail_type": "ACH", "rebate_pct": 3.0},
                    weight=1.5,
                    description="High cashback for ACH"
                )
            ]
        )
        
        policy_2 = PolicyDSL(
            policy_id="speed_policy",
            policy_name="Speed-Focused Policy",
            description="Policy focused on settlement speed",
            merchant_id="merchant_001",
            enabled=True,
            early_pay_rules={"debit": 200.0, "credit": 150.0},
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.EARLY_PAY_INCENTIVE,
                    parameters={"rail_type": "debit", "incentive_bps": 200.0},
                    weight=1.5,
                    description="Fast settlement for debit"
                )
            ]
        )
        
        base_evaluation_request = PolicyEvaluationRequest(
            transaction_amount=200.0,
            merchant_id="merchant_001",
            rail_candidates=[
                {"rail_type": "ACH", "score": 0.5},
                {"rail_type": "debit", "score": 0.5},
                {"rail_type": "credit", "score": 0.5}
            ],
            current_scores={"ACH": 0.5, "debit": 0.5, "credit": 0.5},
            trace_id="explainability_test"
        )
        
        # Test with cashback policy
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[policy_1]):
            response_1 = evaluator.evaluate_policies_enhanced(base_evaluation_request)
        
        # Test with speed policy
        with patch.object(evaluator.storage, 'get_policies_by_merchant', return_value=[policy_2]):
            response_2 = evaluator.evaluate_policies_enhanced(base_evaluation_request)
        
        # Should have different explanations and score adjustments
        cashback_adjustments = [adj for adj in response_1.adjustments if "cashback" in adj.description.lower()]
        speed_adjustments = [adj for adj in response_2.adjustments if "settlement" in adj.description.lower()]
        
        assert len(cashback_adjustments) > 0
        assert len(speed_adjustments) > 0
        
        # ACH should score higher with cashback policy
        assert response_1.updated_scores["ACH"] > response_2.updated_scores["ACH"]
        
        # Debit should score higher with speed policy
        assert response_2.updated_scores["debit"] > response_1.updated_scores["debit"]


@pytest.fixture
def sample_policy_dsl():
    """Create a sample enhanced policy DSL for tests."""
    return PolicyDSL(
        policy_id="test_policy_001",
        policy_name="Test Enhanced Policy",
        description="Test policy for unit tests",
        prefer_rail=RailType.ACH,
        loyalty_rebate_pct=2.0,
        early_pay_discount_bps=100.0,
        merchant_id="merchant_001",
        priority=1,
        enabled=True,
        rebate_rules={"ACH": 1.5, "debit": 1.0},
        early_pay_rules={"ACH": 100.0},
        loyalty_incentives={"premium": 1.2},
        tax_validation_rules={"required": True},
        enforcement_mode="advisory"
    )








