import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from olive.ml.policy_optimization import PolicyOptimizationModel
from olive.ml.incentive_effectiveness import IncentiveEffectivenessModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_models():
    logger.info("🚀 Starting Olive ML model training...")

    # Train Policy Optimization Model
    logger.info("🔧 Training policy optimization model...")
    policy_optimizer = PolicyOptimizationModel()
    
    # Create synthetic data for policy optimization
    n_samples = 5000
    np.random.seed(42)
    
    data_policy = {
        "transaction_amount": np.random.uniform(10, 5000, n_samples),
        "transaction_frequency": np.random.uniform(0.1, 20.0, n_samples),
        "avg_transaction_amount": np.random.uniform(20, 3000, n_samples),
        "loyalty_tier": np.random.randint(1, 6, n_samples),
        "time_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "seasonal_factor": np.random.uniform(0.8, 1.2, n_samples),
        "competitive_pressure": np.random.uniform(0.0, 1.0, n_samples),
        "cost_sensitivity": np.random.uniform(0.0, 1.0, n_samples),
        "reward_preference": np.random.uniform(0.0, 1.0, n_samples),
        "payment_method_age_days": np.random.randint(10, 2000, n_samples),
        "account_age_days": np.random.randint(10, 3000, n_samples),
        "previous_policy_success_rate": np.random.uniform(0.3, 1.0, n_samples),
        "merchant_volume_tier": np.random.randint(1, 6, n_samples),
        "channel_online": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "channel_mobile": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "channel_in_store": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "segment_new": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "segment_regular": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "segment_vip": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "rail_ach": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "rail_debit": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "rail_credit": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "rail_bnpl": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }
    df_policy = pd.DataFrame(data_policy)
    
    # Generate synthetic policy optimization targets (0-1 scale)
    df_policy['optimization_score'] = (
        df_policy['loyalty_tier'] / 5.0 * 0.2 +
        (1 - df_policy['cost_sensitivity']) * 0.15 +
        df_policy['reward_preference'] * 0.15 +
        df_policy['previous_policy_success_rate'] * 0.2 +
        (1 - df_policy['competitive_pressure']) * 0.1 +
        df_policy['seasonal_factor'] / 2.0 * 0.1 +
        np.random.normal(0, 0.05, n_samples)
    )
    df_policy['optimization_score'] = np.clip(df_policy['optimization_score'], 0, 1)
    
    policy_optimizer.train_model(df_policy[policy_optimizer.feature_names], df_policy['optimization_score'])
    logger.info("✅ Policy optimization model trained and saved")

    # Train Incentive Effectiveness Model
    logger.info("🔧 Training incentive effectiveness model...")
    effectiveness_predictor = IncentiveEffectivenessModel()
    
    # Create synthetic data for incentive effectiveness
    data_effectiveness = {
        "transaction_amount": np.random.uniform(10, 5000, n_samples),
        "customer_age_days": np.random.randint(10, 3000, n_samples),
        "loyalty_tier": np.random.randint(1, 6, n_samples),
        "transaction_frequency": np.random.uniform(0.1, 20.0, n_samples),
        "avg_transaction_amount": np.random.uniform(20, 3000, n_samples),
        "incentive_value": np.random.uniform(1, 200, n_samples),
        "incentive_percentage": np.random.uniform(0.5, 15.0, n_samples),
        "time_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "seasonal_factor": np.random.uniform(0.8, 1.2, n_samples),
        "competitive_pressure": np.random.uniform(0.0, 1.0, n_samples),
        "customer_engagement_score": np.random.uniform(0.0, 1.0, n_samples),
        "previous_incentive_response_rate": np.random.uniform(0.2, 1.0, n_samples),
        "merchant_trust_score": np.random.uniform(0.5, 1.0, n_samples),
        "economic_indicator": np.random.uniform(0.3, 1.0, n_samples),
        "incentive_type_cashback": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "incentive_type_points": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "incentive_type_discount": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "incentive_type_bonus": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "customer_segment_new": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "customer_segment_regular": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "customer_segment_vip": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "merchant_category_retail": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "merchant_category_food": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "merchant_category_travel": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "merchant_category_other": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "channel_online": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "channel_mobile": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "channel_in_store": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }
    df_effectiveness = pd.DataFrame(data_effectiveness)
    
    # Generate synthetic effectiveness labels (target variable)
    df_effectiveness['is_effective'] = (
        (df_effectiveness['customer_engagement_score'] > 0.6) * 0.2 +
        (df_effectiveness['previous_incentive_response_rate'] > 0.7) * 0.2 +
        (df_effectiveness['loyalty_tier'] >= 3) * 0.15 +
        (df_effectiveness['incentive_percentage'] > 5.0) * 0.1 +
        (df_effectiveness['merchant_trust_score'] > 0.8) * 0.1 +
        (df_effectiveness['economic_indicator'] > 0.7) * 0.1 +
        (df_effectiveness['transaction_amount'] > 100) * 0.05 +
        np.random.choice([0, 1], n_samples, p=[0.2, 0.8]) * 0.1
    ) > 0.5
    
    effectiveness_predictor.train_model(df_effectiveness[effectiveness_predictor.feature_names], df_effectiveness['is_effective'])
    logger.info("✅ Incentive effectiveness model trained and saved")

    logger.info("🎉 All ML models trained successfully!")


if __name__ == "__main__":
    train_all_models()
