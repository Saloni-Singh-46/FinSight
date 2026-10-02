"""Unit tests for ML classifiers, Anomaly Detection, Credit Risk, and Backtesting."""

import numpy as np
import pandas as pd
import pytest

from finsight.backtest.engine import BacktestEngine
from finsight.features.technical import compute_technical_features
from finsight.models.ml.anomaly import MarketAnomalyDetector
from finsight.models.ml.classifier import DirectionClassifierPipeline
from finsight.models.ml.credit_score import CreditRiskScorecard


@pytest.fixture
def sample_features():
    np.random.seed(42)
    dates = pd.bdate_range("2021-01-01", periods=600)
    prices = 200 * np.cumprod(1 + np.random.normal(0.0006, 0.018, 600))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.995,
            "High": prices * 1.015,
            "Low": prices * 0.985,
            "Close": prices,
            "Volume": np.random.randint(50000, 200000, 600),
        }
    )
    return compute_technical_features(df, horizons=[5, 21])


def test_classifier_walk_forward(sample_features):
    pipeline = DirectionClassifierPipeline(horizon_days=21, model_type="xgboost", seed=42)
    metrics = pipeline.train_walk_forward(sample_features, n_splits=3)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] >= 0.0

    pred = pipeline.predict_latest(sample_features)
    assert "signal" in pred
    assert 0.0 <= pred["probability_up"] <= 1.0


def test_naive_baseline(sample_features):
    baseline = DirectionClassifierPipeline(horizon_days=21, model_type="naive")
    metrics = baseline.train_walk_forward(sample_features, n_splits=3)
    assert "accuracy" in metrics


def test_anomaly_detection(sample_features):
    detector = MarketAnomalyDetector(contamination=0.03)
    anom_df = detector.fit_detect(sample_features)
    assert "is_anomaly" in anom_df.columns
    assert "anomaly_type" in anom_df.columns
    anomalies = detector.get_recent_anomalies(anom_df)
    assert isinstance(anomalies, list)


def test_credit_risk_scorecard():
    scorecard = CreditRiskScorecard()
    health = scorecard.evaluate_company_health(
        {
            "altman_z_score": 2.85,
            "piotroski_f_score": 8,
            "debt_to_equity": 0.88,
            "interest_coverage": 8.45,
            "roe_pct": 31.6,
        }
    )
    assert "composite_health_score" in health
    assert health["composite_health_score"] > 50
    assert "risk_tier" in health


def test_backtest_friction_and_sharpe():
    dates = pd.bdate_range("2023-01-01", periods=100)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 100))
    df = pd.DataFrame({"Date": dates, "Close": prices})
    signals = pd.Series(np.random.choice([0, 1], size=100))

    engine = BacktestEngine(initial_capital=1000000.0, transaction_cost_bps=10.0, slippage_bps=5.0)
    res = engine.run_strategy(df, signals)
    assert "metrics" in res
    assert "sharpe_ratio" in res["metrics"]
    assert "sharpe_95_ci" in res["metrics"]
    assert len(res["equity_curve"]) == 100
