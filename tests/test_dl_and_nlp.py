"""Unit tests for Deep Learning Forecasters and FinBERT Sentiment NLP."""

import numpy as np
import pandas as pd
import pytest

from finsight.models.dl.baselines import ForecastingBaselinesEvaluator
from finsight.models.dl.lstm import LSTMPipeline
from finsight.models.nlp.aggregator import SentimentAggregator
from finsight.models.nlp.finbert import FinancialSentimentClassifier


@pytest.fixture
def sample_price_series():
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=250)
    prices = 150 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 250))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(50000, 150000, 250),
        }
    )
    return df


def test_lstm_forecaster_train_and_fan(sample_price_series):
    pipeline = LSTMPipeline(seq_length=15, forecast_horizon=10, epochs=3, batch_size=16)
    metrics = pipeline.train_walk_forward(sample_price_series)
    assert "mae" in metrics
    assert "interval_coverage_80pct" in metrics

    fan_chart = pipeline.predict_future_fan(sample_price_series, horizon_days=21)
    assert len(fan_chart) == 21
    assert "price_p10" in fan_chart[0]
    assert "price_p50" in fan_chart[0]
    assert "price_p90" in fan_chart[0]
    # Check interval ordering
    assert fan_chart[0]["price_p10"] <= fan_chart[0]["price_p50"] <= fan_chart[0]["price_p90"]


def test_forecasting_baselines(sample_price_series):
    evaluator = ForecastingBaselinesEvaluator(horizon_days=10)
    results = evaluator.evaluate_all(sample_price_series)
    assert len(results) >= 3
    assert any("Naive" in r["model_name"] for r in results)


def test_finbert_sentiment_classification():
    clf = FinancialSentimentClassifier()
    pos_res = clf.predict_single(
        "Tata Motors Q4 profit surges 218% on robust JLR demand and operating margins"
    )
    assert pos_res["label"] == "positive"
    assert pos_res["sentiment_score"] > 0

    neg_res = clf.predict_single(
        "Company reports steep losses amid recession, inflation headwinds, and supply chain crisis"
    )
    assert neg_res["label"] == "negative"
    assert neg_res["sentiment_score"] < 0


def test_sentiment_aggregator():
    agg = SentimentAggregator()
    news = [
        {"title": "Tata Motors commercial vehicle sales grow 15% YoY", "date": "2024-03-01"},
        {"title": "JLR order book reaches record 150,000 units", "date": "2024-03-02"},
        {"title": "Raw material inflation puts minor pressure on margins", "date": "2024-03-03"},
    ]
    summary = agg.aggregate_news_items(news)
    assert "net_sentiment_score" in summary
    assert summary["positive_count"] >= 1
    assert len(summary["top_positive_headlines"]) > 0
