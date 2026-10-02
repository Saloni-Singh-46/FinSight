"""Unit tests for Temporal Safety and Look-Ahead Bias Prevention."""

import numpy as np
import pandas as pd

from finsight.data.leakage_guard import assert_no_lookahead_leakage
from finsight.features.technical import compute_technical_features


def test_technical_features_no_lookahead_leakage():
    """Asserts that modifying future data points leaves all historical feature values strictly unchanged."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=200)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 200))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(100000, 500000, 200),
        }
    )

    # Test at multiple cutoff points
    for check_idx in [40, 80, 120, 150]:
        assert assert_no_lookahead_leakage(compute_technical_features, df, check_idx=check_idx)


def test_target_shift_future_only():
    """Asserts target columns strictly depend on future returns and not past."""
    dates = pd.bdate_range("2023-01-01", periods=50)
    prices = np.arange(1, 51, dtype=float)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": 1000,
        }
    )
    feat_df = compute_technical_features(df, horizons=[5])
    # For monotonically increasing prices, target_dir_5d must be 1 everywhere except the last 5 rows where it is NaN/0
    assert (feat_df["target_dir_5d"].iloc[:-5] == 1).all()
