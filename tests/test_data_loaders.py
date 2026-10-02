"""Unit tests for Data Loaders and Cleaning Utilities."""

import pandas as pd

from finsight.data.cleaning import clean_price_dataframe
from finsight.data.loaders import FinancialDataLoader


def test_data_loader_price_history():
    loader = FinancialDataLoader()
    df = loader.load_price_history("TATAMOTORS.NS")
    assert not df.empty
    assert "Close" in df.columns
    assert "Volume" in df.columns
    assert len(df) > 500


def test_data_loader_fundamentals():
    loader = FinancialDataLoader()
    fund = loader.load_fundamentals("TATAMOTORS.NS")
    assert isinstance(fund, dict)
    assert fund["ticker"] == "TATAMOTORS.NS"
    assert "pe_ratio" in fund
    assert "peers" in fund
    assert len(fund["peers"]) >= 3


def test_clean_price_dataframe():
    raw = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-02"],  # Duplicate date
            "Open": [100.0, 105.0, 105.0],
            "High": [102.0, 104.0, 104.0],  # High < Open anomaly
            "Low": [98.0, 95.0, 95.0],
            "Close": [101.0, 103.0, 103.0],
            "Volume": [10000, -50, 10000],  # Negative volume
        }
    )
    cleaned = clean_price_dataframe(raw)
    assert len(cleaned) == 2  # Deduplicated
    assert cleaned["High"].iloc[1] >= cleaned["Open"].iloc[1]  # High corrected
    assert (cleaned["Volume"] >= 0).all()  # Non-negative volume
