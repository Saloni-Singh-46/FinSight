"""Technical Indicators and Temporal Feature Engineering for Financial Time Series."""

import numpy as np
import pandas as pd


def compute_technical_features(
    df: pd.DataFrame,
    horizons: list[int] = [5, 21, 63],
    benchmark_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Computes leakage-safe technical indicators and calendar features.

    All features at index t are computed using data strictly at or prior to t.
    """
    feat_df = df.copy().sort_values("Date").reset_index(drop=True)
    close = feat_df["Close"]
    high = feat_df["High"]
    low = feat_df["Low"]
    volume = feat_df["Volume"]

    # 1. Historical Log Returns & Lags (at or before t)
    feat_df["ret_1d"] = close.pct_change(1)
    feat_df["ret_5d"] = close.pct_change(5)
    feat_df["ret_10d"] = close.pct_change(10)
    feat_df["ret_21d"] = close.pct_change(21)
    feat_df["ret_63d"] = close.pct_change(63)

    # 2. Rolling Realized Volatilities (Annualized, 252 trading days)
    feat_df["vol_5d"] = feat_df["ret_1d"].rolling(5).std() * np.sqrt(252)
    feat_df["vol_21d"] = feat_df["ret_1d"].rolling(21).std() * np.sqrt(252)
    feat_df["vol_63d"] = feat_df["ret_1d"].rolling(63).std() * np.sqrt(252)

    # 3. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    feat_df["rsi_14"] = 100 - (100 / (1 + rs))

    # 4. Moving Average Convergence Divergence (MACD 12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    feat_df["macd_line"] = ema_12 - ema_26
    feat_df["macd_signal"] = feat_df["macd_line"].ewm(span=9, adjust=False).mean()
    feat_df["macd_hist"] = feat_df["macd_line"] - feat_df["macd_signal"]

    # 5. Bollinger Bands (20, 2)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feat_df["bb_upper"] = sma_20 + 2 * std_20
    feat_df["bb_lower"] = sma_20 - 2 * std_20
    feat_df["bb_pct_b"] = (close - feat_df["bb_lower"]) / (
        feat_df["bb_upper"] - feat_df["bb_lower"] + 1e-9
    )

    # 6. Average True Range (ATR 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat_df["atr_14"] = tr.rolling(14).mean()
    feat_df["natr_14"] = (feat_df["atr_14"] / close) * 100  # Normalized ATR %

    # 7. Volume Z-Score & Moving Average
    vol_sma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat_df["vol_zscore_20"] = (volume - vol_sma_20) / (vol_std_20 + 1e-9)
    feat_df["vol_ratio_20"] = volume / (vol_sma_20 + 1e-9)

    # 8. Rolling Max Drawdown (63d and 252d)
    rolling_max_63 = close.rolling(63, min_periods=1).max()
    feat_df["drawdown_63d"] = (close - rolling_max_63) / rolling_max_63
    rolling_max_252 = close.rolling(252, min_periods=1).max()
    feat_df["drawdown_252d"] = (close - rolling_max_252) / rolling_max_252

    # 9. Moving Average Ratios (Price vs 50d, 200d SMA)
    feat_df["sma_50"] = close.rolling(50).mean()
    feat_df["sma_200"] = close.rolling(200).mean()
    feat_df["price_to_sma50"] = close / (feat_df["sma_50"] + 1e-9)
    feat_df["price_to_sma200"] = close / (feat_df["sma_200"] + 1e-9)

    # 10. Benchmark-Relative Excess Return (if available)
    if benchmark_df is not None and "Close" in benchmark_df.columns:
        bench = benchmark_df.copy().sort_values("Date").reset_index(drop=True)
        bench["bench_ret_1d"] = bench["Close"].pct_change(1)
        bench["bench_ret_21d"] = bench["Close"].pct_change(21)
        merged = pd.merge(
            feat_df,
            bench[["Date", "bench_ret_1d", "bench_ret_21d"]],
            on="Date",
            how="left",
        )
        feat_df["excess_ret_1d"] = feat_df["ret_1d"] - merged["bench_ret_1d"].fillna(0)
        feat_df["excess_ret_21d"] = feat_df["ret_21d"] - merged["bench_ret_21d"].fillna(0)
    else:
        feat_df["excess_ret_1d"] = feat_df["ret_1d"]
        feat_df["excess_ret_21d"] = feat_df["ret_21d"]

    # 11. Calendar Features
    dates = pd.to_datetime(feat_df["Date"])
    feat_df["day_of_week"] = dates.dt.dayofweek
    feat_df["month"] = dates.dt.month
    feat_df["quarter"] = dates.dt.quarter
    feat_df["is_month_end"] = dates.dt.is_month_end.astype(int)

    # 12. Create Supervised Forward Targets (Clearly segregated)
    for h in horizons:
        # Target: Return from t to t+h
        feat_df[f"target_ret_{h}d"] = (close.shift(-h) - close) / close
        # Direction: 1 if positive return, 0 if flat or negative
        feat_df[f"target_dir_{h}d"] = (feat_df[f"target_ret_{h}d"] > 0).astype(int)

    return feat_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns all engineered feature column names excluding Date, targets, and raw identifiers."""
    exclude = {
        "Date",
        "Ticker",
        "target_ret_5d",
        "target_ret_21d",
        "target_ret_63d",
        "target_dir_5d",
        "target_dir_21d",
        "target_dir_63d",
        "Close",
        "Open",
        "High",
        "Low",
        "Adj Close",
        "Volume",
    }
    return [col for col in df.columns if col not in exclude and not col.startswith("target_")]
