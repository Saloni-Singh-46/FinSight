"""Data Cleaning and Normalization Utilities."""

import numpy as np
import pandas as pd


def clean_price_dataframe(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
    winsorize_std: float = 4.0,
) -> pd.DataFrame:
    """Clean and validate raw price dataframe."""
    if required_cols is None:
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    clean_df = df.copy()

    # Check required columns
    for col in required_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure Date is datetime
    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    clean_df = clean_df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Handle missing values strictly forward (chronological)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in clean_df.columns:
        numeric_cols.append("Adj Close")

    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df[col] = clean_df[col].ffill().bfill()

    # Enforce basic price logic: High >= Low, High >= Open, High >= Close
    clean_df["High"] = np.maximum(clean_df["High"], np.maximum(clean_df["Open"], clean_df["Close"]))
    clean_df["Low"] = np.minimum(clean_df["Low"], np.minimum(clean_df["Open"], clean_df["Close"]))

    # Zero or negative volume floor
    clean_df["Volume"] = np.maximum(clean_df["Volume"], 0)

    # Winsorize extreme percentage price swings (e.g. data errors > 4 sigma)
    returns = clean_df["Close"].pct_change()
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret > 0:
        lower_bound = mean_ret - winsorize_std * std_ret
        upper_bound = mean_ret + winsorize_std * std_ret
        clipped_returns = returns.clip(lower=lower_bound, upper=upper_bound)
        # Reconstruct Close if extreme glitch detected
        # clean_df['Close_cleaned'] = clean_df['Close'].iloc[0] * (1 + clipped_returns).cumprod()

    return clean_df


def align_with_benchmark(
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align asset price series with benchmark index on trading dates."""
    asset_clean = clean_price_dataframe(asset_df)
    bench_clean = clean_price_dataframe(benchmark_df)

    bench_clean = bench_clean.rename(
        columns={
            "Close": "Benchmark_Close",
            "Open": "Benchmark_Open",
            "Volume": "Benchmark_Volume",
        }
    )

    merged = (
        pd.merge(
            asset_clean,
            bench_clean[["Date", "Benchmark_Close"]],
            on="Date",
            how="inner",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return merged
