"""Sentiment Time Series & Momentum Features."""

import pandas as pd


def compute_sentiment_time_series_features(
    sentiment_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "sentiment_score",
    half_life_days: float = 14.0,
) -> pd.DataFrame:
    """Aggregates raw sentiment into daily volume-weighted and recency-decayed signals."""
    if sentiment_df.empty:
        return pd.DataFrame()

    df = sentiment_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Daily aggregation
    daily = (
        df.groupby(date_col)
        .agg(
            sent_mean=(score_col, "mean"),
            sent_pos_count=(score_col, lambda x: (x > 0.1).sum()),
            sent_neg_count=(score_col, lambda x: (x < -0.1).sum()),
            sent_volume=(score_col, "count"),
        )
        .reset_index()
    )

    daily["sent_pos_ratio"] = daily["sent_pos_count"] / (daily["sent_volume"] + 1e-9)
    daily["sent_neg_ratio"] = daily["sent_neg_count"] / (daily["sent_volume"] + 1e-9)
    daily["sent_net_score"] = daily["sent_pos_ratio"] - daily["sent_neg_ratio"]

    # Exponential recency decay
    daily["sent_ema_7d"] = daily["sent_mean"].ewm(span=7).mean()
    daily["sent_ema_30d"] = daily["sent_mean"].ewm(span=30).mean()
    daily["sent_momentum"] = daily["sent_ema_7d"] - daily["sent_ema_30d"]

    return daily
