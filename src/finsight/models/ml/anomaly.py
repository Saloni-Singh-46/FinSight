"""Financial Time-Series Anomaly Detection (Isolation Forest & Rolling MAD)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class MarketAnomalyDetector:
    """Detects volatility, volume, and price shocks using Isolation Forest and Rolling MAD."""

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits anomaly detector on price/volume features and flags unusual market regimes."""
        data = df.copy()

        # Anomaly feature set: 1d returns, normalized volume, ATR, rolling volatility
        if "ret_1d" not in data.columns:
            data["ret_1d"] = data["Close"].pct_change()
        if "vol_ratio_20" not in data.columns:
            data["vol_ratio_20"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)

        feat_cols = ["ret_1d", "vol_ratio_20"]
        if "vol_21d" in data.columns:
            feat_cols.append("vol_21d")
        if "natr_14" in data.columns:
            feat_cols.append("natr_14")

        clean_subset = data[feat_cols].dropna()
        if len(clean_subset) < 30:
            data["is_anomaly"] = 0
            data["anomaly_score"] = 0.0
            data["anomaly_type"] = "Normal"
            return data

        # Isolation forest scores (-1 = anomaly, 1 = normal)
        self.iso_forest.fit(clean_subset)
        raw_preds = self.iso_forest.predict(clean_subset)
        scores = -self.iso_forest.score_samples(clean_subset)  # higher = more anomalous

        # Rolling MAD Baseline (Z-score on returns)
        ret_median = clean_subset["ret_1d"].rolling(20, min_periods=5).median()
        mad = (clean_subset["ret_1d"] - ret_median).abs().rolling(20, min_periods=5).median()
        mad_z = 0.6745 * (clean_subset["ret_1d"] - ret_median).abs() / (mad + 1e-9)

        # Merge anomaly flags back to full dataframe
        data["is_anomaly"] = 0
        data["anomaly_score"] = 0.0
        data["anomaly_type"] = "Normal"

        anom_mask = (raw_preds == -1) | (mad_z > 3.5)
        clean_indices = clean_subset.index[anom_mask]

        data.loc[clean_indices, "is_anomaly"] = 1
        data.loc[clean_subset.index, "anomaly_score"] = np.round(scores, 4)

        # Assign descriptive anomaly type
        for idx in clean_indices:
            ret = data.loc[idx, "ret_1d"]
            vol = data.loc[idx, "vol_ratio_20"]
            if ret < -0.04 and vol > 2.0:
                data.loc[idx, "anomaly_type"] = "High Volume Sell-off / Crash"
            elif ret > 0.04 and vol > 2.0:
                data.loc[idx, "anomaly_type"] = "Volume Breakout / Rally"
            elif vol > 3.0:
                data.loc[idx, "anomaly_type"] = "Unusual Volume Spike"
            elif abs(ret) > 0.05:
                data.loc[idx, "anomaly_type"] = "Severe Price Jump"
            else:
                data.loc[idx, "anomaly_type"] = "Statistical Outlier"

        return data

    def get_recent_anomalies(self, anom_df: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
        """Extracts list of the most recent anomaly events."""
        flagged = anom_df[anom_df["is_anomaly"] == 1].sort_values("Date", ascending=False)
        records = []
        for _, row in flagged.head(limit).iterrows():
            records.append(
                {
                    "date": (
                        str(row["Date"]).split("T")[0]
                        if "T" in str(row["Date"])
                        else str(row["Date"])
                    ),
                    "close": float(row["Close"]),
                    "daily_return_pct": float(np.round(row.get("ret_1d", 0) * 100, 2)),
                    "volume_multiple": float(np.round(row.get("vol_ratio_20", 1.0), 2)),
                    "anomaly_type": row["anomaly_type"],
                    "score": float(row["anomaly_score"]),
                }
            )
        return records
