"""Time Series Forecasting Baselines (Naive Last-Value, Seasonal Naive, ARIMA, XGBoost Regressor)."""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from finsight.features.technical import get_feature_columns


class ForecastingBaselinesEvaluator:
    """Evaluates transparent baselines against deep learning forecasters."""

    def __init__(self, horizon_days: int = 21):
        self.horizon_days = horizon_days

    def evaluate_all(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Evaluates Naive Last-Value, Seasonal Naive, Auto-Regressive, and XGBoost Regressors."""
        close = df["Close"].values
        n = len(close)
        if n < 100:
            return []

        # Out of sample split (last 20%)
        split_idx = int(n * 0.8)
        y_true_returns = []

        # Ground truth horizon returns
        for i in range(split_idx, n - self.horizon_days):
            y_true_returns.append((close[i + self.horizon_days] - close[i]) / close[i])

        y_true = np.array(y_true_returns)
        if len(y_true) < 10:
            return []

        results = []

        # 1. Naive Last-Value Baseline (Predicts 0% change from current price)
        naive_preds = np.zeros(len(y_true))
        results.append(self._compute_metrics("Naive Last-Value (Zero Return)", y_true, naive_preds))

        # 2. Seasonal Naive Baseline (Predicts previous horizon's return)
        seasonal_preds = []
        for i in range(split_idx, n - self.horizon_days):
            prev_ret = (close[i] - close[i - self.horizon_days]) / close[i - self.horizon_days]
            seasonal_preds.append(prev_ret)
        results.append(
            self._compute_metrics("Seasonal Naive (Lagged Cycle)", y_true, np.array(seasonal_preds))
        )

        # 3. Simple Auto-Regressive / Moving Average Drift
        ar_preds = []
        for i in range(split_idx, n - self.horizon_days):
            hist_drift = np.mean(np.diff(np.log(close[max(0, i - 63) : i + 1]))) * self.horizon_days
            ar_preds.append(hist_drift)
        results.append(
            self._compute_metrics("ARIMA / Historical Drift", y_true, np.array(ar_preds))
        )

        # 4. XGBoost Regressor Baseline
        try:
            from finsight.features.technical import compute_technical_features

            feat_df = compute_technical_features(df, horizons=[self.horizon_days])
            features = get_feature_columns(feat_df)
            valid_df = feat_df.dropna(
                subset=features + [f"target_ret_{self.horizon_days}d"]
            ).reset_index(drop=True)

            X = valid_df[features]
            y = valid_df[f"target_ret_{self.horizon_days}d"]

            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

            reg = xgb.XGBRegressor(
                n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42
            )
            reg.fit(X_train, y_train)
            xgb_preds = reg.predict(X_test)

            results.append(self._compute_metrics("XGBoost Regressor", y_test.values, xgb_preds))
        except Exception:
            pass

        return results

    def _compute_metrics(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        dir_acc = float(np.mean((y_pred > 0) == (y_true > 0)) * 100)
        return {
            "model_name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "directional_accuracy_pct": round(dir_acc, 2),
        }
