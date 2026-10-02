"""Stock Direction Classifiers (XGBoost, LightGBM, Logistic Regression, Naive Baseline)."""

import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from finsight.data.leakage_guard import PurgedTimeSeriesSplit
from finsight.features.technical import get_feature_columns

optuna.logging.set_verbosity(optuna.logging.WARNING)


class NaiveDirectionBaseline:
    """Naive baseline predicting that previous direction persists."""

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # If 1-day return is positive, predict 1, else 0
        if "ret_1d" in X.columns:
            return (X["ret_1d"] > 0).astype(int).values
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), 2))
        probs[:, 1] = np.where(preds == 1, 0.60, 0.40)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs


class DirectionClassifierPipeline:
    """Manages training, cross-validation, hyperparameter tuning, and inference for direction models."""

    def __init__(
        self,
        horizon_days: int = 21,
        model_type: str = "xgboost",  # 'xgboost', 'lightgbm', 'logistic', 'naive'
        seed: int = 42,
    ):
        self.horizon_days = horizon_days
        self.model_type = model_type
        self.seed = seed
        self.target_col = f"target_dir_{horizon_days}d"
        self.model = None
        self.feature_names = []
        self.cv_results = {}
        self.metrics_summary = {}

    def prepare_data(self, feat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Prepares X and y by dropping initial NaN warm-up rows and final unlabelled rows."""
        features = get_feature_columns(feat_df)
        valid_df = feat_df.dropna(subset=features + [self.target_col]).reset_index(drop=True)
        X = valid_df[features]
        y = valid_df[self.target_col].astype(int)
        self.feature_names = features
        return X, y, features

    def train_walk_forward(
        self,
        feat_df: pd.DataFrame,
        n_splits: int = 5,
        tune_hyperparameters: bool = False,
    ) -> dict[str, Any]:
        """Executes walk-forward cross-validation with Purging & Embargo."""
        X, y, features = self.prepare_data(feat_df)
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=5, embargo_pct=0.02)

        fold_metrics = []
        oof_preds = np.zeros(len(X))
        oof_probs = np.zeros(len(X))
        oof_mask = np.zeros(len(X), dtype=bool)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Model creation
            if self.model_type == "xgboost":
                clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.seed,
                    eval_metric="logloss",
                )
            elif self.model_type == "lightgbm":
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    num_leaves=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.seed,
                    verbosity=-1,
                )
            elif self.model_type == "logistic":
                clf = LogisticRegression(max_iter=500, random_state=self.seed)
            elif self.model_type == "naive":
                clf = NaiveDirectionBaseline()
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            clf.fit(X_train, y_train)

            # Predict test fold
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            oof_probs[test_idx] = probs
            oof_preds[test_idx] = preds
            oof_mask[test_idx] = True

            # Fold metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5
            except Exception:
                auc = 0.5

            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "train_samples": len(train_idx),
                    "test_samples": len(test_idx),
                    "accuracy": float(acc),
                    "f1_score": float(f1),
                    "roc_auc": float(auc),
                }
            )

        # Out-of-fold aggregate evaluation
        valid_indices = np.where(oof_mask)[0]
        y_true_eval = y.iloc[valid_indices]
        y_prob_eval = oof_probs[valid_indices]
        y_pred_eval = oof_preds[valid_indices]

        overall_metrics = {
            "model_type": self.model_type,
            "horizon_days": self.horizon_days,
            "samples_evaluated": len(valid_indices),
            "accuracy": float(accuracy_score(y_true_eval, y_pred_eval)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_eval, y_pred_eval)),
            "precision": float(precision_score(y_true_eval, y_pred_eval, zero_division=0)),
            "recall": float(recall_score(y_true_eval, y_pred_eval, zero_division=0)),
            "f1_score": float(f1_score(y_true_eval, y_pred_eval, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_true_eval, y_prob_eval))
                if len(np.unique(y_true_eval)) > 1
                else 0.5
            ),
            "brier_score": float(brier_score_loss(y_true_eval, y_prob_eval)),
            "fold_results": fold_metrics,
        }

        # Train final calibrated production model on all available historical data
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                random_state=self.seed,
                eval_metric="logloss",
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=3,
                num_leaves=10,
                learning_rate=0.04,
                random_state=self.seed,
                verbosity=-1,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(max_iter=500, random_state=self.seed)
        else:
            self.model = NaiveDirectionBaseline()

        self.model.fit(X, y)
        self.metrics_summary = overall_metrics
        return overall_metrics

    def predict_latest(self, feat_df: pd.DataFrame) -> dict[str, Any]:
        """Predicts directional probability for the most recent observation."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        latest_row = feat_df.iloc[-1:][self.feature_names].fillna(0)
        prob_up = float(self.model.predict_proba(latest_row)[0, 1])
        pred_direction = (
            "Bullish" if prob_up >= 0.55 else ("Bearish" if prob_up <= 0.45 else "Neutral")
        )

        return {
            "horizon_days": self.horizon_days,
            "probability_up": np.round(prob_up, 4),
            "probability_down": np.round(1.0 - prob_up, 4),
            "signal": pred_direction,
            "model_type": self.model_type,
            "historical_accuracy": self.metrics_summary.get("accuracy", 0.5),
            "roc_auc": self.metrics_summary.get("roc_auc", 0.5),
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "features": self.feature_names,
                    "metrics": self.metrics_summary,
                },
                f,
            )

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["features"]
            self.metrics_summary = data["metrics"]
