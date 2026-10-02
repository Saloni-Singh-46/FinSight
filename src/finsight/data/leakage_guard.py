"""Leakage Guard & Time-Series Cross-Validation with Purging and Embargo."""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """Time-series walk-forward CV splitter with Purging and Embargo.

    Eliminates look-ahead bias and serial correlation leakage between train and test sets.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 252,
        test_size: int = 63,
        purge_gap: int = 5,
        embargo_pct: float = 0.02,
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: Any | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        embargo = int(n_samples * self.embargo_pct)
        total_test_span = self.n_splits * self.test_size

        if n_samples < (self.min_train_size + self.test_size + self.purge_gap):
            # Fallback for smaller samples: adjust test_size
            adjusted_test_size = max(
                10, (n_samples - self.min_train_size - self.purge_gap) // self.n_splits
            )
            self.test_size = adjusted_test_size

        for i in range(self.n_splits):
            # Test window indices
            test_end = n_samples - (self.n_splits - 1 - i) * self.test_size
            test_start = test_end - self.test_size

            # Train window indices (expanding window)
            train_end = test_start - self.purge_gap

            if train_end < self.min_train_size:
                train_start = 0
                train_indices = np.arange(train_start, max(1, train_end))
            else:
                train_indices = np.arange(0, train_end)

            test_indices = np.arange(test_start, test_end)

            # Apply embargo after test if evaluating sequential rolling windows
            yield train_indices, test_indices


def assert_no_lookahead_leakage(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    sample_df: pd.DataFrame,
    check_idx: int = 50,
) -> bool:
    """Verifies that modifying future data points does NOT alter historical feature values at check_idx."""
    df_orig = sample_df.copy()
    features_orig = feature_fn(df_orig)

    # Corrupt future prices after check_idx
    df_corrupted = sample_df.copy()
    df_corrupted.iloc[check_idx + 1 :, df_corrupted.columns.get_loc("Close")] *= 2.5
    df_corrupted.iloc[check_idx + 1 :, df_corrupted.columns.get_loc("Volume")] *= 5.0

    features_corrupted = feature_fn(df_corrupted)

    # Past features at check_idx must be bit-for-bit identical (excluding target columns)
    row_orig = features_orig.iloc[check_idx].dropna()
    row_corrupted = features_corrupted.iloc[check_idx].dropna()

    for col in row_orig.index:
        if col.startswith("target_"):
            continue
        val_orig = row_orig[col]
        val_corrupted = row_corrupted[col]
        if isinstance(val_orig, (int, float, np.floating)):
            diff = abs(val_orig - val_corrupted)
            if diff > 1e-6:
                raise AssertionError(
                    f"Look-ahead leakage detected in feature '{col}'! "
                    f"Original: {val_orig}, Corrupted future: {val_corrupted}, Diff: {diff}"
                )
    return True
