"""Probability Calibration and Reliability Curve Utilities."""

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Computes empirical calibration curve and Brier reliability score."""
    brier = float(brier_score_loss(y_true, y_prob))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    curve_data = [
        {"predicted_prob": float(p), "empirical_prob": float(t)}
        for p, t in zip(prob_pred, prob_true)
    ]

    # Maximum Calibration Error (MCE) and Expected Calibration Error (ECE)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    mce = float(np.max(np.abs(prob_true - prob_pred)))

    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "calibration_curve": curve_data,
        "is_well_calibrated": ece < 0.15,
    }
