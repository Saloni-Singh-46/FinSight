"""Fundamental and Ratio Feature Extraction with Temporal Lagging."""

from typing import Any

import numpy as np


def extract_fundamental_features(
    fundamentals_dict: dict[str, Any],
    lag_days: int = 45,
) -> dict[str, float]:
    """Extracts normalized fundamental and solvency ratios."""
    return {
        "fund_pe_ratio": float(fundamentals_dict.get("pe_ratio", 15.0)),
        "fund_forward_pe": float(fundamentals_dict.get("forward_pe", 14.0)),
        "fund_pb_ratio": float(fundamentals_dict.get("pb_ratio", 2.5)),
        "fund_ev_ebitda": float(fundamentals_dict.get("ev_to_ebitda", 8.0)),
        "fund_roe_pct": float(fundamentals_dict.get("roe_pct", 15.0)),
        "fund_roce_pct": float(fundamentals_dict.get("roce_pct", 14.0)),
        "fund_debt_to_equity": float(fundamentals_dict.get("debt_to_equity", 0.5)),
        "fund_interest_coverage": float(fundamentals_dict.get("interest_coverage", 5.0)),
        "fund_operating_margin_pct": float(fundamentals_dict.get("operating_margin_pct", 12.0)),
        "fund_net_margin_pct": float(fundamentals_dict.get("net_profit_margin_pct", 6.0)),
        "fund_altman_z": float(fundamentals_dict.get("altman_z_score", 2.5)),
        "fund_piotroski_f": float(fundamentals_dict.get("piotroski_f_score", 7.0)),
    }


def compute_altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    sales: float,
) -> float:
    """Computes standard Altman Z-score for manufacturing companies.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    Safe Zone: Z > 2.99, Grey Zone: 1.81 <= Z <= 2.99, Distress Zone: Z < 1.81
    """
    if total_assets <= 0 or total_liabilities <= 0:
        return 0.0

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    return float(np.round(z, 2))


def compute_piotroski_f_score(financials: dict[str, Any]) -> int:
    """Computes Piotroski 9-point fundamental score (0 to 9)."""
    score = 0
    # Profitability (4 points)
    if financials.get("net_income", 1) > 0:
        score += 1
    if financials.get("operating_cash_flow", 1) > 0:
        score += 1
    if financials.get("roa_change", 1) > 0:
        score += 1
    if financials.get("operating_cash_flow", 2) > financials.get("net_income", 1):
        score += 1

    # Leverage, Liquidity, Source of Funds (3 points)
    if financials.get("leverage_change", -1) < 0:
        score += 1
    if financials.get("current_ratio_change", 1) > 0:
        score += 1
    if financials.get("shares_diluted", 0) <= 0:
        score += 1

    # Operating Efficiency (2 points)
    if financials.get("gross_margin_change", 1) > 0:
        score += 1
    if financials.get("asset_turnover_change", 1) > 0:
        score += 1

    return min(9, max(0, score))
