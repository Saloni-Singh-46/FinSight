"""Financial Backtest Performance Metrics & Statistical Sanity Checks."""

from typing import Any

import numpy as np
import pandas as pd


def compute_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.065,  # 6.5% risk-free rate proxy for India
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Computes institutional backtest metrics with bootstrap confidence intervals."""
    strat_ret = strategy_returns.dropna()
    bench_ret = benchmark_returns.reindex(strat_ret.index).dropna()

    if len(strat_ret) < 10:
        return {"error": "Insufficient return series length."}

    # Cumulative returns
    strat_cum = (1 + strat_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()

    n_years = len(strat_ret) / periods_per_year
    total_strat_return = float(strat_cum.iloc[-1] - 1.0)
    total_bench_return = float(bench_cum.iloc[-1] - 1.0)

    # CAGR
    cagr_strat = float((strat_cum.iloc[-1]) ** (1.0 / max(0.1, n_years)) - 1.0)
    cagr_bench = float((bench_cum.iloc[-1]) ** (1.0 / max(0.1, n_years)) - 1.0)

    # Volatility
    vol_strat = float(strat_ret.std() * np.sqrt(periods_per_year))
    vol_bench = float(bench_ret.std() * np.sqrt(periods_per_year))

    # Daily risk free
    daily_rf = (1 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_strat = strat_ret - daily_rf
    excess_bench = bench_ret - daily_rf

    # Sharpe Ratio
    sharpe_strat = float(
        (excess_strat.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(periods_per_year)
    )
    sharpe_bench = float(
        (excess_bench.mean() / (bench_ret.std() + 1e-9)) * np.sqrt(periods_per_year)
    )

    # Sortino Ratio (Downside deviation only)
    downside_strat = strat_ret[strat_ret < daily_rf]
    downside_std_strat = (
        float(downside_strat.std() * np.sqrt(periods_per_year)) if len(downside_strat) > 1 else 1e-9
    )
    sortino_strat = (
        float((cagr_strat - risk_free_rate) / downside_std_strat)
        if downside_std_strat > 1e-9
        else 0.0
    )

    # Max Drawdown & Drawdown Series
    running_max = strat_cum.cummax()
    drawdown_series = (strat_cum - running_max) / running_max
    max_drawdown_strat = float(drawdown_series.min())

    bench_running_max = bench_cum.cummax()
    bench_dd_series = (bench_cum - bench_running_max) / bench_running_max
    max_drawdown_bench = float(bench_dd_series.min())

    # Hit Rate / Win Rate on non-zero trading days
    active_days = strat_ret[strat_ret != 0]
    win_days = strat_ret[strat_ret > 0]
    hit_rate = float(len(win_days) / len(active_days)) if len(active_days) > 0 else 0.0

    # Bootstrap Confidence Interval for Sharpe Ratio (1,000 iterations)
    np.random.seed(42)
    boot_sharpes = []
    n_pts = len(strat_ret)
    for _ in range(1000):
        sample_indices = np.random.choice(n_pts, size=n_pts, replace=True)
        boot_sample = strat_ret.iloc[sample_indices]
        sample_excess = boot_sample - daily_rf
        sample_std = boot_sample.std()
        if sample_std > 1e-9:
            boot_sharpes.append((sample_excess.mean() / sample_std) * np.sqrt(periods_per_year))

    sharpe_ci_lower = float(np.percentile(boot_sharpes, 2.5)) if boot_sharpes else sharpe_strat
    sharpe_ci_upper = float(np.percentile(boot_sharpes, 97.5)) if boot_sharpes else sharpe_strat

    return {
        "cagr_pct": round(cagr_strat * 100, 2),
        "benchmark_cagr_pct": round(cagr_bench * 100, 2),
        "annualized_volatility_pct": round(vol_strat * 100, 2),
        "benchmark_volatility_pct": round(vol_bench * 100, 2),
        "sharpe_ratio": round(sharpe_strat, 2),
        "benchmark_sharpe": round(sharpe_bench, 2),
        "sharpe_95_ci": [round(sharpe_ci_lower, 2), round(sharpe_ci_upper, 2)],
        "sortino_ratio": round(sortino_strat, 2),
        "max_drawdown_pct": round(max_drawdown_strat * 100, 2),
        "benchmark_max_drawdown_pct": round(max_drawdown_bench * 100, 2),
        "hit_rate_pct": round(hit_rate * 100, 2),
        "total_return_pct": round(total_strat_return * 100, 2),
        "benchmark_total_return_pct": round(total_bench_return * 100, 2),
    }
