"""Backtest Engine for Model-Driven Trading Strategies with Realistic Costs."""

from typing import Any

import pandas as pd

from finsight.backtest.metrics import compute_performance_metrics


class BacktestEngine:
    """Simulates systematic trading strategies with realistic friction (costs & slippage)."""

    def __init__(
        self,
        initial_capital: float = 1000000.0,  # 10 Lakhs INR
        transaction_cost_bps: float = 10.0,  # 10 bps (0.10%)
        slippage_bps: float = 5.0,  # 5 bps (0.05%)
        risk_free_rate: float = 0.065,  # 6.5%
    ):
        self.initial_capital = initial_capital
        self.cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0  # Total drag per trade
        self.risk_free_rate = risk_free_rate

    def run_strategy(
        self,
        price_df: pd.DataFrame,
        signals: pd.Series,
        benchmark_col: str | None = None,
    ) -> dict[str, Any]:
        """Runs long/flat strategy simulation based on binary signals (1 = Long, 0 = Flat/Cash)."""
        df = price_df.copy().sort_values("Date").reset_index(drop=True)
        signals = signals.reindex(df.index).fillna(0).astype(int)

        # Asset daily return
        asset_ret = df["Close"].pct_change().fillna(0)

        # Position shifts: We trade at t Close, position active for return from t to t+1
        positions = signals.shift(1).fillna(0)

        # Calculate turnover: absolute change in position
        turnover = positions.diff().abs().fillna(0)
        costs = turnover * self.cost_rate

        # Daily strategy return
        strat_ret = (positions * asset_ret) - costs

        # Benchmark return (Buy and Hold)
        bench_ret = asset_ret.copy()

        # Cumulative equity curves
        equity_strat = self.initial_capital * (1 + strat_ret).cumprod()
        equity_bench = self.initial_capital * (1 + bench_ret).cumprod()

        # Drawdowns
        dd_strat = (equity_strat - equity_strat.cummax()) / equity_strat.cummax()
        dd_bench = (equity_bench - equity_bench.cummax()) / equity_bench.cummax()

        metrics = compute_performance_metrics(
            strategy_returns=strat_ret,
            benchmark_returns=bench_ret,
            risk_free_rate=self.risk_free_rate,
        )

        metrics["initial_capital"] = self.initial_capital
        metrics["final_equity_strat"] = round(float(equity_strat.iloc[-1]), 2)
        metrics["final_equity_bench"] = round(float(equity_bench.iloc[-1]), 2)
        metrics["total_trades"] = int(turnover.sum())
        metrics["annualized_turnover"] = round(float(turnover.mean() * 252), 2)

        curve_df = pd.DataFrame(
            {
                "Date": df["Date"].astype(str),
                "Strategy_Equity": equity_strat,
                "Benchmark_Equity": equity_bench,
                "Strategy_Drawdown": dd_strat,
                "Benchmark_Drawdown": dd_bench,
                "Position": positions,
            }
        )

        return {
            "metrics": metrics,
            "equity_curve": curve_df.to_dict(orient="records"),
            "daily_df": curve_df,
        }
