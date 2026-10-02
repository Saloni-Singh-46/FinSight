"""Data Loaders for Financial Time Series, Fundamentals, Macro, and NLP Datasets."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from finsight.utils.config import logger


class FinancialDataLoader:
    """Offline-first loader with optional live caching for equity prices, fundamentals, and macro."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        raw_dataset_dir: str = "Datasets",
        cache_dir: str = "data/cache",
    ):
        self.processed_dir = Path(processed_dir)
        self.raw_dataset_dir = Path(raw_dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_price_history(self, ticker: str = "TATAMOTORS.NS") -> pd.DataFrame:
        """Load historical daily OHLCV dataframe for a given ticker."""
        clean_symbol = ticker.replace("^", "").replace(".NS", "").replace(".BO", "")
        file_path = self.processed_dir / f"{clean_symbol}_daily.csv"

        if not file_path.exists():
            # Try alternate names or case-insensitive search
            candidates = list(self.processed_dir.glob(f"*{clean_symbol}*.csv"))
            if candidates:
                file_path = candidates[0]
            else:
                logger.warning(
                    f"Price history file not found for {ticker} at {file_path}. Falling back to TATAMOTORS."
                )
                file_path = self.processed_dir / "TATAMOTORS_daily.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Cannot find price history at {file_path}. Please run seed script first."
            )

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def load_benchmark_history(self, benchmark_ticker: str = "^NSEI") -> pd.DataFrame:
        """Load benchmark index (e.g. NIFTY 50) history."""
        return self.load_price_history(benchmark_ticker)

    def load_fundamentals(self, ticker: str = "TATAMOTORS.NS") -> dict[str, Any]:
        """Load company balance sheet, valuation ratios, and quarterly financials."""
        file_path = self.processed_dir / "company_fundamentals.json"
        if not file_path.exists():
            logger.warning(f"Fundamentals file not found at {file_path}.")
            return {}

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if ticker in data:
            return data[ticker]
        elif "TATAMOTORS.NS" in data:
            return data["TATAMOTORS.NS"]
        return next(iter(data.values()), {})

    def load_peers(self, ticker: str = "TATAMOTORS.NS") -> list[dict[str, Any]]:
        """Load peer comparison metrics for a company."""
        fundamentals = self.load_fundamentals(ticker)
        return fundamentals.get("peers", [])

    def load_sentiment_dataset(self) -> pd.DataFrame:
        """Load labeled sentiment training data from Datasets/sent_train.csv."""
        path = self.raw_dataset_dir / "sent_train.csv"
        if not path.exists():
            # Fallback path check
            path = Path("datasets/sent_train.csv")

        if path.exists():
            df = pd.read_csv(path)
            # Map numeric labels: 0: Neutral/Negative, 1: Positive, 2: Negative
            return df
        logger.warning(f"Sentiment dataset not found at {path}.")
        return pd.DataFrame(columns=["text", "label"])

    def load_tatqa_dataset(self) -> list[dict[str, Any]]:
        """Load TAT-QA financial table and text Q&A dataset."""
        path = self.raw_dataset_dir / "tatqa_dataset_train.json"
        if not path.exists():
            path = Path("datasets/tatqa_dataset_train.json")

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_german_credit_data(self) -> pd.DataFrame:
        """Load Statlog German Credit data for credit risk modeling."""
        path = self.raw_dataset_dir / "statlog+german+credit+data" / "german.data-numeric"
        if not path.exists():
            path = Path("datasets/statlog+german+credit+data/german.data-numeric")

        if path.exists():
            # 24 numeric features + 1 target (1 = Good, 2 = Bad)
            cols = [f"f_{i}" for i in range(1, 25)] + ["credit_risk_target"]
            df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
            # Convert target to binary: 0 = Bad (Default), 1 = Good (Non-default)
            df["target"] = (df["credit_risk_target"] == 1).astype(int)
            return df
        return pd.DataFrame()

    def load_macro_rates(self) -> pd.DataFrame:
        """Load SOFR or reference interest rates."""
        path = self.raw_dataset_dir / "SOFR90DAYAVG.csv"
        if not path.exists():
            path = Path("datasets/SOFR90DAYAVG.csv")
        if path.exists():
            df = pd.read_csv(path)
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            return df.sort_values("observation_date").reset_index(drop=True)
        return pd.DataFrame()
