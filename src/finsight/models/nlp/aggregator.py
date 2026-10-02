"""Financial News Sentiment Aggregator with Exponential Recency Decay."""

from typing import Any

import numpy as np
import pandas as pd

from finsight.models.nlp.finbert import FinancialSentimentClassifier


class SentimentAggregator:
    """Aggregates itemized news items into volume-weighted, recency-decayed sentiment indicators."""

    def __init__(
        self, classifier: FinancialSentimentClassifier | None = None, half_life_days: float = 14.0
    ):
        self.classifier = classifier or FinancialSentimentClassifier()
        self.half_life_days = half_life_days

    def aggregate_news_items(
        self,
        news_items: list[dict[str, Any]],
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        """Processes and aggregates news items."""
        if not news_items:
            return {
                "overall_sentiment": "Neutral",
                "net_sentiment_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "top_positive_headlines": [],
                "top_negative_headlines": [],
                "sentiment_trend": [],
            }

        df_items = []
        ref_dt = pd.to_datetime(reference_date) if reference_date else pd.Timestamp.now()

        for item in news_items:
            title = item.get("title", item.get("headline", item.get("text", "")))
            date_val = item.get("date", item.get("published_at", str(ref_dt.date())))
            pred = self.classifier.predict_single(title)

            dt = pd.to_datetime(date_val)
            age_days = max(0.0, (ref_dt - dt).total_seconds() / 86400.0)
            recency_weight = float(np.exp(-np.log(2) * age_days / self.half_life_days))

            df_items.append(
                {
                    "title": title,
                    "date": str(dt.date()),
                    "label": pred["label"],
                    "score": pred["sentiment_score"],
                    "weight": recency_weight,
                    "source": item.get("source", "Financial News Feed"),
                }
            )

        df = pd.DataFrame(df_items)

        pos_items = df[df["score"] > 0.1].sort_values("score", ascending=False)
        neg_items = df[df["score"] < -0.1].sort_values("score", ascending=True)

        weighted_score = float(np.sum(df["score"] * df["weight"]) / (np.sum(df["weight"]) + 1e-9))

        pos_count = int((df["label"] == "positive").sum())
        neg_count = int((df["label"] == "negative").sum())
        neu_count = int((df["label"] == "neutral").sum())

        if weighted_score > 0.15:
            overall = "Bullish / Positive"
        elif weighted_score < -0.15:
            overall = "Bearish / Negative"
        else:
            overall = "Neutral"

        # Daily trend
        daily_trend = (
            df.groupby("date")
            .agg(
                avg_score=("score", "mean"),
                article_count=("score", "count"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

        return {
            "overall_sentiment": overall,
            "net_sentiment_score": round(weighted_score, 4),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "total_articles_scored": len(df),
            "top_positive_headlines": pos_items.head(3).to_dict(orient="records"),
            "top_negative_headlines": neg_items.head(3).to_dict(orient="records"),
            "sentiment_trend": daily_trend,
        }
