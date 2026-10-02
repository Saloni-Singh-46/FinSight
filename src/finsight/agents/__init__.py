"""FinSight Multi-Agent System package."""

from finsight.agents.critic_agent import CriticAgent
from finsight.agents.data_agent import DataAgent
from finsight.agents.planner import PlannerAgent
from finsight.agents.quant_agent import QuantAgent
from finsight.agents.rag_agent import FilingsRAGAgent
from finsight.agents.sentiment_agent import SentimentAgent
from finsight.agents.writer_agent import WriterAgent

__all__ = [
    "CriticAgent",
    "DataAgent",
    "FilingsRAGAgent",
    "PlannerAgent",
    "QuantAgent",
    "SentimentAgent",
    "WriterAgent",
]
