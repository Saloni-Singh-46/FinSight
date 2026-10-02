"""Configuration and Logging Utilities for FinSight."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="FinSight", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3:8b", alias="OLLAMA_MODEL")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    ui_port: int = Field(default=8501, alias="UI_PORT")
    enable_live_fetch: bool = Field(default=False, alias="ENABLE_LIVE_FETCH")
    cache_dir: str = Field(default="data/cache", alias="CACHE_DIR")
    dataset_dir: str = Field(default="Datasets", alias="DATASET_DIR")
    processed_dir: str = Field(default="data/processed", alias="PROCESSED_DIR")
    vectorstore_dir: str = Field(default="data/vectorstore", alias="VECTORSTORE_DIR")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")


def load_yaml_config(file_path: str) -> dict[str, Any]:
    p = Path(file_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logger(name: str = "finsight", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# Global instances
settings = AppSettings()
logger = setup_logger()
