"""
config.py — Application Configuration
======================================
Centralised settings loaded from environment variables via Pydantic-Settings.
All other modules import `settings` from here — no magic strings elsewhere.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Enterprise application settings.

    Values are read from environment variables (case-insensitive).
    A `.env` file in the project root is loaded automatically when
    running outside Docker.
    """

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "postgresql://erp_user:erp_pass@localhost:5432/erp_forecast"

    # ------------------------------------------------------------------ #
    # Model storage
    # ------------------------------------------------------------------ #
    MODELS_DIR: str = "./models"

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Application metadata
    # ------------------------------------------------------------------ #
    APP_VERSION: str = "1.0.0"
    APP_NAME: str = "ERP Demand Forecasting Microservice"

    # ------------------------------------------------------------------ #
    # Forecasting parameters (business rules)
    # ------------------------------------------------------------------ #
    FORECAST_HORIZON: int = 30   # days ahead to predict
    SEQUENCE_LENGTH: int = 30    # look-back window for sequence models
    TEST_DAYS: int = 30          # held-out days used for evaluation

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def setup_logging(self) -> None:
        """Configure root logger based on LOG_LEVEL setting."""
        numeric_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Using lru_cache avoids re-reading the .env file on every import.
    """
    s = Settings()
    # Ensure MODELS_DIR exists on first access
    os.makedirs(s.MODELS_DIR, exist_ok=True)
    return s


# Module-level singleton — import this everywhere
settings: Settings = get_settings()
