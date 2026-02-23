"""
schemas.py — Pydantic API Schemas
===================================
All FastAPI request/response models are defined here.
Pydantic v2 style with strict typing and documentation strings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# Health Check
# ============================================================

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])
    timestamp: datetime
    database: str = Field(..., examples=["connected"])
    best_model: Optional[str] = Field(None, examples=["xgboost"])


# ============================================================
# Forecast
# ============================================================

class ForecastRequest(BaseModel):
    product_id: str = Field(..., examples=["P001"])
    warehouse_id: str = Field(..., examples=["W001"])
    model: Optional[str] = Field(
        None,
        description="Model name to use. If omitted, the best model from the registry is used.",
        examples=["xgboost"],
    )


class DailyForecast(BaseModel):
    day: int = Field(..., description="Forecast day index (1-based)", examples=[1])
    forecast_units: float = Field(..., description="Predicted units sold", examples=[142.5])


class ForecastResponse(BaseModel):
    product_id: str
    warehouse_id: str
    model_used: str
    forecast_horizon_days: int = 30
    forecast: List[DailyForecast]


# ============================================================
# Training
# ============================================================

class ModelMetrics(BaseModel):
    MAE: float
    RMSE: float
    MAPE: float
    training_time_seconds: float
    inference_time_seconds: float


class TrainResponse(BaseModel):
    status: str = Field(..., examples=["success"])
    message: str
    best_model: str
    metrics: Dict[str, ModelMetrics]


# ============================================================
# Error
# ============================================================

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
