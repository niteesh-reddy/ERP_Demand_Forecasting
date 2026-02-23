"""
main.py — FastAPI Application Entry Point
==========================================
Production-grade FastAPI microservice for ERP demand forecasting.

Endpoints
---------
GET  /health                       → service + DB status
POST /train                        → train all 5 models, update registry
GET  /forecast?product_id=&warehouse_id=&model=
                                   → 30-day demand forecast

Enterprise design decisions
----------------------------
- Structured logging via Python logging (JSON-compatible)
- All errors return consistent ErrorResponse schema
- /train runs synchronously (use a background task queue in production)
- Model loading is lazy — models are loaded from disk on demand
- CORS enabled for integration with SAP Fiori / BTP frontends
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import check_db_connection, get_db, init_db
from app.model_registry import get_best_model_name, load_model, load_metrics, list_all_metrics
from app.preprocessing import load_feature_data
from app.schemas import (
    DailyForecast,
    ErrorResponse,
    ForecastResponse,
    HealthResponse,
    TrainResponse,
    ModelMetrics,
)

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
settings.setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# FastAPI application
# ------------------------------------------------------------------ #
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Multi-model ERP demand forecasting microservice. "
        "Supports SARIMA, Prophet, XGBoost, LSTM, and TFT. "
        "Designed to integrate with SAP S/4HANA-style ERP systems."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow SAP BTP, Fiori, or any frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Startup
# ------------------------------------------------------------------ #

@app.on_event("startup")
async def startup_event() -> None:
    """Initialise DB tables on first start."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    try:
        init_db()
        logger.info("Database initialised successfully.")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        # Don't crash — /health will report the failure


# ------------------------------------------------------------------ #
# Global Exception Handler
# ------------------------------------------------------------------ #

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now(timezone.utc),
        ).model_dump(mode="json"),
    )


# ================================================================== #
# ENDPOINTS
# ================================================================== #

# ------------------------------------------------------------------ #
# GET /health
# ------------------------------------------------------------------ #

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Ops"],
)
async def health_check() -> HealthResponse:
    """
    Returns the current operational status of the service.

    Checks:
    - Database connectivity
    - Best model availability in registry
    """
    db_ok = check_db_connection()
    db_status = "connected" if db_ok else "unreachable"

    best_model: Optional[str] = None
    try:
        best_model = get_best_model_name()
    except FileNotFoundError:
        best_model = None  # Not trained yet — valid state

    status = "ok" if db_ok else "degraded"
    logger.info("/health → status=%s, db=%s, best_model=%s", status, db_status, best_model)

    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        database=db_status,
        best_model=best_model,
    )


# ------------------------------------------------------------------ #
# POST /train
# ------------------------------------------------------------------ #

@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Train all forecasting models",
    tags=["Training"],
)
async def train_models(
    product_id: str = Query(default="P001", description="Product ID to train on"),
    warehouse_id: str = Query(default="W001", description="Warehouse ID to train on"),
) -> TrainResponse:
    """
    Trains all five forecasting models (SARIMA, Prophet, XGBoost, LSTM, TFT)
    on the specified product-warehouse combination.

    - Evaluates each model on the held-out last 30 days of data
    - Updates the model registry with the best model (lowest MAPE)
    - Returns a full metrics comparison table

    **Warning:** This is a long-running operation (2–10 minutes depending on hardware).
    In production, this should be triggered via a job scheduler (e.g., SAP Data Intelligence).
    """
    logger.info("POST /train → product=%s, warehouse=%s", product_id, warehouse_id)

    try:
        # Import here to avoid loading all ML frameworks at startup
        from training.train_all_models import train_all
        evaluations = train_all(product_id=product_id, warehouse_id=warehouse_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("/train failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    try:
        best_model_name = get_best_model_name()
    except FileNotFoundError:
        best_model_name = "none"

    # Build metrics response — skip models that errored
    metrics_response = {}
    for name, m in evaluations.items():
        if m.get("MAPE", float("inf")) < float("inf"):
            metrics_response[name] = ModelMetrics(
                MAE=m["MAE"],
                RMSE=m["RMSE"],
                MAPE=m["MAPE"],
                training_time_seconds=m.get("training_time_seconds", 0.0),
                inference_time_seconds=m.get("inference_time_seconds", 0.0),
            )

    return TrainResponse(
        status="success",
        message=f"Training complete. {len(metrics_response)}/5 models trained successfully.",
        best_model=best_model_name,
        metrics=metrics_response,
    )


# ------------------------------------------------------------------ #
# GET /forecast
# ------------------------------------------------------------------ #

@app.get(
    "/forecast",
    response_model=ForecastResponse,
    summary="Generate 30-day demand forecast",
    tags=["Forecasting"],
)
async def get_forecast(
    product_id: str = Query(..., description="Product ID (e.g. P001)"),
    warehouse_id: str = Query(..., description="Warehouse ID (e.g. W001)"),
    model: Optional[str] = Query(
        default=None,
        description="Model to use: sarima | prophet | xgboost | lstm | tft. "
                    "Defaults to the best model in the registry.",
    ),
    db: Session = Depends(get_db),
) -> ForecastResponse:
    """
    Returns a 30-day daily demand forecast for the specified
    product-warehouse combination.

    - If `model` is not specified, the best-performing model from the
      registry (lowest MAPE) is used automatically.
    - Forecasts are clipped to ≥ 0 (demand cannot be negative).

    **SAP Integration Note:** This endpoint is designed to be called
    by SAP IBP or BTP via an OData-compatible HTTP request.
    """
    # --- Resolve model name ---
    if model is None:
        try:
            model_name = get_best_model_name()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="No trained models found. Run POST /train first.",
            )
    else:
        allowed = {"sarima", "prophet", "xgboost", "lstm", "tft"}
        if model.lower() not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model '{model}'. Allowed: {sorted(allowed)}",
            )
        model_name = model.lower()

    logger.info(
        "GET /forecast → product=%s, warehouse=%s, model=%s",
        product_id, warehouse_id, model_name,
    )

    # --- Load feature data ---
    try:
        feature_df = load_feature_data(product_id, warehouse_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Feature loading failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Feature loading failed: {exc}")

    # --- Dispatch to the correct model ---
    horizion = settings.FORECAST_HORIZON
    predictions: list[float] = []

    try:
        if model_name == "sarima":
            from app.forecasting.sarima_model import SarimaModel
            fitted = load_model("sarima", "model.joblib")
            m = SarimaModel()
            predictions = m.predict(fitted, horizon=horizion)

        elif model_name == "prophet":
            from app.forecasting.prophet_model import ProphetModel
            fitted = load_model("prophet", "model.joblib")
            m = ProphetModel()
            predictions = m.predict(fitted, horizon=horizion)

        elif model_name == "xgboost":
            from app.forecasting.xgboost_model import XGBoostModel
            regressor = load_model("xgboost", "model.joblib")
            scaler = load_model("xgboost", "scaler.joblib")
            m = XGBoostModel()
            predictions = m.predict(regressor, scaler, feature_df, horizon=horizion)

        elif model_name == "lstm":
            from app.forecasting.lstm_model import LSTMModel
            from app.model_registry import load_keras_model
            keras_model = load_keras_model("lstm")
            scaler = load_model("lstm", "scaler.joblib")
            m = LSTMModel()
            predictions = m.predict(keras_model, scaler, feature_df, horizon=horizion)

        elif model_name == "tft":
            from app.forecasting.tft_model import TFTModel
            import torch
            from pytorch_forecasting import TemporalFusionTransformer
            ckpt_ref_path = Path(settings.MODELS_DIR) / "tft" / "checkpoints" / "best.ckpt"
            if not ckpt_ref_path.exists():
                raise FileNotFoundError("TFT checkpoint not found. Run POST /train first.")
            tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_ref_path))
            m = TFTModel()
            predictions = m.predict(None, tft, feature_df, horizon=horizion)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Forecast generation failed [%s]: %s\n%s",
                     model_name, exc, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed for model '{model_name}': {exc}",
        )

    # --- Format response ---
    forecast_items = [
        DailyForecast(day=i + 1, forecast_units=round(v, 2))
        for i, v in enumerate(predictions)
    ]

    return ForecastResponse(
        product_id=product_id,
        warehouse_id=warehouse_id,
        model_used=model_name,
        forecast_horizon_days=horizion,
        forecast=forecast_items,
    )


# ------------------------------------------------------------------ #
# GET /models  (bonus: list available models & metrics)
# ------------------------------------------------------------------ #

@app.get(
    "/models",
    summary="List all models and their metrics",
    tags=["Registry"],
)
async def list_models() -> dict:
    """Returns training metrics for all models currently in the registry."""
    all_metrics = list_all_metrics()
    try:
        best = get_best_model_name()
    except FileNotFoundError:
        best = None

    return {
        "best_model": best,
        "models": all_metrics,
    }


# ------------------------------------------------------------------ #
# Needed for TFT path resolution in /forecast
# ------------------------------------------------------------------ #
from pathlib import Path  # noqa: E402 (already imported at top, this is IDE-safe)
