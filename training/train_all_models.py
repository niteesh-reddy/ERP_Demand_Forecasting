"""
train_all_models.py — Training Orchestration
=============================================
Trains all five forecasting models on the first product-warehouse
pair available in the database, evaluates on the held-out 30-day
test set, updates the model registry, and prints a comparison table.

Usage
-----
    python training/train_all_models.py

Environment
-----------
Set DATABASE_URL and MODELS_DIR in .env or environment before running.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import SessionLocal, init_db
from app.evaluation import Timer, compute_metrics, format_comparison_table
from app.model_registry import (
    save_model,
    save_metrics,
    update_best_model,
    save_keras_model,
    save_tft_checkpoint,
)
from app.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_feature_data,
    train_test_split,
)

settings.setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Pick a representative product-warehouse pair for training
# (In production, train per combination and aggregate or ensemble)
# ------------------------------------------------------------------ #
TRAIN_PRODUCT_ID = "P001"
TRAIN_WAREHOUSE_ID = "W001"


def _train_sarima(train_df, test_df) -> Dict[str, Any]:
    """Train & evaluate SARIMA."""
    from app.forecasting.sarima_model import SarimaModel
    import joblib

    model_obj = SarimaModel()
    series = SarimaModel.prepare_series(train_df)

    with Timer() as train_timer:
        fitted = model_obj.train(series)

    with Timer() as infer_timer:
        preds = model_obj.predict(fitted)

    y_true = test_df[TARGET_COLUMN].values
    metrics = compute_metrics(y_true, preds)
    metrics["training_time_seconds"] = train_timer.elapsed
    metrics["inference_time_seconds"] = infer_timer.elapsed

    # Persist
    save_model("sarima", fitted, filename="model.joblib")
    save_metrics("sarima", metrics)
    return metrics


def _train_prophet(train_df, test_df) -> Dict[str, Any]:
    """Train & evaluate Prophet."""
    from app.forecasting.prophet_model import ProphetModel

    model_obj = ProphetModel()

    with Timer() as train_timer:
        fitted = model_obj.train(train_df)

    with Timer() as infer_timer:
        preds = model_obj.predict(fitted)

    y_true = test_df[TARGET_COLUMN].values
    metrics = compute_metrics(y_true, preds)
    metrics["training_time_seconds"] = train_timer.elapsed
    metrics["inference_time_seconds"] = infer_timer.elapsed

    save_model("prophet", fitted, filename="model.joblib")
    save_metrics("prophet", metrics)
    return metrics


def _train_xgboost(train_df, test_df) -> Dict[str, Any]:
    """Train & evaluate XGBoost."""
    from app.forecasting.xgboost_model import XGBoostModel

    model_obj = XGBoostModel()

    with Timer() as train_timer:
        regressor, scaler = model_obj.train(train_df)

    with Timer() as infer_timer:
        preds = model_obj.predict(regressor, scaler, train_df)

    y_true = test_df[TARGET_COLUMN].values
    metrics = compute_metrics(y_true, preds)
    metrics["training_time_seconds"] = train_timer.elapsed
    metrics["inference_time_seconds"] = infer_timer.elapsed

    save_model("xgboost", regressor, filename="model.joblib")
    save_model("xgboost", scaler, filename="scaler.joblib")
    save_metrics("xgboost", metrics)
    return metrics


def _train_lstm(train_df, test_df) -> Dict[str, Any]:
    """Train & evaluate LSTM."""
    from app.forecasting.lstm_model import LSTMModel

    model_obj = LSTMModel()

    with Timer() as train_timer:
        keras_model, scaler = model_obj.train(train_df, epochs=30)

    with Timer() as infer_timer:
        preds = model_obj.predict(keras_model, scaler, train_df)

    y_true = test_df[TARGET_COLUMN].values
    metrics = compute_metrics(y_true, preds)
    metrics["training_time_seconds"] = train_timer.elapsed
    metrics["inference_time_seconds"] = infer_timer.elapsed

    save_keras_model("lstm", keras_model)
    save_model("lstm", scaler, filename="scaler.joblib")
    save_metrics("lstm", metrics)
    return metrics


def _train_tft(train_df, test_df) -> Dict[str, Any]:
    """Train & evaluate Temporal Fusion Transformer."""
    from app.forecasting.tft_model import TFTModel

    model_obj = TFTModel()

    with Timer() as train_timer:
        trainer, tft = model_obj.train(train_df, max_epochs=20)

    with Timer() as infer_timer:
        preds = model_obj.predict(trainer, tft, train_df)

    y_true = test_df[TARGET_COLUMN].values
    metrics = compute_metrics(y_true, preds)
    metrics["training_time_seconds"] = train_timer.elapsed
    metrics["inference_time_seconds"] = infer_timer.elapsed

    # Save checkpoint reference and dataset params
    try:
        save_tft_checkpoint("tft", trainer)
        model_obj.save_dataset_params(tft)
    except Exception as e:
        logger.warning("[tft] Could not save checkpoint reference: %s", e)

    save_metrics("tft", metrics)
    return metrics


# ------------------------------------------------------------------ #
# Main Orchestrator
# ------------------------------------------------------------------ #

def train_all(product_id: str = TRAIN_PRODUCT_ID,
              warehouse_id: str = TRAIN_WAREHOUSE_ID) -> Dict[str, Dict[str, Any]]:
    """
    Train all 5 models and return evaluation results dict.
    Each model is wrapped in a try/except so one failure doesn't
    block the others from training.
    """
    logger.info("=" * 60)
    logger.info("ERP Demand Forecasting — Training Pipeline")
    logger.info("Product: %s | Warehouse: %s", product_id, warehouse_id)
    logger.info("=" * 60)

    init_db()

    with SessionLocal() as session:
        logger.info("Loading feature data from database …")
        full_df = load_feature_data(product_id, warehouse_id, session)

    train_df, test_df = train_test_split(full_df)
    logger.info(
        "Train rows: %d | Test rows: %d", len(train_df), len(test_df)
    )

    evaluations: Dict[str, Dict[str, Any]] = {}

    # --- Train each model ---
    model_trainers = {
        "sarima": _train_sarima,
        "prophet": _train_prophet,
        "xgboost": _train_xgboost,
        "lstm": _train_lstm,
        "tft": _train_tft,
    }

    for name, trainer_fn in model_trainers.items():
        logger.info("-" * 40)
        logger.info("Training: %s …", name.upper())
        try:
            metrics = trainer_fn(train_df, test_df)
            evaluations[name] = metrics
            logger.info(
                "[%s] MAE=%.4f | RMSE=%.4f | MAPE=%.4f%%",
                name, metrics["MAE"], metrics["RMSE"], metrics["MAPE"],
            )
        except Exception as exc:
            logger.error("[%s] FAILED: %s", name, exc)
            logger.debug(traceback.format_exc())
            evaluations[name] = {
                "MAE": float("inf"), "RMSE": float("inf"),
                "MAPE": float("inf"),
                "training_time_seconds": 0.0,
                "inference_time_seconds": 0.0,
                "error": str(exc),
            }

    # --- Update registry ---
    successful = {k: v for k, v in evaluations.items() if v["MAPE"] < float("inf")}
    if successful:
        best = update_best_model(successful)
        logger.info("=" * 60)
        logger.info("BEST MODEL: %s", best.upper())
    else:
        logger.warning("No models trained successfully!")

    # --- Print comparison table ---
    print("\n")
    print(format_comparison_table(
        {k: v for k, v in evaluations.items() if v["MAPE"] < float("inf")}
    ))

    return evaluations


if __name__ == "__main__":
    train_all()
