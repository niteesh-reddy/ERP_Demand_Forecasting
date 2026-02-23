"""
model_registry.py — Model Persistence & Registry
==================================================
Provides a file-based model registry that:
  - Saves / loads trained model artefacts (joblib or torch)
  - Persists evaluation metrics as JSON
  - Tracks & retrieves the best-performing model (lowest MAPE)

Directory layout under MODELS_DIR
-----------------------------------
models/
  sarima/
    model.joblib
    metrics.json
  prophet/
    model.joblib
    metrics.json
  xgboost/
    model.joblib
    metrics.json
  lstm/
    model.keras
    scaler.joblib
    metrics.json
  tft/
    model.ckpt            (Lightning checkpoint)
    metrics.json
  best_model.json         ← {"best_model": "xgboost", "MAPE": 4.12}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from app.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _model_dir(model_name: str) -> Path:
    """Return the directory path for a given model, creating it if needed."""
    d = Path(settings.MODELS_DIR) / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metrics_path(model_name: str) -> Path:
    return _model_dir(model_name) / "metrics.json"


def _best_model_path() -> Path:
    return Path(settings.MODELS_DIR) / "best_model.json"


# ------------------------------------------------------------------ #
# Save / Load
# ------------------------------------------------------------------ #

def save_model(model_name: str, model_obj: Any, filename: str = "model.joblib") -> Path:
    """
    Persist a model object to disk.

    For Keras models the caller should pass filename='model.keras' and
    use model.save() before calling this (which saves via Keras' own API).
    For everything else (SARIMA, Prophet, XGBoost, scalers) joblib is used.

    Returns
    -------
    Path to the saved file.
    """
    dest = _model_dir(model_name) / filename

    if filename.endswith(".keras"):
        # Keras saves itself — model_obj.save() should already have been called
        logger.info("[%s] Keras model already saved at %s", model_name, dest)
        return dest

    try:
        joblib.dump(model_obj, dest, compress=3)
        logger.info("[%s] Model saved → %s", model_name, dest)
    except Exception as exc:
        logger.error("[%s] Failed to save model: %s", model_name, exc)
        raise

    return dest


def load_model(model_name: str, filename: str = "model.joblib") -> Any:
    """
    Load a previously persisted model from disk.

    Raises
    ------
    FileNotFoundError  if the model has not been trained yet.
    RuntimeError       if deserialisation fails.
    """
    path = _model_dir(model_name) / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}. "
            "Run POST /train first."
        )

    try:
        model = joblib.load(path)
        logger.info("[%s] Model loaded from %s", model_name, path)
        return model
    except Exception as exc:
        logger.error("[%s] Failed to load model: %s", model_name, exc)
        raise RuntimeError(f"Could not deserialise model '{model_name}': {exc}") from exc


def save_keras_model(model_name: str, keras_model: Any) -> Path:
    """
    Save a Keras model using Keras' native .keras format.
    Returns the path.
    """
    dest = _model_dir(model_name) / "model.keras"
    try:
        keras_model.save(str(dest))
        logger.info("[%s] Keras model saved → %s", model_name, dest)
    except Exception as exc:
        logger.error("[%s] Failed to save Keras model: %s", model_name, exc)
        raise
    return dest


def load_keras_model(model_name: str) -> Any:
    """Load a Keras model from the registry."""
    from tensorflow import keras  # type: ignore[import]
    path = _model_dir(model_name) / "model.keras"
    if not path.exists():
        raise FileNotFoundError(f"Keras model '{model_name}' not found at {path}.")
    try:
        model = keras.models.load_model(str(path))
        logger.info("[%s] Keras model loaded from %s", model_name, path)
        return model
    except Exception as exc:
        raise RuntimeError(f"Could not load Keras model '{model_name}': {exc}") from exc


def save_tft_checkpoint(model_name: str, trainer: Any) -> Path:
    """
    Save the best Lightning checkpoint path reference.
    PyTorch-Forecasting checkpoints are already saved by the trainer;
    we just record their path in metrics.json.
    """
    ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
    ref_path = _model_dir(model_name) / "checkpoint_path.txt"
    ref_path.write_text(ckpt)
    logger.info("[%s] TFT checkpoint path saved → %s (points to %s)", model_name, ref_path, ckpt)
    return ref_path


def load_tft_checkpoint_path(model_name: str) -> str:
    """Return the stored Lightning checkpoint path for TFT."""
    ref_path = _model_dir(model_name) / "checkpoint_path.txt"
    if not ref_path.exists():
        raise FileNotFoundError(f"TFT checkpoint reference not found for '{model_name}'.")
    return ref_path.read_text().strip()


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def save_metrics(model_name: str, metrics: Dict[str, Any]) -> None:
    """Persist evaluation metrics JSON for a model."""
    path = _metrics_path(model_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("[%s] Metrics saved → %s", model_name, path)
    except Exception as exc:
        logger.error("[%s] Failed to save metrics: %s", model_name, exc)
        raise


def load_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics for a model. Returns None if not found."""
    path = _metrics_path(model_name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("[%s] Could not load metrics: %s", model_name, exc)
        return None


# ------------------------------------------------------------------ #
# Best Model Selection
# ------------------------------------------------------------------ #

def update_best_model(evaluations: Dict[str, Dict[str, Any]]) -> str:
    """
    Determine the best model by lowest MAPE and write best_model.json.

    Parameters
    ----------
    evaluations : dict
        { model_name: { "MAE": ..., "RMSE": ..., "MAPE": ..., ... } }

    Returns
    -------
    Name of the best model.
    """
    if not evaluations:
        raise ValueError("evaluations dict is empty — nothing to compare.")

    best_name = min(evaluations, key=lambda k: evaluations[k].get("MAPE", float("inf")))
    best_mape = evaluations[best_name].get("MAPE", float("inf"))

    record = {"best_model": best_name, "MAPE": best_mape, "all_models": evaluations}
    path = _best_model_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("Best model updated → '%s' (MAPE=%.4f)", best_name, best_mape)
    except Exception as exc:
        logger.error("Failed to write best_model.json: %s", exc)
        raise

    return best_name


def get_best_model_name() -> str:
    """
    Return the name of the best model from the registry.

    Raises
    ------
    FileNotFoundError if training has not been run yet.
    """
    path = _best_model_path()
    if not path.exists():
        raise FileNotFoundError(
            "No trained models found in registry. Run POST /train first."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["best_model"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Corrupted best_model.json: {exc}") from exc


def list_all_metrics() -> Dict[str, Optional[Dict[str, Any]]]:
    """Return metrics for all known models (used by /health or admin endpoints)."""
    known_models = ["sarima", "prophet", "xgboost", "lstm", "tft"]
    return {name: load_metrics(name) for name in known_models}
