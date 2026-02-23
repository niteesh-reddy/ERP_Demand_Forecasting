"""
evaluation.py — Model Evaluation Utilities
==========================================
Provides standardized metric computation and a timing decorator
used by all five forecasting models to ensure fair comparison.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Core Metric Functions
# ============================================================

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Small epsilon prevents division-by-zero on near-zero actuals.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE for a pair of actual / predicted arrays.

    Parameters
    ----------
    y_true : np.ndarray  — ground truth (shape [n])
    y_pred : np.ndarray  — model predictions (shape [n])

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE (all floats, rounded to 4 dp)

    Raises
    ------
    ValueError if arrays have different lengths or contain NaN/inf.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must not contain NaN or Inf values.")

    return {
        "MAE": round(_mae(y_true, y_pred), 4),
        "RMSE": round(_rmse(y_true, y_pred), 4),
        "MAPE": round(_mape(y_true, y_pred), 4),
    }


# ============================================================
# Timing Utilities
# ============================================================

class Timer:
    """
    Context manager that measures elapsed wall-clock time.

    Usage::

        with Timer() as t:
            model.fit(X, y)
        print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = round(time.perf_counter() - self._start, 4)


def timed(fn: Callable) -> Callable:
    """
    Decorator that wraps a function and returns (result, elapsed_seconds).

    Usage::

        @timed
        def train(data):
            ...
            return model

        model, elapsed = train(data)
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 4)
        logger.debug("'%s' completed in %.4f s", fn.__name__, elapsed)
        return result, elapsed
    return wrapper


# ============================================================
# Comparison Formatter
# ============================================================

def format_comparison_table(evaluations: Dict[str, Dict[str, float]]) -> str:
    """
    Pretty-print a model comparison table to the console.

    Parameters
    ----------
    evaluations : dict
        { model_name: { "MAE": ..., "RMSE": ..., "MAPE": ...,
                        "training_time_seconds": ...,
                        "inference_time_seconds": ... } }
    """
    header = f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'Train(s)':>10} {'Infer(s)':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for model_name, m in sorted(evaluations.items(), key=lambda kv: kv[1].get("MAPE", 9999)):
        lines.append(
            f"{model_name:<20} "
            f"{m.get('MAE', 0):>10.4f} "
            f"{m.get('RMSE', 0):>10.4f} "
            f"{m.get('MAPE', 0):>10.4f} "
            f"{m.get('training_time_seconds', 0):>10.4f} "
            f"{m.get('inference_time_seconds', 0):>10.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)
