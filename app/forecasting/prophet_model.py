"""
prophet_model.py — Facebook Prophet Forecasting Model
======================================================
Uses Meta Prophet with weekly + yearly seasonality and a
simulated holiday regressor (Black Friday / year-end peaks).

SAP Context
-----------
Equivalent to SAP IBP's "Causal & ML Forecasting" module that
incorporates external regressors and holiday calendars.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Simulated holiday calendar (mirrors SAP holiday calendars in IBP)
# ------------------------------------------------------------------ #
def _build_holiday_df() -> pd.DataFrame:
    """
    Create a simple holiday DataFrame for Prophet.
    Covers 2022–2025 so the model can be used in predictions.
    """
    holidays = []
    for year in range(2022, 2026):
        holidays += [
            {"holiday": "black_friday",   "ds": f"{year}-11-25", "lower_window": -1, "upper_window": 3},
            {"holiday": "christmas",       "ds": f"{year}-12-25", "lower_window": -3, "upper_window": 1},
            {"holiday": "new_year",        "ds": f"{year}-01-01", "lower_window": 0,  "upper_window": 1},
            {"holiday": "independence_day","ds": f"{year}-07-04", "lower_window": 0,  "upper_window": 0},
        ]
    return pd.DataFrame(holidays)


HOLIDAY_DF = _build_holiday_df()


class ProphetModel:
    """
    Facebook Prophet wrapper.

    Usage
    -----
        model = ProphetModel()
        fitted = model.train(train_df)
        forecast = model.predict(fitted)
    """

    MODEL_NAME = "prophet"

    def __init__(self) -> None:
        self._fitted_model = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, train_df: pd.DataFrame):
        """
        Fit a Prophet model including weekly/yearly seasonality and holidays.

        Parameters
        ----------
        train_df : feature DataFrame from preprocessing.load_feature_data()
                   Must contain 'date' and 'units_sold' columns.

        Returns
        -------
        Fitted Prophet model object.

        Raises
        ------
        ImportError  if prophet is not installed.
        RuntimeError if fitting fails.
        """
        try:
            from prophet import Prophet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Prophet is not installed. Run: pip install prophet"
            ) from exc

        # --- Ensure cmdstanpy can find the CmdStan binary ---
        # In Docker with non-root user, the default ~/.cmdstan path is
        # not writable. Search common install locations and set explicitly.
        import os
        import glob
        _cmdstan_candidates = [
            os.environ.get("CMDSTAN", ""),           # env var if set
            "/tmp/cmdstan",                           # our manual install dir
        ]
        # Also search for versioned subdirs like /tmp/cmdstan/cmdstan-2.38.0
        for _base in ["/tmp/cmdstan", "/tmp"]:
            _cmdstan_candidates += glob.glob(f"{_base}/cmdstan-*")

        for _path in _cmdstan_candidates:
            if _path and os.path.isdir(_path):
                try:
                    import cmdstanpy as _csp
                    _csp.set_cmdstan_path(_path)
                    logger.info("[%s] CmdStan path set to: %s", self.MODEL_NAME, _path)
                    break
                except Exception:
                    continue

        # Prophet expects columns: ds (datetime), y (target)
        prophet_df = train_df[["date", "units_sold", "promotion_flag"]].copy()
        prophet_df = prophet_df.rename(columns={"date": "ds", "units_sold": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        logger.info(
            "[%s] Fitting Prophet on %d observations …", self.MODEL_NAME, len(prophet_df)
        )

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=HOLIDAY_DF,
                seasonality_mode="multiplicative",
                interval_width=0.95,
                changepoint_prior_scale=0.05,
            )
            # Add promotion as an additive extra regressor
            model.add_regressor("promotion_flag", mode="additive")

            # Suppress verbose Stan output
            import logging as _logging
            _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

            model.fit(prophet_df)
            self._fitted_model = model
            logger.info("[%s] Prophet fitting complete.", self.MODEL_NAME)
        except AttributeError:
            # Some prophet builds require explicit backend init
            try:
                import cmdstanpy  # noqa: F401
                cmdstanpy.install_cmdstan()
            except Exception:
                pass
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                interval_width=0.95,
                changepoint_prior_scale=0.05,
            )
            model.add_regressor("promotion_flag", mode="additive")
            model.fit(prophet_df)
            self._fitted_model = model
            logger.info("[%s] Prophet fitting complete (fallback init).", self.MODEL_NAME)
        except Exception as exc:
            logger.error("[%s] Prophet fitting failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"Prophet training failed: {exc}") from exc

        return model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        fitted_model=None,
        last_date: pd.Timestamp | None = None,
        horizon: int = None,
    ) -> List[float]:
        """
        Generate a multi-step ahead forecast.

        Parameters
        ----------
        fitted_model : optional; uses self._fitted_model if not provided
        last_date    : last observed date (used to build future df)
        horizon      : forecast steps, defaults to settings.FORECAST_HORIZON

        Returns
        -------
        List[float] of length `horizon`, clipped to ≥ 0.
        """
        if horizon is None:
            horizon = settings.FORECAST_HORIZON

        model = fitted_model or self._fitted_model
        if model is None:
            raise RuntimeError("No fitted model. Call train() first.")

        if last_date is None:
            # Infer from the last training date stored in Prophet
            last_date = model.history["ds"].max()

        future = model.make_future_dataframe(periods=horizon, freq="D")
        # Assume no active promotions in the forecast window
        future["promotion_flag"] = 0

        try:
            forecast = model.predict(future)
            # Extract only the future horizon rows
            preds = forecast["yhat"].values[-horizon:]
            preds = np.maximum(preds, 0.0)
            logger.info("[%s] Generated %d-step forecast.", self.MODEL_NAME, horizon)
            return preds.tolist()
        except Exception as exc:
            logger.error("[%s] Inference failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"Prophet inference failed: {exc}") from exc
