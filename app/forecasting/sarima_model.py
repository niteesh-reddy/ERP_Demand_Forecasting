"""
sarima_model.py — SARIMA Forecasting Model
===========================================
Uses statsmodels SARIMAX to fit a seasonal ARIMA model per
product-warehouse time series.

SAP Context
-----------
Equivalent to SAP IBP's statistical forecasting engine using
Box-Jenkins methodology for short, stationary time series.

Model config: SARIMA(1,1,1)(1,1,1,7)
- (1,1,1) → non-seasonal AR, differencing, MA
- (1,1,1,7) → weekly seasonal component
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from app.config import settings

logger = logging.getLogger(__name__)

# Suppress convergence warnings during training (expected for automated fitting)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SarimaModel:
    """
    SARIMA wrapper with a scikit-learn–style interface.

    Usage
    -----
        model = SarimaModel()
        fitted = model.train(train_series)
        forecast = model.predict(fitted)
    """

    MODEL_NAME = "sarima"

    # SARIMAX orders — tuned for weekly ERP demand data
    ORDER = (1, 1, 1)           # (p, d, q)
    SEASONAL_ORDER = (1, 1, 1, 7)  # (P, D, Q, s)

    def __init__(self) -> None:
        self._fitted_model: SARIMAXResultsWrapper | None = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, series: pd.Series) -> SARIMAXResultsWrapper:
        """
        Fit a SARIMAX model to a univariate demand time series.

        Parameters
        ----------
        series : pd.Series indexed by date, values = units_sold

        Returns
        -------
        Fitted SARIMAXResultsWrapper

        Raises
        ------
        ValueError  if series has fewer than 2 × seasonal period rows.
        RuntimeError if fitting fails.
        """
        min_len = 2 * self.SEASONAL_ORDER[3]  # 14 rows minimum for s=7
        if len(series) < min_len:
            raise ValueError(
                f"SARIMA requires at least {min_len} rows; got {len(series)}."
            )

        logger.info(
            "[%s] Fitting SARIMAX%s × %s on %d observations …",
            self.MODEL_NAME,
            self.ORDER,
            self.SEASONAL_ORDER,
            len(series),
        )

        try:
            sarimax = SARIMAX(
                series,
                order=self.ORDER,
                seasonal_order=self.SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fitted_model = sarimax.fit(disp=False, maxiter=200)
            logger.info("[%s] SARIMAX fitting complete. AIC=%.2f", self.MODEL_NAME,
                        self._fitted_model.aic)
        except Exception as exc:
            logger.error("[%s] SARIMAX fitting failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"SARIMA training failed: {exc}") from exc

        return self._fitted_model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        fitted_model: SARIMAXResultsWrapper | None = None,
        horizon: int = None,
    ) -> List[float]:
        """
        Generate a multi-step forecast.

        Parameters
        ----------
        fitted_model : optional; uses self._fitted_model if not provided
        horizon      : forecast steps, defaults to settings.FORECAST_HORIZON

        Returns
        -------
        List[float] of length `horizon`, clipped to ≥ 0.
        """
        if horizon is None:
            horizon = settings.FORECAST_HORIZON

        model = fitted_model or self._fitted_model
        if model is None:
            raise RuntimeError("No fitted model available. Call train() first.")

        try:
            forecast_obj = model.get_forecast(steps=horizon)
            predictions = forecast_obj.predicted_mean.values
            # Clip negative values (demand cannot be negative)
            predictions = np.maximum(predictions, 0.0)
            logger.info("[%s] Generated %d-step forecast.", self.MODEL_NAME, horizon)
            return predictions.tolist()
        except Exception as exc:
            logger.error("[%s] Inference failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"SARIMA inference failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @staticmethod
    def prepare_series(train_df: pd.DataFrame) -> pd.Series:
        """
        Extract units_sold as a DatetimeIndex Series from the feature df.
        SARIMA only needs the univariate target.
        """
        series = train_df.set_index("date")["units_sold"].asfreq("D")
        # Fill any calendar gaps with forward fill (should not occur for seeded data)
        series = series.ffill().bfill()
        return series
