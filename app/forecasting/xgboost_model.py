"""
xgboost_model.py — XGBoost Demand Forecasting Model
=====================================================
Trains an XGBRegressor on lag + rolling + calendar features.
Uses recursive multi-step forecasting: each prediction is fed
back as input to generate the next step.

SAP Context
-----------
Mirrors SAP Integrated Business Planning (IBP) machine-learning
add-on's gradient-boosted tree approach for demand sensing.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost-based demand forecasting model.

    Usage
    -----
        model = XGBoostModel()
        regressor, scaler = model.train(train_df)
        forecast = model.predict(regressor, scaler, train_df)
    """

    MODEL_NAME = "xgboost"

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self, train_df: pd.DataFrame
    ) -> tuple[xgb.XGBRegressor, StandardScaler]:
        """
        Fit XGBoost regressor on the feature matrix.

        Parameters
        ----------
        train_df : feature DataFrame (output of preprocessing.load_feature_data)

        Returns
        -------
        (xgb.XGBRegressor, StandardScaler) — fitted model + scaler

        Raises
        ------
        ValueError  if required feature columns are missing.
        RuntimeError if training fails.
        """
        missing = set(FEATURE_COLUMNS) - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = train_df[FEATURE_COLUMNS].values.astype(np.float32)
        y = train_df[TARGET_COLUMN].values.astype(np.float32)

        # Feature scaling (helps XGBoost converge better & aids recursive forecast)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(
            "[%s] Training on %d samples, %d features …",
            self.MODEL_NAME, X.shape[0], X.shape[1],
        )

        try:
            # Split off last 20% as internal validation for early stopping
            split = int(len(X_scaled) * 0.8)
            X_tr, X_val = X_scaled[:split], X_scaled[split:]
            y_tr, y_val = y[:split], y[split:]

            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                early_stopping_rounds=20,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            logger.info(
                "[%s] Training complete. Best iteration: %d",
                self.MODEL_NAME, model.best_iteration,
            )
        except Exception as exc:
            logger.error("[%s] Training failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"XGBoost training failed: {exc}") from exc

        self._model = model
        self._scaler = scaler
        return model, scaler

    # ------------------------------------------------------------------ #
    # Inference — Recursive Multi-Step Forecast
    # ------------------------------------------------------------------ #

    def predict(
        self,
        model: xgb.XGBRegressor,
        scaler: StandardScaler,
        history_df: pd.DataFrame,
        horizon: int = None,
    ) -> List[float]:
        """
        Generate a recursive multi-step forecast.

        The last row of `history_df` is used as the starting point.
        Each prediction is appended as new lag features for the next step.

        Parameters
        ----------
        model      : fitted XGBRegressor
        scaler     : fitted StandardScaler
        history_df : full history DataFrame (train + optionally test)
        horizon    : steps ahead (default: settings.FORECAST_HORIZON)

        Returns
        -------
        List[float] of length `horizon`, clipped to ≥ 0.
        """
        if horizon is None:
            horizon = settings.FORECAST_HORIZON

        # Working buffer — keeps the last 14 rows for lag computation
        buffer = history_df[TARGET_COLUMN].values.tolist()
        # Static features from the last row (they don't change in forecast)
        last_row = history_df.iloc[-1]

        predictions: List[float] = []

        for step in range(horizon):
            # Build feature vector for this forecast step
            arr = buffer[-max(14, len(buffer)):]
            lag_1 = arr[-1]
            lag_7 = arr[-7] if len(arr) >= 7 else arr[0]
            lag_14 = arr[-14] if len(arr) >= 14 else arr[0]
            roll_7 = float(np.mean(arr[-7:]))
            roll_14 = float(np.mean(arr[-14:]))
            roll_std_7 = float(np.std(arr[-7:])) if len(arr) >= 7 else 0.0

            # Calendar: advance from last observed date by `step+1` days
            future_date = pd.Timestamp(last_row["date"]) + pd.Timedelta(days=step + 1)
            dow = future_date.dayofweek
            month = future_date.month
            promo = 0          # assume no promotion in forecast horizon
            inv = float(last_row["inventory_level"])
            cat = int(last_row["category_encoded"])
            reg = int(last_row["region_encoded"])

            feat = np.array(
                [[lag_1, lag_7, lag_14, roll_7, roll_14, roll_std_7,
                  dow, month, promo, inv, cat, reg]],
                dtype=np.float32,
            )
            feat_scaled = scaler.transform(feat)

            pred = float(model.predict(feat_scaled)[0])
            pred = max(0.0, pred)

            predictions.append(pred)
            buffer.append(pred)   # feed prediction back as next lag

        logger.info("[%s] Recursive forecast complete (%d steps).", self.MODEL_NAME, horizon)
        return predictions
