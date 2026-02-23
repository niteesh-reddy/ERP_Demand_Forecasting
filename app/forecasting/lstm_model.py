"""
lstm_model.py — LSTM Demand Forecasting Model
===============================================
Multivariate LSTM that takes a 30-day look-back window
of features and outputs a 30-step direct forecast.

Architecture:
  Input: (batch, 30, n_features)
  LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(30)
  Loss: MAE  |  Optimizer: Adam

SAP Context
-----------
Mirrors SAP BTP ML-based demand sensing models that learn
temporal patterns from multivariate input signals.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.config import settings
from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Number of days the LSTM looks back
SEQ_LEN = settings.SEQUENCE_LENGTH   # 30


class LSTMModel:
    """
    Keras LSTM for multivariate demand forecasting.

    Usage
    -----
        model = LSTMModel()
        keras_model, scaler = model.train(train_df)
        forecast = model.predict(keras_model, scaler, train_df)
    """

    MODEL_NAME = "lstm"

    def __init__(self) -> None:
        self._model = None
        self._scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------ #
    # Data Preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window over X to create (seq_len → horizon) pairs.

        Returns
        -------
        X_seq : (n_samples, seq_len, n_features)
        y_seq : (n_samples, horizon)
        """
        n = len(X)
        X_seqs, y_seqs = [], []
        for i in range(n - seq_len - horizon + 1):
            X_seqs.append(X[i: i + seq_len])
            y_seqs.append(y[i + seq_len: i + seq_len + horizon])
        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Build Model Architecture
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_model(n_features: int, horizon: int):
        """Construct and compile the LSTM graph."""
        try:
            from tensorflow import keras  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("TensorFlow is not installed: pip install tensorflow") from exc

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(SEQ_LEN, n_features)),
                keras.layers.LSTM(128, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(64, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(horizon),
            ],
            name="demand_lstm",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="mae",
            metrics=["mse"],
        )
        return model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> Tuple:
        """
        Fit the LSTM model.

        Parameters
        ----------
        train_df   : feature DataFrame from preprocessing
        epochs     : training epochs (default 30 for speed; increase for production)
        batch_size : mini-batch size

        Returns
        -------
        (keras.Model, MinMaxScaler)

        Raises
        ------
        ValueError  if insufficient rows for sequence length + horizon.
        RuntimeError if Keras training fails.
        """
        horizon = settings.FORECAST_HORIZON

        all_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing = set(all_cols) - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing columns for LSTM: {missing}")

        # Scale all features + target to [0, 1]
        scaler = MinMaxScaler()
        data = train_df[all_cols].values.astype(np.float32)
        data_scaled = scaler.fit_transform(data)

        # Feature matrix (all cols) and target index
        X = data_scaled                                     # (n, n_features+1)
        target_idx = all_cols.index(TARGET_COLUMN)
        y = data_scaled[:, target_idx]                     # (n,)

        min_needed = SEQ_LEN + horizon
        if len(X) < min_needed:
            raise ValueError(
                f"LSTM needs at least {min_needed} rows; got {len(X)}."
            )

        X_seq, y_seq = self._build_sequences(X, y, SEQ_LEN, horizon)
        logger.info(
            "[%s] Training on %d sequences of shape (%d, %d) …",
            self.MODEL_NAME, X_seq.shape[0], SEQ_LEN, X_seq.shape[2],
        )

        try:
            from tensorflow import keras  # type: ignore[import]

            model = self._build_model(n_features=X_seq.shape[2], horizon=horizon)

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, verbose=0
                ),
            ]

            model.fit(
                X_seq, y_seq,
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=True,
                verbose=0,
            )
            logger.info("[%s] Training complete.", self.MODEL_NAME)
        except Exception as exc:
            logger.error("[%s] Training failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"LSTM training failed: {exc}") from exc

        self._model = model
        self._scaler = scaler
        return model, scaler

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        model,
        scaler: MinMaxScaler,
        history_df: pd.DataFrame,
        horizon: int = None,
    ) -> List[float]:
        """
        Predict the next `horizon` days from the last SEQ_LEN rows of history.

        Parameters
        ----------
        model      : fitted Keras model
        scaler     : fitted MinMaxScaler
        history_df : full history (at least SEQ_LEN rows)
        horizon    : defaults to settings.FORECAST_HORIZON

        Returns
        -------
        List[float] of length `horizon`, clipped to ≥ 0.
        """
        if horizon is None:
            horizon = settings.FORECAST_HORIZON

        all_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        data = history_df[all_cols].values[-SEQ_LEN:].astype(np.float32)

        if len(data) < SEQ_LEN:
            # Pad with zeros if history is shorter than sequence length
            pad = np.zeros((SEQ_LEN - len(data), data.shape[1]), dtype=np.float32)
            data = np.vstack([pad, data])

        data_scaled = scaler.transform(data)
        X_input = data_scaled[np.newaxis, :, :]   # (1, SEQ_LEN, n_features)

        try:
            y_scaled = model.predict(X_input, verbose=0)[0]  # (horizon,)
        except Exception as exc:
            raise RuntimeError(f"LSTM inference failed: {exc}") from exc

        # Inverse scale: build a dummy full-width array to use scaler.inverse_transform
        target_idx = all_cols.index(TARGET_COLUMN)
        dummy = np.zeros((horizon, len(all_cols)), dtype=np.float32)
        dummy[:, target_idx] = y_scaled
        preds_full = scaler.inverse_transform(dummy)
        predictions = preds_full[:, target_idx].tolist()
        predictions = [max(0.0, p) for p in predictions]

        logger.info("[%s] Generated %d-step forecast.", self.MODEL_NAME, horizon)
        return predictions
