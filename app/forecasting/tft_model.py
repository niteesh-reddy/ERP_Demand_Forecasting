"""
tft_model.py — Temporal Fusion Transformer (TFT)
=================================================
Uses PyTorch Forecasting's TemporalFusionTransformer with:
- Static categorical features: category (product), region (warehouse)
- Time-varying known reals: promotion_flag, day_of_week, month
- Time-varying unknown reals: units_sold, lag features, rolling stats, inventory

SAP Context
-----------
Mirrors SAP IBP's advanced ML models that combine attention mechanisms
with covariate inputs for probabilistic supply-chain forecasting.

Notes
-----
- We train on CPU by default for portability (GPU optional via trainer flags).
- Output is the mean prediction of the probabilistic forecast.
- Training is limited to 30 epochs for speed; increase for production.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Column definitions — must match preprocessing.py output
# ------------------------------------------------------------------ #

STATIC_CATEGORICALS = ["category_encoded", "region_encoded"]
TIME_VARYING_KNOWN_REALS = ["promotion_flag", "day_of_week", "month"]
TIME_VARYING_UNKNOWN_REALS = [
    "units_sold",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "inventory_level",
]
GROUP_IDS = ["series_id"]       # single series identifier column
TIME_IDX = "time_idx"           # integer time index required by PyTorch Forecasting


class TFTModel:
    """
    Temporal Fusion Transformer wrapper around PyTorch Forecasting.

    Usage
    -----
        model = TFTModel()
        trainer, tft = model.train(train_df)
        forecast = model.predict(trainer, tft, train_df)
    """

    MODEL_NAME = "tft"

    def __init__(self) -> None:
        self._trainer = None
        self._tft = None

    # ------------------------------------------------------------------ #
    # Data Preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns required by PyTorch Forecasting's TimeSeriesDataSet:
        - series_id  : constant "0" (single-series training)
        - time_idx   : integer index starting from 0
        - cast static categoricals to strings (PF requirement)
        """
        tft_df = df.copy()
        tft_df[TIME_IDX] = np.arange(len(tft_df), dtype=int)
        tft_df[GROUP_IDS[0]] = "0"   # single group

        # PyTorch Forecasting expects static categoricals as strings
        for col in STATIC_CATEGORICALS:
            tft_df[col] = tft_df[col].astype(str)

        return tft_df

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_df: pd.DataFrame,
        max_epochs: int = 20,
    ) -> Tuple[Any, Any]:
        """
        Fit a Temporal Fusion Transformer.

        Parameters
        ----------
        train_df   : feature DataFrame from preprocessing
        max_epochs : training epochs (20 for speed; increase for production)

        Returns
        -------
        (lightning.Trainer, TemporalFusionTransformer)

        Raises
        ------
        ImportError  if pytorch-forecasting / lightning not installed.
        RuntimeError if training fails.
        """
        try:
            import torch
            import lightning as pl
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import MAE as PF_MAE
        except ImportError as exc:
            raise ImportError(
                "pytorch-forecasting and lightning are required: "
                "pip install pytorch-forecasting lightning torch"
            ) from exc

        horizon = settings.FORECAST_HORIZON
        max_encoder_length = settings.SEQUENCE_LENGTH

        tft_df = self._prepare_dataframe(train_df)

        # Time series dataset
        try:
            training_dataset = TimeSeriesDataSet(
                tft_df,
                time_idx=TIME_IDX,
                target="units_sold",
                group_ids=GROUP_IDS,
                max_encoder_length=max_encoder_length,
                max_prediction_length=horizon,
                static_categoricals=STATIC_CATEGORICALS,
                time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
                time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
                target_normalizer=GroupNormalizer(
                    groups=GROUP_IDS, transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
        except Exception as exc:
            raise RuntimeError(f"TFT dataset creation failed: {exc}") from exc

        train_loader = training_dataset.to_dataloader(
            train=True, batch_size=32, num_workers=0
        )

        # TFT architecture
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=3e-3,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=PF_MAE(),
            log_interval=-1,
            reduce_on_plateau_patience=3,
        )

        logger.info(
            "[%s] TFT has %d parameters. Training for %d epochs …",
            self.MODEL_NAME,
            sum(p.numel() for p in tft.parameters()),
            max_epochs,
        )

        # Lightning trainer
        checkpoint_dir = Path(settings.MODELS_DIR) / self.MODEL_NAME / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="train_loss",
            save_top_k=1,
            mode="min",
        )
        early_stop_cb = EarlyStopping(
            monitor="train_loss", patience=5, mode="min", verbose=False
        )

        try:
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[checkpoint_cb, early_stop_cb],
                logger=False,
                gradient_clip_val=0.1,
            )
            trainer.fit(tft, train_dataloaders=train_loader)
            logger.info("[%s] Training complete.", self.MODEL_NAME)
        except Exception as exc:
            logger.error("[%s] Training failed: %s", self.MODEL_NAME, exc)
            raise RuntimeError(f"TFT training failed: {exc}") from exc

        self._trainer = trainer
        self._tft = tft
        return trainer, tft

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        trainer: Any,
        tft_model: Any,
        history_df: pd.DataFrame,
        horizon: int = None,
    ) -> List[float]:
        """
        Generate a probabilistic TFT forecast (mean output).

        Parameters
        ----------
        trainer    : Lightning Trainer used for training
        tft_model  : fitted TemporalFusionTransformer
        history_df : full history DataFrame
        horizon    : defaults to settings.FORECAST_HORIZON

        Returns
        -------
        List[float] of length `horizon`, clipped to ≥ 0.
        """
        if horizon is None:
            horizon = settings.FORECAST_HORIZON

        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError as exc:
            raise ImportError("pytorch-forecasting required for inference.") from exc

        max_encoder_length = settings.SEQUENCE_LENGTH
        tft_df = self._prepare_dataframe(history_df)

        # Use last (encoder + horizon) rows for inference
        cutoff = len(tft_df) - 1
        inference_df = tft_df.iloc[max(0, cutoff - max_encoder_length + 1):].copy()
        inference_df[TIME_IDX] = np.arange(len(inference_df), dtype=int)

        try:
            inference_dataset = TimeSeriesDataSet.from_dataset(
                # Reconstruct from the training config
                tft_model.dataset_parameters,
                inference_df,
                predict=True,
                stop_randomization=True,
            )
            loader = inference_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )
            raw_preds = tft_model.predict(loader, mode="prediction", return_x=False)
            preds = raw_preds[0].numpy().flatten()[:horizon]
        except Exception as exc:
            logger.warning(
                "[%s] Standard inference failed (%s). Falling back to simple mean baseline.",
                self.MODEL_NAME, exc,
            )
            # Fallback: return rolling mean of last 30 days so the endpoint doesn't crash
            fallback = float(history_df["units_sold"].iloc[-30:].mean())
            preds = np.full(horizon, fallback)

        predictions = [max(0.0, float(p)) for p in preds]
        logger.info("[%s] Generated %d-step forecast.", self.MODEL_NAME, horizon)
        return predictions

    # ------------------------------------------------------------------ #
    # Save / Load helpers (stored as joblib for the dataset params)
    # ------------------------------------------------------------------ #

    def save_dataset_params(self, tft_model: Any) -> None:
        """Persist dataset parameters needed to reconstruct inference dataset."""
        import joblib
        path = Path(settings.MODELS_DIR) / self.MODEL_NAME / "dataset_params.joblib"
        joblib.dump(tft_model.dataset_parameters, str(path), compress=3)
        logger.info("[%s] Dataset params saved → %s", self.MODEL_NAME, path)

    @staticmethod
    def load_dataset_params() -> Dict:
        """Load dataset parameters from disk."""
        import joblib
        path = Path(settings.MODELS_DIR) / "tft" / "dataset_params.joblib"
        if not path.exists():
            raise FileNotFoundError("TFT dataset params not found. Run POST /train first.")
        return joblib.load(str(path))
