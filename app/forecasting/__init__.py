"""
app/forecasting/__init__.py
============================
Forecasting models package.
Exposes a unified MODEL_REGISTRY dict for easy dynamic dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, Type


# Lazy imports to avoid loading all heavy frameworks at startup
def _import_sarima():
    from app.forecasting.sarima_model import SarimaModel
    return SarimaModel

def _import_prophet():
    from app.forecasting.prophet_model import ProphetModel
    return ProphetModel

def _import_xgboost():
    from app.forecasting.xgboost_model import XGBoostModel
    return XGBoostModel

def _import_lstm():
    from app.forecasting.lstm_model import LSTMModel
    return LSTMModel

def _import_tft():
    from app.forecasting.tft_model import TFTModel
    return TFTModel


MODEL_LOADERS: Dict[str, Any] = {
    "sarima": _import_sarima,
    "prophet": _import_prophet,
    "xgboost": _import_xgboost,
    "lstm": _import_lstm,
    "tft": _import_tft,
}

__all__ = ["MODEL_LOADERS"]
