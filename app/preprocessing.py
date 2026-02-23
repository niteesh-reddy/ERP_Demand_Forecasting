"""
preprocessing.py — Feature Engineering Pipeline
================================================
Loads raw ERP data from PostgreSQL and transforms it into a
feature matrix ready for all five forecasting models.

Feature set
-----------
- units_sold          (target)
- lag_1, lag_7, lag_14
- rolling_mean_7, rolling_mean_14
- rolling_std_7
- day_of_week         (0-6)
- month               (1-12)
- promotion_flag      (1 if discount > 0)
- inventory_level     (from inventory_levels table)
- category_encoded    (ordinal from ProductMaster)
- region_encoded      (ordinal from WarehouseMaster)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.database import InventoryLevel, ProductMaster, SalesTransaction, WarehouseMaster

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Encoding maps (simulating SAP category / region codes)
# ------------------------------------------------------------------ #
CATEGORY_MAP: dict[str, int] = {
    "Electronics": 0,
    "Apparel": 1,
    "Grocery": 2,
    "Furniture": 3,
    "Pharmaceuticals": 4,
}
REGION_MAP: dict[str, int] = {
    "North": 0,
    "South": 1,
    "Central": 2,
}


# ============================================================
# Public API
# ============================================================

def load_feature_data(
    product_id: str,
    warehouse_id: str,
    session: Session,
    min_rows: int = 60,
) -> pd.DataFrame:
    """
    Fetch all available daily sales + inventory for one product-warehouse
    combination and return a fully-engineered feature DataFrame.

    Parameters
    ----------
    product_id   : str  — e.g. "P001"
    warehouse_id : str  — e.g. "W001"
    session      : SQLAlchemy Session
    min_rows     : int  — minimum history required (raises if fewer found)

    Returns
    -------
    pd.DataFrame sorted by date, columns:
        date, units_sold, lag_1, lag_7, lag_14,
        rolling_mean_7, rolling_mean_14, rolling_std_7,
        day_of_week, month, promotion_flag, inventory_level,
        category_encoded, region_encoded, discount_percent

    Raises
    ------
    ValueError  if product/warehouse not found or insufficient history.
    """
    # --- Validate product & warehouse existence ---
    product: Optional[ProductMaster] = session.get(ProductMaster, product_id)
    if product is None:
        raise ValueError(f"Product '{product_id}' not found in product_master.")

    warehouse: Optional[WarehouseMaster] = session.get(WarehouseMaster, warehouse_id)
    if warehouse is None:
        raise ValueError(f"Warehouse '{warehouse_id}' not found in warehouse_master.")

    # --- Query sales transactions ---
    sales_rows = (
        session.query(
            SalesTransaction.sales_date,
            SalesTransaction.units_sold,
            SalesTransaction.discount_percent,
        )
        .filter(
            SalesTransaction.product_id == product_id,
            SalesTransaction.warehouse_id == warehouse_id,
        )
        .order_by(SalesTransaction.sales_date)
        .all()
    )

    if not sales_rows:
        raise ValueError(
            f"No sales data for product='{product_id}', warehouse='{warehouse_id}'. "
            "Run seed_data.py first."
        )

    df = pd.DataFrame(sales_rows, columns=["date", "units_sold", "discount_percent"])
    df["date"] = pd.to_datetime(df["date"])

    # --- Query inventory levels ---
    inv_rows = (
        session.query(InventoryLevel.date, InventoryLevel.stock_available)
        .filter(
            InventoryLevel.product_id == product_id,
            InventoryLevel.warehouse_id == warehouse_id,
        )
        .order_by(InventoryLevel.date)
        .all()
    )
    inv_df = pd.DataFrame(inv_rows, columns=["date", "stock_available"])
    inv_df["date"] = pd.to_datetime(inv_df["date"])

    # --- Merge sales + inventory ---
    df = df.merge(inv_df, on="date", how="left")
    df["inventory_level"] = df["stock_available"].fillna(method="ffill").fillna(0.0)
    df.drop(columns=["stock_available"], inplace=True)

    # Check minimum history
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient history for {product_id}×{warehouse_id}: "
            f"{len(df)} rows < required {min_rows}."
        )

    # --- Feature Engineering ---
    df = _engineer_features(df, product, warehouse)

    logger.info(
        "Feature data loaded: product=%s, warehouse=%s, rows=%d, features=%d",
        product_id,
        warehouse_id,
        len(df),
        len(df.columns),
    )
    return df


def _engineer_features(
    df: pd.DataFrame,
    product: ProductMaster,
    warehouse: WarehouseMaster,
) -> pd.DataFrame:
    """Apply all feature transformations in-place and return the DataFrame."""
    df = df.sort_values("date").copy()

    target = "units_sold"

    # --- Lag features ---
    df["lag_1"] = df[target].shift(1)
    df["lag_7"] = df[target].shift(7)
    df["lag_14"] = df[target].shift(14)

    # --- Rolling statistics ---
    df["rolling_mean_7"] = df[target].rolling(7, min_periods=1).mean().shift(1)
    df["rolling_mean_14"] = df[target].rolling(14, min_periods=1).mean().shift(1)
    df["rolling_std_7"] = df[target].rolling(7, min_periods=1).std().shift(1).fillna(0.0)

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek      # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month                 # 1-12

    # --- Promotion flag ---
    df["promotion_flag"] = (df["discount_percent"] > 0).astype(int)

    # --- Static encoding ---
    df["category_encoded"] = CATEGORY_MAP.get(product.category, -1)
    df["region_encoded"] = REGION_MAP.get(warehouse.region, -1)

    # --- Drop rows with NaN from lagging (first 14 days) ---
    df.dropna(subset=["lag_1", "lag_7", "lag_14"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# Train / Test Split Helper
# ============================================================

def train_test_split(
    df: pd.DataFrame,
    test_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature DataFrame into train and test sets.
    The last `test_days` rows form the test (hold-out) set.

    Parameters
    ----------
    df        : feature DataFrame from load_feature_data()
    test_days : int, defaults to settings.TEST_DAYS (30)

    Returns
    -------
    (train_df, test_df) tuple
    """
    if test_days is None:
        test_days = settings.TEST_DAYS

    if len(df) <= test_days:
        raise ValueError(
            f"DataFrame has only {len(df)} rows; cannot hold out {test_days} for testing."
        )

    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    logger.debug(
        "Train/test split: train=%d rows, test=%d rows", len(train_df), len(test_df)
    )
    return train_df, test_df


# ============================================================
# Feature Column List (shared across models)
# ============================================================

FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "day_of_week", "month",
    "promotion_flag",
    "inventory_level",
    "category_encoded",
    "region_encoded",
]

TARGET_COLUMN = "units_sold"
