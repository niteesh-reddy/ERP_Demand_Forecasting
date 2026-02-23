"""
seed_data.py — Synthetic ERP Data Generator
=============================================
Generates 2 years of realistic demand data for:
  - 5 products across 3 warehouses (15 time series total)
  - Seasonal patterns, promotions, and inventory correlated with sales

Run once before training:
    python scripts/seed_data.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure app/ is importable when this script is run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import (
    Base,
    InventoryLevel,
    ProductMaster,
    SalesTransaction,
    SessionLocal,
    WarehouseMaster,
    engine,
    init_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Seed constants
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
START_DATE = date(2022, 1, 1)
END_DATE = date(2023, 12, 31)       # ~730 days of history

PRODUCTS = [
    {"product_id": "P001", "category": "Electronics",  "unit_price": 299.99, "lead_time_days": 7},
    {"product_id": "P002", "category": "Apparel",      "unit_price":  49.99, "lead_time_days": 3},
    {"product_id": "P003", "category": "Grocery",      "unit_price":   4.99, "lead_time_days": 1},
    {"product_id": "P004", "category": "Furniture",    "unit_price": 599.99, "lead_time_days": 14},
    {"product_id": "P005", "category": "Pharmaceuticals", "unit_price": 19.99, "lead_time_days": 2},
]

WAREHOUSES = [
    {"warehouse_id": "W001", "region": "North",  "capacity": 10_000},
    {"warehouse_id": "W002", "region": "South",  "capacity": 8_000},
    {"warehouse_id": "W003", "region": "Central","capacity": 12_000},
]

# Base demand per product (units/day)
BASE_DEMAND: dict[str, float] = {
    "P001": 50.0,
    "P002": 120.0,
    "P003": 300.0,
    "P004": 15.0,
    "P005": 200.0,
}

# Per-warehouse demand multiplier
WAREHOUSE_MULTIPLIER: dict[str, float] = {
    "W001": 1.0,
    "W002": 0.75,
    "W003": 1.25,
}


def _generate_demand_series(
    product_id: str,
    warehouse_id: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic demand for one product-warehouse pair.

    Incorporates:
    - Weekly seasonality (weekend dip for Electronics/Furniture)
    - Annual seasonality (Q4 uplift for Electronics, summer peak for Apparel)
    - Random promotional events (10 % of days, +30-60 % uplift)
    - Gaussian noise
    """
    n = len(dates)
    base = BASE_DEMAND[product_id] * WAREHOUSE_MULTIPLIER[warehouse_id]

    # --- Weekly seasonality ---
    dow = dates.dayofweek.values.astype(float)  # 0=Mon … 6=Sun
    if product_id in ("P001", "P004"):  # high-ticket items dip on weekends
        weekly = 1.0 - 0.2 * (dow >= 5).astype(float)
    else:
        weekly = 1.0 + 0.1 * np.sin(2 * np.pi * dow / 7)

    # --- Annual seasonality ---
    day_of_year = dates.dayofyear.values.astype(float)
    if product_id == "P001":   # Electronics → Q4 peak
        annual = 1.0 + 0.5 * np.sin(2 * np.pi * (day_of_year - 274) / 365)
    elif product_id == "P002":  # Apparel → summer peak
        annual = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 91) / 365)
    else:
        annual = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)

    # --- Promotional flag ---
    promo_flag = rng.random(n) < 0.10          # 10 % promotion days
    promo_uplift = 1.0 + promo_flag * rng.uniform(0.3, 0.6, n)
    discount_percent = np.where(promo_flag, rng.uniform(10, 40, n), 0.0)

    # --- Compose demand & add noise ---
    demand = base * weekly * annual * promo_uplift
    noise = rng.normal(0, base * 0.05, n)
    units_sold = np.maximum(0.0, demand + noise).round(1)

    return units_sold, discount_percent.round(2)


def _generate_inventory(
    units_sold: np.ndarray,
    warehouse_capacity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate daily inventory as a rolling stock that replenishes periodically.
    Stock is clipped between 0 and warehouse capacity.
    """
    n = len(units_sold)
    stock = np.zeros(n)
    current_stock = warehouse_capacity * rng.uniform(0.4, 0.7)

    for i in range(n):
        current_stock -= units_sold[i]
        current_stock = max(0.0, current_stock)
        # Weekly replenishment on Mondays (simulated)
        if i % 7 == 0:
            replenishment = warehouse_capacity * rng.uniform(0.3, 0.5)
            current_stock = min(warehouse_capacity, current_stock + replenishment)
        stock[i] = round(current_stock, 2)

    return stock


def seed_master_data(session) -> None:
    """Insert product and warehouse master records (idempotent)."""
    # Products
    for p in PRODUCTS:
        exists = session.get(ProductMaster, p["product_id"])
        if not exists:
            session.add(ProductMaster(**p))
    # Warehouses
    for w in WAREHOUSES:
        exists = session.get(WarehouseMaster, w["warehouse_id"])
        if not exists:
            session.add(WarehouseMaster(**w))
    session.commit()
    logger.info("Master data seeded: %d products, %d warehouses", len(PRODUCTS), len(WAREHOUSES))


def seed_transactional_data(session) -> None:
    """Generate and insert sales + inventory records (idempotent via upsert logic)."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    total_sales_rows = 0
    total_inv_rows = 0

    for product in PRODUCTS:
        pid = product["product_id"]
        for warehouse in WAREHOUSES:
            wid = warehouse["warehouse_id"]

            logger.info("Generating data for %s × %s …", pid, wid)

            units_sold, discount_pct = _generate_demand_series(pid, wid, dates, rng)
            stock = _generate_inventory(units_sold, warehouse["capacity"], rng)

            # --- Sales Transactions ---
            sales_rows = [
                SalesTransaction(
                    product_id=pid,
                    warehouse_id=wid,
                    sales_date=d.date(),
                    units_sold=float(u),
                    discount_percent=float(disc),
                )
                for d, u, disc in zip(dates, units_sold, discount_pct)
            ]
            # Bulk insert with conflict skip (re-seed safe)
            try:
                session.bulk_save_objects(sales_rows, update_changed_only=False)
                session.commit()
                total_sales_rows += len(sales_rows)
            except Exception as exc:
                session.rollback()
                logger.warning("Sales upsert skipped for %s×%s: %s", pid, wid, exc)

            # --- Inventory Levels ---
            inv_rows = [
                InventoryLevel(
                    product_id=pid,
                    warehouse_id=wid,
                    date=d.date(),
                    stock_available=float(s),
                )
                for d, s in zip(dates, stock)
            ]
            try:
                session.bulk_save_objects(inv_rows, update_changed_only=False)
                session.commit()
                total_inv_rows += len(inv_rows)
            except Exception as exc:
                session.rollback()
                logger.warning("Inventory upsert skipped for %s×%s: %s", pid, wid, exc)

    logger.info(
        "Seeding complete: %d sales rows, %d inventory rows inserted.",
        total_sales_rows,
        total_inv_rows,
    )


def main() -> None:
    logger.info("=== ERP Data Seeder Starting ===")
    logger.info("Creating database tables if not present …")
    init_db()

    with SessionLocal() as session:
        logger.info("Seeding master data …")
        seed_master_data(session)

        logger.info("Seeding transactional data (this may take ~30 s) …")
        seed_transactional_data(session)

    logger.info("=== Seeding Finished Successfully ===")


if __name__ == "__main__":
    main()
