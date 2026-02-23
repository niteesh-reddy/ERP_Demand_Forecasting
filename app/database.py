"""
database.py — SQLAlchemy ORM Models & Database Layer
=====================================================
Defines the four ERP-style tables that mirror a subset of SAP S/4HANA's
materials management (MM) and sales & distribution (SD) master data.

Tables
------
- product_master   → SAP Material Master (MM60)
- warehouse_master → SAP Plant / Storage Location
- sales_transactions → SAP SD Sales Orders aggregated to daily grain
- inventory_levels   → SAP MM Stock Overview (MMBE)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Engine & Session Factory
# ============================================================

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,        # keeps connections alive through network drops
    pool_size=5,
    max_overflow=10,
    echo=False,                # set True for SQL query debugging
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ============================================================
# Declarative Base
# ============================================================

class Base(DeclarativeBase):
    """All ORM models inherit from this base."""
    pass


# ============================================================
# ORM Models (ERP Tables)
# ============================================================

class ProductMaster(Base):
    """
    SAP equivalent: Material Master (MM60).
    One row per SKU in the ERP catalogue.
    """
    __tablename__ = "product_master"
    __allow_unmapped__ = True   # SQLAlchemy 2.x compatibility

    product_id = Column(String(20), primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    unit_price = Column(Float, nullable=False)
    lead_time_days = Column(Integer, nullable=False)

    # Relationships
    sales = relationship("SalesTransaction", back_populates="product", lazy="dynamic")
    inventory = relationship("InventoryLevel", back_populates="product", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<ProductMaster id={self.product_id} cat={self.category}>"


class WarehouseMaster(Base):
    """
    SAP equivalent: Plant / Storage Location master.
    One row per distribution centre.
    """
    __tablename__ = "warehouse_master"
    __allow_unmapped__ = True   # SQLAlchemy 2.x compatibility

    warehouse_id = Column(String(20), primary_key=True, index=True)
    region = Column(String(50), nullable=False)
    capacity = Column(Integer, nullable=False)

    # Relationships
    sales = relationship("SalesTransaction", back_populates="warehouse", lazy="dynamic")
    inventory = relationship("InventoryLevel", back_populates="warehouse", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<WarehouseMaster id={self.warehouse_id} region={self.region}>"


class SalesTransaction(Base):
    """
    SAP equivalent: Daily aggregation of SD Sales Orders.
    Each row = one product × one warehouse × one day.
    """
    __tablename__ = "sales_transactions"
    __allow_unmapped__ = True   # SQLAlchemy 2.x compatibility

    transaction_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(
        String(20), ForeignKey("product_master.product_id"), nullable=False, index=True
    )
    warehouse_id = Column(
        String(20), ForeignKey("warehouse_master.warehouse_id"), nullable=False, index=True
    )
    sales_date = Column(Date, nullable=False, index=True)
    units_sold = Column(Float, nullable=False)
    discount_percent = Column(Float, nullable=False, default=0.0)

    # Relationships
    product = relationship("ProductMaster", back_populates="sales")
    warehouse = relationship("WarehouseMaster", back_populates="sales")

    __table_args__ = (
        UniqueConstraint("product_id", "warehouse_id", "sales_date",
                         name="uq_sales_product_warehouse_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<SalesTransaction pid={self.product_id} wid={self.warehouse_id} "
            f"date={self.sales_date} units={self.units_sold}>"
        )


class InventoryLevel(Base):
    """
    SAP equivalent: MM Stock Overview (MMBE) at daily granularity.
    Composite primary key: product × warehouse × date.
    """
    __tablename__ = "inventory_levels"
    __allow_unmapped__ = True   # SQLAlchemy 2.x compatibility

    product_id = Column(
        String(20), ForeignKey("product_master.product_id"), primary_key=True
    )
    warehouse_id = Column(
        String(20), ForeignKey("warehouse_master.warehouse_id"), primary_key=True
    )
    date = Column(Date, primary_key=True)
    stock_available = Column(Float, nullable=False)

    # Relationships
    product = relationship("ProductMaster", back_populates="inventory")
    warehouse = relationship("WarehouseMaster", back_populates="inventory")

    def __repr__(self) -> str:
        return (
            f"<InventoryLevel pid={self.product_id} wid={self.warehouse_id} "
            f"date={self.date} stock={self.stock_available}>"
        )


# ============================================================
# Initialisation Helper
# ============================================================

def init_db() -> None:
    """
    Create all tables if they do not already exist.
    Called once on application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised successfully.")
    except Exception as exc:
        logger.error("Failed to initialise database tables: %s", exc)
        raise


def check_db_connection() -> bool:
    """
    Lightweight connectivity check used by /health endpoint.
    Returns True if the DB is reachable, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False


# ============================================================
# Dependency-Injection Helper (for FastAPI)
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.
    Usage::

        @app.get("/endpoint")
        def my_endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def managed_session() -> Generator[Session, None, None]:
    """
    Context manager for use outside of FastAPI (e.g., training scripts).
    Usage::

        with managed_session() as session:
            rows = session.query(SalesTransaction).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
