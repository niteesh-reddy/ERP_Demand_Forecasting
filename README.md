# Enterprise Demand Forecasting Microservice

> **Production-grade, multi-model demand forecasting system with SAP S/4HANA-style ERP integration.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)](https://postgresql.org)

---

## Business Problem

Modern enterprises running SAP S/4HANA face a critical challenge: **demand volatility**. Supply chain disruptions, seasonal spikes, and promotion events make accurate 30-day demand forecasting essential for:

- Optimizing inventory replenishment cycles
- Reducing warehouse carrying costs
- Preventing stockouts that cost lost revenue
- Aligning production schedules with predicted demand

This microservice **benchmarks five industry-standard forecasting models** and automatically promotes the best one into a live REST API — replacing manual SAP IBP configuration with a self-tuning ML pipeline.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     ERP Demand Forecasting Stack                  │
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │  PostgreSQL  │────▶│  FastAPI     │────▶│  Model Registry  │  │
│  │  (ERP Data)  │     │  Microservice│     │  /models/*.json  │  │
│  └──────────────┘     └──────┬───────┘     └──────────────────┘  │
│                              │                                    │
│                  ┌───────────▼──────────┐                        │
│                  │  Model Dispatcher    │                        │
│                  │                      │                        │
│          ┌───────┼──────┬──────┬───────┤                        │
│          ▼       ▼      ▼      ▼       ▼                        │
│       SARIMA  Prophet XGBoost LSTM    TFT                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  SAP S/4HANA Tables (simulated)                          │    │
│  │  product_master | warehouse_master |                     │    │
│  │  sales_transactions | inventory_levels                   │    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Model Comparison

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **SARIMA** | Interpretable, strong on stationary series | Slow, manual tuning | Stable products |
| **Prophet** | Holiday effects, trend changepoints | Less accurate on noisy data | Seasonal spikes |
| **XGBoost** | Fast, handles lag features natively | No temporal structure | Feature-rich datasets |
| **LSTM** | Learns temporal dependencies | Needs lots of data | Complex multivariate patterns |
| **TFT** | Probabilistic, attention mechanism | Slowest, most complex | Production at scale |

The system **auto-selects the best model based on lowest MAPE** and registers it.

---

## SAP S/4HANA Integration

This service simulates SAP's key ERP entities:

| SAP Module | SAP Table | Our Table |
|-----------|-----------|-----------|
| MM (Materials) | T001W / MARA | `product_master` |
| SD (Sales) | VBAP / VBAK | `sales_transactions` |
| WM (Warehouse) | LGORT / T001L | `warehouse_master` |
| MM (Inventory) | MARD / MMBE | `inventory_levels` |

**In production**, replace the PostgreSQL queries with SAP RFC/BAPI calls or OData service consumption via SAP's Python RFC connector (`pyrfc`).

---

## Project Structure

```
erp-demand-forecasting/
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app (all endpoints)
│   ├── config.py            ← Settings via pydantic-settings
│   ├── database.py          ← SQLAlchemy ORM + session management
│   ├── schemas.py           ← Pydantic request/response models
│   ├── preprocessing.py     ← Feature engineering pipeline
│   ├── evaluation.py        ← MAE / RMSE / MAPE + timing
│   ├── model_registry.py    ← Save / load / best-model tracking
│   └── forecasting/
│       ├── __init__.py
│       ├── sarima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       └── tft_model.py
├── training/
│   └── train_all_models.py  ← Orchestration script
├── scripts/
│   └── seed_data.py         ← Synthetic ERP data generator
├── models/                  ← Trained model artefacts (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ running locally (or use Docker)
- 8 GB RAM recommended (TFT and LSTM are memory-intensive)

### 1. Clone & Install

```bash
git clone https://github.com/yourorg/erp-demand-forecasting.git
cd erp-demand-forecasting

python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Seed the Database

```bash
python scripts/seed_data.py
```

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Train the Models

```bash
# Via API:
curl -X POST "http://localhost:8000/train?product_id=P001&warehouse_id=W001"

# Or directly:
python training/train_all_models.py
```

---

## Docker Deployment

```bash
# Build & start all services (PostgreSQL + FastAPI)
docker compose up --build -d

# Seed the database
docker exec erp_forecast_api python scripts/seed_data.py

# Check logs
docker compose logs -f api
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-02-23T14:00:00Z",
  "database": "connected",
  "best_model": "xgboost"
}
```

### Train All Models
```bash
curl -X POST "http://localhost:8000/train?product_id=P001&warehouse_id=W001"
```

### Get Forecast (auto-select best model)
```bash
curl "http://localhost:8000/forecast?product_id=P001&warehouse_id=W001"
```

### Get Forecast (specific model)
```bash
curl "http://localhost:8000/forecast?product_id=P001&warehouse_id=W001&model=xgboost"
```

```json
{
  "product_id": "P001",
  "warehouse_id": "W001",
  "model_used": "xgboost",
  "forecast_horizon_days": 30,
  "forecast": [
    {"day": 1, "forecast_units": 48.3},
    {"day": 2, "forecast_units": 51.7},
    ...
  ]
}
```

### List Models & Metrics
```bash
curl http://localhost:8000/models
```

### Interactive Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1`, `lag_7`, `lag_14` | Units sold 1, 7, 14 days ago |
| `rolling_mean_7/14` | 7 and 14-day rolling average |
| `rolling_std_7` | 7-day demand volatility |
| `day_of_week` | 0=Monday … 6=Sunday |
| `month` | 1–12 seasonality |
| `promotion_flag` | 1 if discount_percent > 0 |
| `inventory_level` | Current stock on hand |
| `category_encoded` | Product category (ordinal) |
| `region_encoded` | Warehouse region (ordinal) |

---

## Evaluation Metrics

- **MAE** — Mean Absolute Error (unit: units/day)
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error (model selection criterion)
- **Training time** — wall-clock seconds
- **Inference time** — wall-clock seconds per forecast call
