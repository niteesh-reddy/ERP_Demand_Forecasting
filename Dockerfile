# ============================================================
# Dockerfile — Enterprise Demand Forecasting Microservice
# Production-grade containerization (Python 3.10-slim, non-root)
# ============================================================

FROM python:3.10-slim

# System-level dependencies required by Prophet, psycopg2, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Set writable env dirs for non-root user (matplotlib, cmdstanpy, etc.)
ENV MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install CmdStan binary (required by Prophet for Bayesian fitting)
# This compiles the C++ Stan backend that cmdstanpy wraps
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(cores=2, progress=False)"

# Copy application code
COPY app/ ./app/
COPY training/ ./training/
COPY scripts/ ./scripts/

# Create models directory and set ownership
RUN mkdir -p /app/models && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check (calls /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); exit(0 if r.status_code==200 else 1)"

# Launch the FastAPI application via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
