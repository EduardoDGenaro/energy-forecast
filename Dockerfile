FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    DBT_PROFILES_DIR=/app/dbt

WORKDIR /app

# Dependencias de sistema necessarias para compilacao e execucao do LightGBM (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY dbt/ ./dbt/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/
COPY tests/ ./tests/

RUN pip install --upgrade pip && \
    pip install -e ".[dev,dashboard]" && \
    cp dbt/profiles.yml.example dbt/profiles.yml

CMD ["pytest"]

