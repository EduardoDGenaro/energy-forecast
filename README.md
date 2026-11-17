# Energy Forecast — Daily Electricity Load Forecasting (ONS)

Documentation reviewed on **2026-11-17**, verified against local scripts, configurations, and artifacts. The numbers presented come from a specific historical evaluation; subsequent runs may produce different results and model versions.

2026-11-17 review: [technical verification, model limitations, and dashboard improvements](docs/project_review_2026-11-17.md).
To recalculate diagnostics and reproduce historical forecasts without modifying MLflow:
`python scripts/audit_project.py --retrain`.

Forecast intervals, monitoring, error thresholds, and calendar/temperature experiments: [implementation and results](docs/risk_and_candidates.md).
Run `python scripts/evaluate_risk.py` to persist risk diagnostics.

## 1. Project Overview

**Energy Forecast** is a Data Engineering, Machine Learning, and MLOps project focused on daily electricity load forecasting for Brazil's National Interconnected System (SIN), using official open data published by the National Electric System Operator (ONS).

The solution implements a local, reproducible Medallion architecture (**Bronze**, **Silver**, and **Gold**) built on Parquet files, **DuckDB**, and **dbt Core**, integrated with a predictive modeling and temporal backtesting pipeline using **LightGBM** and experiment tracking via **MLflow**.

## 2. Objective and Business Problem

Short-term power system operation planning requires accurate daily electricity demand estimates for thermal and hydro dispatch, reservoir management, and operational security.

The primary goal of this project is to **forecast daily electricity load (in MWmed) for the Southeast/Central-West subsystem (code `SE`) over a 7-day horizon** ($t+1$ to $t+7$), rigorously comparing two approaches under walk-forward temporal validation with expanding origins:

1. **Seasonal Baseline (Seasonal Naive):** forecast for day $t+h$ ($h \in \{1, \dots, 7\}$) equal to the actual observed load on the same day of the previous week ($t+h-7$), strictly using data observed up to forecast origin $t$.
2. **Machine Learning Model (LightGBM):** gradient-boosted decision tree model using calendar features (day of week, month, weekend, Brazilian national holidays), temporal lags (7, 14, and 28 days respecting the forecast horizon to prevent data leakage), and rolling window statistics computed exclusively from information available up to the forecast origin date.

The success criterion is verifying whether the LightGBM model outperforms the seasonal baseline and quantifying the performance gap using **MAE** (Mean Absolute Error, in MWmed) and **MAPE** (Mean Absolute Percentage Error, in %), both overall and stratified by forecast horizon ($h=1 \dots 7$).

## 3. Official Data Source

- **Organization:** Operador Nacional do Sistema Eletrico (ONS)
- **Open Data Portal:** https://dados.ons.org.br/dataset/carga-energia
- **Public S3 Bucket:** `s3://ons-aws-prod-opendata/dataset/carga_energia_di/`
- **Bucket HTTPS Endpoint:** `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/`
- **Ingestion Time Range:** January 2018 through the latest date available at the source at runtime.
- **Local Dataset Range Used in Results Below:** 2018-01-01 to 2026-11-14, totaling 3,240 days for the `SE` subsystem. This does not imply automatic updates to the project.
- **Geographic/Electrical Scope:** Southeast/Central-West Subsystem (`id_subsistema = 'SE'`).
- **Unit of Measurement:** `MWmed` (daily average Megawatt).

Additional details on the schema and pre-ingestion validation rules are available in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## 4. Architecture

The project follows the Medallion architecture combined with a structured MLOps workflow:

- **Bronze Layer (`data/bronze/`):** idempotent ingestion of annual ONS files, preserving raw bytes and source schema. The manifest (`ingestion_manifest.json`) records source URL, hash, date range, and timestamps. Files updated at the source can be re-downloaded and overwrite the local annual copy; Bronze does not retain multiple historical versions from the source.
- **Silver Layer (`dbt/models/silver/`):** column and type standardization (dates and numeric fields), explicit handling of nulls and documented inconsistencies, duplicate checks, and traceability back to Bronze Parquet files in DuckDB.
- **Gold Layer (`dbt/models/gold/`):** analytical daily time series filtered for the `SE` subsystem, ordered chronologically, with a unique time key and automated quality tests (`not_null`, `unique`, and temporal continuity checks) ready for feature engineering.
- **Feature Engineering, Modeling, and Backtesting (`src/energy_forecast/`):** leakage-free feature construction, expanding-window walk-forward backtesting across multiple forecast origins, logging of hyperparameters, metrics, and artifacts in **MLflow** (`artifacts/`), and forecast persistence in `data/predictions/`.
- **Optional Visualization (`dashboard/app.py`):** Streamlit application with dark theme, sidebar navigation, 7-day forecast, empirical uncertainty intervals, historical evaluation, filters, tables, and CSV export. The UI layer does not modify saved models or forecasts.
- **Risk Diagnostics and Experiments:** calibration, issued-forecast tracking, and candidate comparison modules. Reports are saved under `artifacts/risk/` and `artifacts/candidates/`; these analyses do not automatically promote candidate models.

Detailed architecture specifications and engineering decisions are documented in [`docs/architecture.md`](docs/architecture.md) and [`docs/decisions.md`](docs/decisions.md).

## 5. Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| Language | Python >= 3.11 | Core language for ingestion, feature engineering, ML, and CLI orchestration |
| Columnar Storage | Apache Parquet (PyArrow) | Preserving downloaded Bronze data and persisting forecast artifacts |
| Local Analytical DB | DuckDB | High-performance local OLAP processing over Parquet files |
| Transformation & Quality | dbt Core + dbt-duckdb | Declarative modeling and data integrity tests for Silver and Gold layers |
| Data Manipulation | pandas + PyArrow | Temporal feature engineering and backtesting evaluation |
| Machine Learning | LightGBM + scikit-learn + holidays | Tabular predictive model, national holiday calendar, and evaluation metrics |
| MLOps / Tracking | MLflow >= 3.0 | Experiment tracking, parameters, metrics, model registry, and artifacts |
| Code Quality | Ruff + pytest | Linting, standardized formatting, and unit/integration tests |
| Packaging | Docker + Docker Compose + Makefile | Containerized environment and execution shortcuts; minimum versions declared in `pyproject.toml` |
| Visualization (Optional) | Streamlit >= 1.49 + Plotly | Optional interactive dashboard decoupled from the pipeline |

## 6. Installation

### Prerequisites

- Python 3.11 or higher
- Git
- Docker and Docker Compose (optional, for containerized execution)
- Internet access to install dependencies and download ONS data; the temperature experiment also queries Open-Meteo.
- `make` is optional. On Windows, the Python commands below can be run directly in PowerShell.

### Local Setup (Virtual Environment)

1. Create and activate a virtual environment at the project root:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the project in editable mode with development dependencies (and optionally the dashboard):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

3. Create local environment and dbt profile files from the versioned templates:

```bash
# Linux / macOS
cp .env.example .env
cp dbt/profiles.yml.example dbt/profiles.yml
```

```powershell
# Windows / PowerShell
Copy-Item .env.example .env
Copy-Item dbt/profiles.yml.example dbt/profiles.yml
```

Run commands from the project root with the virtual environment active. Core
settings live in `config/settings.yaml`; `.env` provides overrides. Copy the
example files only if the local files do not exist yet, so existing custom
configurations are preserved. The project is configured for the `SE` subsystem
and a 7-day horizon; changing a single environment variable does not guarantee
support for other scopes or horizons.

### Running with Docker

```bash
docker compose build
docker compose run --rm pipeline python scripts/ingest.py
docker compose run --rm pipeline python scripts/run_dbt.py --command build
docker compose run --rm pipeline python scripts/evaluate.py
docker compose run --rm pipeline pytest
```

The `pipeline` service runs `pytest` by default, though integration tests require
the data and artifacts produced by the preceding steps. Both `data/` and
`artifacts/` are mounted as local volumes.

For optional services:

```bash
docker compose --profile optional up dashboard
# In another terminal:
docker compose --profile tracking up mlflow
```

The dashboard listens on port 8501 and MLflow on port 5000. Docker commands
were verified against `docker-compose.yml`; the Docker build was not executed
during the 2026-11-17 review.

## 7. Pipeline Stages

Entry points are centralized under `scripts/`; core stages have shortcuts in
`Makefile`. On a first run, execute ingestion, dbt, and evaluation in that
order. `evaluate.py` also trains the final model and generates the operational
forecast.

1. **Bronze Ingestion:**
   ```bash
   python scripts/ingest.py
   # or: make ingest
   ```
2. **Transformation and Quality Tests (Silver and Gold via dbt + DuckDB):**
   ```bash
   python scripts/run_dbt.py --command build
   # or: make dbt
   ```
3. **Operational training, when only updating the forecast:**
   ```bash
   python scripts/train.py
   # or: make train
   ```
   Saves the 7-day forecast and feature importance, preserves the forecast
   issuance record, and attaches prediction intervals if a backtest is
   available. **Does not register a new version in MLflow.** Does not replace
   the historical evaluation required by the dashboard.

4. **Backtesting, final forecast, and MLflow logging:**
   ```bash
   python scripts/evaluate.py
   # or: make evaluate
   ```
   Evaluates the 52 most recent weekly origins by default, retrains on the full
   history, and logs the model, metrics, and artifacts to MLflow. The number of
   origins can be configured with `--max-origins`; tests and experiments for the
   documented snapshot expect 52.

5. **MLflow UI and Optional Dashboard:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
   # or: make mlflow-ui
   ```
   In another terminal, from the project root:
   ```bash
   python -m streamlit run dashboard/app.py
   # or: make dashboard
   ```
   The default dashboard port is 8501. To run on port 8502, used during the
   visual review of this project:
   ```bash
   python -m streamlit run dashboard/app.py --server.port 8502
   ```

6. **Additional diagnostics, run on demand:**
   ```bash
   python scripts/audit_project.py
   python scripts/audit_project.py --retrain
   python scripts/evaluate_risk.py
   ```
   The audit script recalculates indicators and checks artifacts; `--retrain`
   also reproduces historical forecasts under `artifacts/review/` without
   creating a new MLflow version. `evaluate_risk.py` persists intervals,
   coverage, thresholds, and tracking outputs; it also updates intervals on the
   current forecast and archives an issuance with a real timestamp. Therefore,
   `evaluate_risk.py` **is not read-only**.

`audit_project.py`, `evaluate_risk.py`, `fetch_weather.py`, and
`evaluate_candidates.py` are invoked directly; they do not currently have
`Makefile` shortcuts. No routine is scheduled automatically.

## 8. Implementation Roadmap

- [x] **Stage 1 — Foundation and Architecture:** directory layout, `pyproject.toml`, Docker, dbt/DuckDB setup, core config/logging utilities, initial tests, Ruff linting, and architecture documentation.
- [x] **Stage 2 — ONS Schema Inspection and Bronze Ingestion:** verification of the official data dictionary and ONS S3 files (2018 to present), idempotent downloads, raw Parquet preservation without schema modification, and ingestion manifest generation (`data/bronze/ingestion_manifest.json`).
- [x] **Stage 3 — Silver and Gold Layers with dbt:** source definitions over Bronze Parquets, standardization and auditing in Silver (`main_silver.stg_ons_carga_diaria`), `SE` subsystem daily series in Gold (`main_gold.fct_carga_diaria_se`), and 22 dbt quality/continuity tests.
- [x] **Stage 4 — Seasonal Baseline and Feature Engineering:** implementation of the Seasonal Naive baseline ($t-7$) and temporal feature pipeline (calendar, Brazilian holidays, 7/14/28-day lags, and rolling means/stds anchored at the forecast origin without data leakage).
- [x] **Stage 5 — LightGBM Model, Backtesting, and MLflow:** temporal validation across 52 successive origins (364 evaluated days), baseline comparison (overall and per-horizon MAE and MAPE), and full MLflow tracking (`lightgbm-se-daily-load` v1).
- [x] **Stage 6 — Results Consolidation and Optional Dashboard:** final documentation of observed backtesting metrics and delivery of the optional Streamlit dashboard (`dashboard/app.py`).

## 9. Running Tests and Quality Checks

To run the automated test suite with `pytest`:

```bash
pytest
# or: make test
```

The suite combines unit, UI, and integration tests. Part of it reads local data
and artifacts, which are not versioned. Integration tests include assertions
against the documented snapshot (for example, 3,240 days ending on
2026-11-14); refreshing the underlying data requires updating those
expectations. Running `pytest` on a fresh clone without data or after a snapshot
refresh may fail those contracts without indicating an algorithmic bug.

To validate code style and formatting with `Ruff`:

```bash
ruff check src scripts tests dashboard
ruff format --check src scripts tests dashboard
# or: make lint
```

## 10. Evaluation Results (Baseline vs. LightGBM)

The results below were obtained via **expanding-window walk-forward temporal backtesting** over the actual Gold-layer daily series for the `SE` subsystem (`2018-01-01` to `2026-11-14`), evaluating **52 successive weekly forecast origins** (`2025-11-15` to `2026-11-07`, totaling **364 forecasted days** from `2025-11-16` to `2026-11-14`) logged in MLflow (`run_id = 1461985175154c79ac2151ff5f6acf15`, model `lightgbm-se-daily-load` version `1`).

### Executive Summary (Does LightGBM Outperform the Baseline?)

**LightGBM outperforms the baseline overall and on average across each of the seven horizons in the documented backtest.** This does not guarantee outperformance on every individual day, week, or month, nor does it guarantee identical results on future forecasts.
Across all 52 weekly windows (364 evaluated days), LightGBM reduced **MAE** from **2,322.70 MWmed** to **1,701.53 MWmed** (**26.74% improvement**, or -621.17 MWmed in mean absolute error) and reduced **MAPE** from **5.22%** to **3.79%** (**27.38% relative improvement**).
The largest average gains occur at horizons $t+1$ and $t+2$ (over **36% MAE reduction**). Recent load, calendar, and lag features are part of the model, though this comparison does not isolate the causal contribution of individual features.

### Overall Comparison Table (52 Origins / 364 Evaluated Days)

| Model | MAE (MWmed) | MAPE (%) | Improvement vs. Baseline (MAE %) | Improvement vs. Baseline (MAPE %) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Seasonal Baseline ($t-7$) | 2,322.70 | 5.22% | — | — | Reference |
| **LightGBM** | **1,701.53** | **3.79%** | **+26.74%** | **+27.38%** | **LightGBM beats Baseline** |

### Performance Breakdown by Forecast Horizon ($t+1$ to $t+7$)

| Horizon ($h$) | Samples ($N$) | Baseline MAE (MWmed) | LightGBM MAE (MWmed) | MAE Improvement (%) | Baseline MAPE (%) | LightGBM MAPE (%) | MAPE Improvement (%) | Horizon Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $t+1$ | 52 | 2,106.04 | **1,342.11** | **+36.27%** | 5.31% | **3.37%** | **+36.49%** | LightGBM beats Baseline |
| $t+2$ | 52 | 2,647.98 | **1,664.95** | **+37.12%** | 5.94% | **3.72%** | **+37.38%** | LightGBM beats Baseline |
| $t+3$ | 52 | 2,524.62 | **1,921.87** | **+23.87%** | 5.48% | **4.15%** | **+24.34%** | LightGBM beats Baseline |
| $t+4$ | 52 | 2,062.76 | **1,831.22** | **+11.22%** | 4.39% | **3.91%** | **+11.00%** | LightGBM beats Baseline |
| $t+5$ | 52 | 2,271.75 | **1,810.03** | **+20.32%** | 4.96% | **3.95%** | **+20.33%** | LightGBM beats Baseline |
| $t+6$ | 52 | 2,443.49 | **1,679.67** | **+31.26%** | 5.37% | **3.64%** | **+32.24%** | LightGBM beats Baseline |
| $t+7$ | 52 | 2,202.26 | **1,660.88** | **+24.58%** | 5.12% | **3.82%** | **+25.39%** | LightGBM beats Baseline |

## 11. Forecast Intervals and Interpretation Limits

Empirical intervals target a nominal 90% coverage and use prior absolute errors
for the same horizon known up to the forecast origin. Calibration requires at
least 26 samples and uses up to the 52 most recent ones. Without sufficient
history, bounds remain unavailable. Each evaluated target is excluded from its
own calibration.

In the available evaluation, observed coverage was **92.31% across 182
evaluable forecasts**. This result does not guarantee future coverage: temporal
dependence, regime shifts, and successive retrainings limit the interpretation
of the nominal target. Likewise, a 3.79% MAPE does not mean 96.21% accuracy or
a ±3.79% confidence interval.

Default reference thresholds live in `config/risk_policy.yaml`: up to 5% daily
error per horizon, weight 2 for under-forecasting, and weight 1 for
over-forecasting. The monitoring routine also references a maximum of 20% of
days exceeding the threshold, 28/84-day rolling windows, and a degradation ratio
of 1.25. These are exploratory parameters without SLA validation or conversion
of weights into financial cost.

The documented backtest uses weekly Saturday origins. Testing relies on the
consolidated historical data available in the project and does not fully
simulate ONS data revisions or publication schedules. There is no documented
untouched prospective evaluation. Local data and artifacts are gitignored, and
dependencies use open version ranges without a lockfile. Verified reproducibility
applies to the reviewed environment and snapshot, not to arbitrary future
installations.

## 12. Calendar and Weather Experiments

The operational profile remains LightGBM with the national holiday calendar.
There is a candidate profile adding Carnival, Ash Wednesday, Corpus Christi,
bridge holidays, and year-end effects, as well as a candidate incorporating
forecasted temperature from ECMWF runs via Open-Meteo.

```bash
python scripts/fetch_weather.py
python scripts/evaluate_candidates.py
```

The download stores raw files and snapshots under `data/weather/` and requires
network access. Comparison uses 39 weeks for development and 13 reserved weeks
for retrospective verification, persisting model selection before evaluating the
reserved period. The current protocol requires artifacts from a 52-origin
backtest.

In the available experiment, the temperature candidate achieved **2.83% MAPE**,
compared to **4.04%** for the operational model on the same **84 targets**. Only
12 of the 13 reserved origins had complete temperature coverage; the
2026-09-19 run was excluded without imputation. This result should not be
compared directly to the 3.79% full-backtest MAPE, as periods and sample sets
differ.

The temperature candidate remains **experimental**. Temperature is a simple
average across four locations, availability assumes a 6-hour delay, and the
provenance of older forecast runs requires confirmation. Because historical data
had already been inspected, the reserved check is not an untouched prospective
holdout. No experiment automatically replaces the operational model in MLflow or
the dashboard.

Details and sources: [intervals, tracking, and candidate models](docs/risk_and_candidates.md).

## 13. Dashboard and Available Artifacts

The dashboard uses a dark theme and sidebar navigation with four sections:

- **Forecast (*Previsão*):** 21 days of history, 7 forecasted days, empirical
  intervals, daily tables, and a **Download CSV** (*Baixar CSV*) button.
- **Validation (*Validação*):** actual vs. forecasted series, horizon filter,
  monthly errors, worst days, and default error reference line.
- **History (*Histórico*):** model feature drivers and time-series explorer with
  year filter.
- **Guide (*Guia*):** explanation of terms, metrics, and interpretation limits.

The header calendar indicates the displayed period alongside the dataset cutoff
date. The former "Risk and Tracking" section and its per-session controls were
removed; the underlying technical routines and reports remain available via CLI.
No routines are scheduled automatically and no external alerts are sent.

| Location | Contents |
| --- | --- |
| `data/bronze/` | ONS Parquet files and ingestion manifest |
| `data/silver/`, `data/gold/` | dbt analytical exports |
| `data/predictions/` | Backtest, current forecast, and `issued_forecasts/` directory |
| `artifacts/backtest_summary.json` | Evaluation summary and MLflow run identification |
| `artifacts/mlflow.db`, `artifacts/mlruns/` | Local tracking database and logged models |
| `artifacts/review/` | Audit outputs and backtest reproduction |
| `artifacts/risk/` | Coverage, thresholds, critical splits, and monitoring |
| `artifacts/candidates/` | Frozen selection and candidate comparison |
| `data/weather/` | Weather forecast runs and coverage report |
| `.streamlit/config.toml`, `dashboard/styles.css` | UI theme and styles |

Issuances are archived with real timestamps. Tracking scores only full days
subsequent to issuance once actual values reach the Gold layer. Backtest data is
never recorded as if it were an operationally issued forecast.
`evaluate_risk.py` recalculates reports, though the underlying dataset must be
updated separately to ingest new actual values.

Internal references: [architecture](docs/architecture.md),
[data dictionary](docs/data_dictionary.md), [decisions](docs/decisions.md), and
[2026-11-17 technical review](docs/project_review_2026-11-17.md).
