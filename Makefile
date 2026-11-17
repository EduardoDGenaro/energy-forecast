.PHONY: help setup lint format test ingest dbt-run dbt-test dbt train evaluate mlflow-ui dashboard docker-build clean

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
RUFF ?= ruff
DBT ?= dbt

help:
	@echo "Comandos disponiveis no projeto Energy Forecast:"
	@echo "  make setup        - Instala o pacote em modo editavel com dependencias de desenvolvimento"
	@echo "  make lint         - Executa verificacao estatica e de formatacao com Ruff"
	@echo "  make format       - Formata o codigo e corrige imports automaticamente com Ruff"
	@echo "  make test         - Executa a suite de testes automatizados com pytest"
	@echo "  make ingest       - Executa a ingestao de dados brutos do ONS para a camada Bronze"
	@echo "  make dbt-run      - Executa os modelos de transformacao Silver e Gold no DuckDB via dbt"
	@echo "  make dbt-test     - Executa os testes de qualidade de dados no dbt"
	@echo "  make dbt          - Executa dbt run seguido de dbt test"
	@echo "  make train        - Executa o pipeline de treinamento e registro no MLflow"
	@echo "  make evaluate     - Executa o backtesting temporal comparando Baseline e LightGBM"
	@echo "  make mlflow-ui    - Inicia a interface local do MLflow na porta 5000"
	@echo "  make dashboard    - Inicia o dashboard opcional em Streamlit na porta 8501"
	@echo "  make docker-build - Constroi a imagem Docker do projeto"
	@echo "  make clean        - Remove caches locais de execucao, testes e build"

setup:
	$(PIP) install -e ".[dev,dashboard]"

lint:
	$(RUFF) check src scripts tests dashboard
	$(RUFF) format --check src scripts tests dashboard

format:
	$(RUFF) check --fix src scripts tests dashboard
	$(RUFF) format src scripts tests dashboard

test:
	$(PYTEST)

ingest:
	$(PYTHON) scripts/ingest.py

dbt-run:
	$(PYTHON) scripts/run_dbt.py --command run

dbt-test:
	$(PYTHON) scripts/run_dbt.py --command test

dbt:
	$(PYTHON) scripts/run_dbt.py --command build

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000

dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker compose build

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'build', 'dist', 'dbt/target', 'dbt/logs']]"

