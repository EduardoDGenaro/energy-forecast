"""Testes unitarios para carregamento de configuracoes, resolucao de caminhos e estrutura."""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_forecast.utils.config import (
    ConfigurationError,
    get_project_root,
    load_settings,
)
from energy_forecast.utils.logging import configure_logging, get_logger


def test_default_settings_load_successfully() -> None:
    """Garante que o arquivo config/settings.yaml padrao e carregado e validado corretamente."""
    settings = load_settings()

    assert settings.project.name == "energy-forecast"
    assert settings.ingestion.target_subsystem == "SE"
    assert settings.ingestion.start_year == 2018
    assert settings.ingestion.target_unit == "MWmed"
    assert settings.forecasting.horizon_days == 7
    assert settings.forecasting.lags == (7, 14, 28)
    assert settings.backtesting.metrics == ("mae", "mape")
    assert settings.paths.root_dir == get_project_root()
    assert settings.paths.bronze_dir.is_absolute()
    assert settings.paths.dbt_project_dir.name == "dbt"


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica se variaveis de ambiente sobrescrevem os valores do YAML quando definidas."""
    monkeypatch.setenv("START_YEAR", "2020")
    monkeypatch.setenv("FORECAST_HORIZON", "14")
    monkeypatch.setenv("DUCKDB_PATH", "data/custom_test.duckdb")

    settings = load_settings()

    assert settings.ingestion.start_year == 2020
    assert settings.forecasting.horizon_days == 14
    assert settings.backtesting.horizon_days == 14
    assert settings.paths.duckdb_path.name == "custom_test.duckdb"


def test_missing_config_file_raises_configuration_error(tmp_path: Path) -> None:
    """Garante erro explicito quando o arquivo de configuracao nao existe."""
    missing_file = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigurationError, match="Arquivo de configuracao nao encontrado"):
        load_settings(config_path=missing_file, root_dir=tmp_path)


def test_invalid_horizon_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garante validacao explicita contra horizonte de previsao invalido (<= 0)."""
    monkeypatch.setenv("FORECAST_HORIZON", "0")
    with pytest.raises(ConfigurationError, match="horizon_days"):
        load_settings()


def test_logging_configuration() -> None:
    """Verifica que o logger estruturado e inicializado sem erros."""
    configure_logging("DEBUG")
    logger = get_logger("energy_forecast.test")
    assert logger.name == "energy_forecast.test"


def test_repository_required_directories_exist() -> None:
    """Valida a existencia dos diretorios e arquivos estruturais exigidos na Fase 1."""
    root = get_project_root()

    expected_dirs = [
        "config",
        "data/bronze",
        "data/silver",
        "data/gold",
        "data/predictions",
        "src/energy_forecast/ingestion",
        "src/energy_forecast/features",
        "src/energy_forecast/forecasting",
        "src/energy_forecast/evaluation",
        "src/energy_forecast/utils",
        "dbt/models/sources",
        "dbt/models/silver",
        "dbt/models/gold",
        "dbt/tests",
        "dashboard",
        "scripts",
        "tests",
        "artifacts",
        "docs",
    ]
    for rel_dir in expected_dirs:
        assert (root / rel_dir).is_dir(), f"Diretorio obrigatorio ausente: {rel_dir}"

    expected_files = [
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".env.example",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile",
        "config/settings.yaml",
        "dbt/dbt_project.yml",
        "dbt/profiles.yml.example",
        "dashboard/app.py",
        "scripts/ingest.py",
        "scripts/run_dbt.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "docs/architecture.md",
        "docs/decisions.md",
        "docs/data_dictionary.md",
    ]
    for rel_file in expected_files:
        assert (root / rel_file).is_file(), f"Arquivo obrigatorio ausente: {rel_file}"
