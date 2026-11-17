"""Carregamento centralizado de configuracoes e resolucao de caminhos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Excecao lancada quando a configuracao do projeto e invalida ou ausente."""


@dataclass(frozen=True)
class ProjectMetadata:
    """Metadados gerais do projeto."""

    name: str
    version: str
    description: str


@dataclass(frozen=True)
class PathsConfig:
    """Caminhos absolutos resolvidos para as camadas de dados e artefatos."""

    root_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    predictions_dir: Path
    artifacts_dir: Path
    duckdb_path: Path
    manifest_path: Path
    dbt_project_dir: Path


@dataclass(frozen=True)
class IngestionConfig:
    """Parametros de ingestao de dados publicos do ONS."""

    dataset_slug: str
    ckan_package_url: str
    s3_base_url: str
    s3_uri_prefix: str
    file_prefix: str
    start_year: int
    target_subsystem: str
    target_unit: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class ForecastingConfig:
    """Parametros de engenharia de atributos e modelagem."""

    target_column: str
    date_column: str
    subsystem_column: str
    target_subsystem: str
    horizon_days: int
    country_holidays: str
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]


@dataclass(frozen=True)
class BacktestingConfig:
    """Parametros de validacao temporal por origens sucessivas."""

    min_train_days: int
    step_days: int
    horizon_days: int
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class MLflowConfig:
    """Parametros de rastreamento de experimentos no MLflow."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str


@dataclass(frozen=True)
class AppSettings:
    """Objeto raiz imutavel contendo todas as configuracoes do sistema."""

    project: ProjectMetadata
    paths: PathsConfig
    ingestion: IngestionConfig
    forecasting: ForecastingConfig
    backtesting: BacktestingConfig
    mlflow: MLflowConfig


def get_project_root() -> Path:
    """Retorna o diretorio raiz do repositorio.

    Returns:
        Path: Caminho absoluto para a raiz do projeto.
    """
    return Path(__file__).resolve().parents[3]


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    """Resolve um caminho relativo em relacao a raiz do projeto."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def _read_yaml(config_path: Path) -> dict[str, Any]:
    """Le e valida um arquivo YAML de configuracao."""
    if not config_path.exists():
        raise ConfigurationError(f"Arquivo de configuracao nao encontrado: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            content = yaml.safe_load(file_handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Falha ao interpretar arquivo YAML em {config_path}: {exc}"
        ) from exc

    if not isinstance(content, dict):
        raise ConfigurationError(
            f"O arquivo de configuracao {config_path} deve conter um mapeamento YAML na raiz."
        )
    return content


def load_settings(
    config_path: Path | str | None = None,
    root_dir: Path | str | None = None,
) -> AppSettings:
    """Carrega as configuracoes do projeto combinando YAML e variaveis de ambiente.

    Args:
        config_path: Caminho opcional para o arquivo settings.yaml.
        root_dir: Diretorio raiz opcional para resolucao de caminhos relativos.

    Returns:
        AppSettings: Configuracoes tipadas e validadas.

    Raises:
        ConfigurationError: Caso o arquivo nao exista ou campos obrigatorios estejam invalidos.
    """
    load_dotenv(override=False)

    resolved_root = Path(root_dir).resolve() if root_dir is not None else get_project_root()

    if config_path is not None:
        resolved_config_path = _resolve_path(resolved_root, str(config_path))
    else:
        env_config_path = os.getenv("PROJECT_CONFIG_PATH", "config/settings.yaml")
        resolved_config_path = _resolve_path(resolved_root, env_config_path)

    raw = _read_yaml(resolved_config_path)

    try:
        project_raw = raw["project"]
        paths_raw = raw["paths"]
        ingestion_raw = raw["ingestion"]
        forecasting_raw = raw["forecasting"]
        backtesting_raw = raw["backtesting"]
        mlflow_raw = raw["mlflow"]
    except KeyError as exc:
        raise ConfigurationError(
            f"Secao obrigatoria ausente no arquivo de configuracao {resolved_config_path}: {exc}"
        ) from exc

    duckdb_raw_path = os.getenv("DUCKDB_PATH", str(paths_raw["duckdb_path"]))
    s3_base_url = os.getenv("ONS_S3_BASE_URL", str(ingestion_raw["s3_base_url"]))
    ckan_package_url = os.getenv("ONS_CKAN_API_URL", str(ingestion_raw["ckan_package_url"]))
    target_subsystem = os.getenv("TARGET_SUBSYSTEM", str(ingestion_raw["target_subsystem"]))
    start_year = int(os.getenv("START_YEAR", str(ingestion_raw["start_year"])))
    horizon_days = int(os.getenv("FORECAST_HORIZON", str(forecasting_raw["horizon_days"])))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(mlflow_raw["tracking_uri"]))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", str(mlflow_raw["experiment_name"]))

    if horizon_days <= 0:
        raise ConfigurationError("O horizonte de previsao (horizon_days) deve ser maior que zero.")
    if start_year < 2000:
        raise ConfigurationError("O ano inicial de ingestao (start_year) deve ser >= 2000.")

    project_cfg = ProjectMetadata(
        name=str(project_raw["name"]),
        version=str(project_raw["version"]),
        description=str(project_raw["description"]),
    )

    paths_cfg = PathsConfig(
        root_dir=resolved_root,
        bronze_dir=_resolve_path(resolved_root, str(paths_raw["bronze_dir"])),
        silver_dir=_resolve_path(resolved_root, str(paths_raw["silver_dir"])),
        gold_dir=_resolve_path(resolved_root, str(paths_raw["gold_dir"])),
        predictions_dir=_resolve_path(resolved_root, str(paths_raw["predictions_dir"])),
        artifacts_dir=_resolve_path(resolved_root, str(paths_raw["artifacts_dir"])),
        duckdb_path=_resolve_path(resolved_root, duckdb_raw_path),
        manifest_path=_resolve_path(resolved_root, str(paths_raw["manifest_path"])),
        dbt_project_dir=_resolve_path(resolved_root, str(paths_raw["dbt_project_dir"])),
    )

    ingestion_cfg = IngestionConfig(
        dataset_slug=str(ingestion_raw["dataset_slug"]),
        ckan_package_url=ckan_package_url,
        s3_base_url=s3_base_url,
        s3_uri_prefix=str(ingestion_raw["s3_uri_prefix"]),
        file_prefix=str(ingestion_raw["file_prefix"]),
        start_year=start_year,
        target_subsystem=target_subsystem,
        target_unit=str(ingestion_raw["target_unit"]),
        request_timeout_seconds=int(ingestion_raw["request_timeout_seconds"]),
    )

    forecasting_cfg = ForecastingConfig(
        target_column=str(forecasting_raw["target_column"]),
        date_column=str(forecasting_raw["date_column"]),
        subsystem_column=str(forecasting_raw["subsystem_column"]),
        target_subsystem=target_subsystem,
        horizon_days=horizon_days,
        country_holidays=str(forecasting_raw["country_holidays"]),
        lags=tuple(int(lag) for lag in forecasting_raw["lags"]),
        rolling_windows=tuple(int(window) for window in forecasting_raw["rolling_windows"]),
    )

    backtesting_cfg = BacktestingConfig(
        min_train_days=int(backtesting_raw["min_train_days"]),
        step_days=int(backtesting_raw["step_days"]),
        horizon_days=horizon_days,
        metrics=tuple(str(metric) for metric in backtesting_raw["metrics"]),
    )

    mlflow_cfg = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registered_model_name=str(mlflow_raw["registered_model_name"]),
    )

    return AppSettings(
        project=project_cfg,
        paths=paths_cfg,
        ingestion=ingestion_cfg,
        forecasting=forecasting_cfg,
        backtesting=backtesting_cfg,
        mlflow=mlflow_cfg,
    )
