"""Ponto de entrada CLI para treinamento do modelo LightGBM e geracao da previsao de 7 dias."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from energy_forecast.evaluation.risk import add_prediction_intervals, load_risk_policy
from energy_forecast.features.engineering import FeatureEngineeringError, load_gold_series
from energy_forecast.forecasting.lightgbm_model import ModelTrainingError
from energy_forecast.forecasting.mlflow_tracking import (
    train_final_model_and_generate_7d_forecast,
)
from energy_forecast.forecasting.monitoring import archive_forecast
from energy_forecast.utils.config import ConfigurationError, load_settings
from energy_forecast.utils.logging import configure_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos de linha de comando para o treinamento operacional."""
    parser = argparse.ArgumentParser(
        description="Treinamento do modelo LightGBM e previsao operacional de 7 dias (SE)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho opcional para o arquivo settings.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Executa o treinamento no historico completo da camada Gold e gera previsao de 7 dias."""
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args(argv)

    try:
        settings = load_settings(config_path=args.config)
        gold_df = load_gold_series(settings, source="parquet")
        forecaster, forecast_7d = train_final_model_and_generate_7d_forecast(
            df=gold_df,
            settings=settings,
        )
    except (ConfigurationError, FeatureEngineeringError, ModelTrainingError) as exc:
        logger.error("Falha no treinamento do modelo LightGBM: %s", exc)
        return 1

    settings.paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    latest_path = settings.paths.predictions_dir / "latest_forecast_7d.parquet"
    feat_imp_path = settings.paths.artifacts_dir / "feature_importance.csv"

    history_path = settings.paths.predictions_dir / "backtest_predictions.parquet"
    if history_path.is_file():
        forecast_7d = add_prediction_intervals(
            forecast_7d, pd.read_parquet(history_path), load_risk_policy(settings.paths.root_dir)
        )
    archive_forecast(forecast_7d, settings.paths.predictions_dir)

    forecast_7d.to_parquet(latest_path, index=False)
    forecaster.get_feature_importance().to_csv(feat_imp_path, index=False)

    logger.info(
        "Modelo LightGBM ajustado com %d amostras supervisionadas (corte=%s). "
        "Previsao de 7 dias salva em %s",
        forecaster.train_rows_,
        forecaster.train_Origin_cutoff_.strftime("%Y-%m-%d")
        if forecaster.train_Origin_cutoff_ is not None
        else "N/A",
        latest_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
