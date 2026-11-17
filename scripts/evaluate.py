"""Ponto de entrada CLI para backtesting temporal, avaliacao comparativa e registro no MLflow."""

from __future__ import annotations

import argparse
import sys

from energy_forecast.evaluation.backtesting import run_walk_forward_backtest
from energy_forecast.evaluation.metrics import EvaluationError
from energy_forecast.features.engineering import FeatureEngineeringError, load_gold_series
from energy_forecast.forecasting.lightgbm_model import ModelTrainingError
from energy_forecast.forecasting.mlflow_tracking import log_backtest_and_model_to_mlflow
from energy_forecast.utils.config import ConfigurationError, load_settings
from energy_forecast.utils.logging import configure_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos de linha de comando para avaliacao e backtesting."""
    parser = argparse.ArgumentParser(
        description="Execucao de backtesting temporal comparando Baseline Sazonal e LightGBM."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho opcional para o arquivo settings.yaml.",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=52,
        help="Numero maximo de origens sucessivas mais recentes no backtest (padrao: 52 semanas).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Executa o backtesting temporal completo e registra resultados e modelo no MLflow."""
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args(argv)

    try:
        settings = load_settings(config_path=args.config)
        gold_df = load_gold_series(settings, source="parquet")

        bt_result = run_walk_forward_backtest(
            df=gold_df,
            min_train_days=settings.backtesting.min_train_days,
            step_days=settings.backtesting.step_days,
            horizon_days=settings.forecasting.horizon_days,
            lags=settings.forecasting.lags,
            rolling_windows=settings.forecasting.rolling_windows,
            country_code=settings.forecasting.country_holidays,
            max_origins=args.max_origins,
        )

        mlflow_reg = log_backtest_and_model_to_mlflow(
            df=gold_df,
            backtest_result=bt_result,
            settings=settings,
        )
    except (
        ConfigurationError,
        FeatureEngineeringError,
        ModelTrainingError,
        EvaluationError,
    ) as exc:
        logger.error("Falha na execucao do backtesting e avaliacao: %s", exc)
        return 1

    s = bt_result.summary
    logger.info(
        "Avaliacao finalizada | Origens=%d | Dias avaliados=%d | "
        "MAE Baseline=%.2f vs LightGBM=%.2f (Melhoria=%.2f%%, %s) | "
        "MAPE Baseline=%.2f%% vs LightGBM=%.2f%% (Melhoria=%.2f%%, %s) | "
        "MLflow run_id=%s, modelo=%s v%s",
        s.num_origins,
        s.total_evaluated_days,
        s.mae_baseline,
        s.mae_lgbm,
        s.mae_improvement_pct,
        s.status_mae,
        s.mape_baseline,
        s.mape_lgbm,
        s.mape_improvement_pct,
        s.status_mape,
        mlflow_reg.run_id,
        mlflow_reg.registered_model_name,
        mlflow_reg.model_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
