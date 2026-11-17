"""Integracao com MLflow para registro de experimentos, metricas, artefatos e modelos."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from energy_forecast.features.engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    build_inference_features_for_origin,
)
from energy_forecast.forecasting.baseline import SeasonalNaiveBaseline  # noqa: E402
from energy_forecast.forecasting.lightgbm_model import LightGBMForecaster  # noqa: E402
from energy_forecast.forecasting.monitoring import archive_forecast  # noqa: E402
from energy_forecast.utils.config import AppSettings  # noqa: E402
from energy_forecast.utils.logging import get_logger  # noqa: E402

if TYPE_CHECKING:
    from energy_forecast.evaluation.backtesting import BacktestResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class MLflowRegistrationResult:
    """Identificadores gerados no registro de uma execucao e versao de modelo no MLflow."""

    experiment_id: str
    experiment_name: str
    run_id: str
    registered_model_name: str
    model_version: str
    summary_json_path: Path
    backtest_predictions_path: Path
    latest_forecast_path: Path


def _resolve_sqlite_tracking_uri(settings: AppSettings) -> str:
    """Resolve um URI SQLite relativo para caminho absoluto compativel com MLflow."""
    raw_uri = settings.mlflow.tracking_uri
    if raw_uri.startswith("sqlite:///") and not raw_uri.startswith("sqlite:////"):
        rel_db_path = raw_uri.removeprefix("sqlite:///")
        abs_db_path = (settings.paths.root_dir / rel_db_path).resolve()
        abs_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_db_path.as_posix()}"
    return raw_uri


def train_final_model_and_generate_7d_forecast(
    df: pd.DataFrame,
    settings: AppSettings,
    lgbm_params: dict[str, Any] | None = None,
) -> tuple[LightGBMForecaster, pd.DataFrame]:
    """Ajusta o modelo LightGBM em todo o historico disponivel e preve os proximos 7 dias.

    Args:
        df: Serie completa da camada Gold.
        settings: Configuracoes globais do projeto.
        lgbm_params: Hiperparametros opcionais do LightGBM.

    Returns:
        tuple[LightGBMForecaster, pd.DataFrame]: Modelo ajustado no historico completo
            e DataFrame com as previsoes operacionais (Baseline e LightGBM) para os proximos 7 dias.
    """
    latest_origin = pd.Timestamp(df["data_referencia"].max()).normalize()

    forecaster = LightGBMForecaster(
        horizon_days=settings.forecasting.horizon_days,
        lags=settings.forecasting.lags,
        rolling_windows=settings.forecasting.rolling_windows,
        country_code=settings.forecasting.country_holidays,
        date_col="data_referencia",
        target_col="carga_mwmed",
    )
    if lgbm_params is not None:
        forecaster.model_params = lgbm_params

    forecaster.fit(df, cutoff_date=latest_origin)
    lgbm_7d = forecaster.predict_from_origin(df, origin_date=latest_origin)

    baseline = SeasonalNaiveBaseline(
        horizon_days=settings.forecasting.horizon_days,
        seasonal_period_days=7,
        date_col="data_referencia",
        target_col="carga_mwmed",
    )
    base_7d = baseline.predict_from_origin(df, origin_date=latest_origin)

    operational_df = pd.merge(
        base_7d[["origin_date", "target_date", "horizon", "reference_date", "y_pred_baseline"]],
        lgbm_7d[["origin_date", "target_date", "horizon", "y_pred_lgbm"]],
        on=["origin_date", "target_date", "horizon"],
        how="inner",
    )
    operational_df["diff_lgbm_minus_baseline"] = (
        operational_df["y_pred_lgbm"] - operational_df["y_pred_baseline"]
    )
    return forecaster, operational_df


def log_backtest_and_model_to_mlflow(
    df: pd.DataFrame,
    backtest_result: BacktestResult,
    settings: AppSettings,
    run_name: str = "walk-forward-backtest-lgbm-vs-baseline",
) -> MLflowRegistrationResult:
    """Persiste artefatos locais e registra parametros, metricas, artefatos e modelo no MLflow.

    Args:
        df: Serie completa da camada Gold.
        backtest_result: Resultado do backtesting temporal.
        settings: Configuracoes da aplicacao.
        run_name: Nome descritivo da run no MLflow.

    Returns:
        MLflowRegistrationResult: Metadados da run e da versao do modelo registrada.
    """
    predictions_dir = settings.paths.predictions_dir
    artifacts_dir = settings.paths.artifacts_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Treinamento do modelo operacional final na data mais recente da serie Gold
    final_forecaster, latest_forecast_df = train_final_model_and_generate_7d_forecast(
        df=df,
        settings=settings,
        lgbm_params=backtest_result.final_forecaster.model_params,
    )
    feature_importance_df = final_forecaster.get_feature_importance()
    # Calibrate using the just-evaluated model, never a previous evaluation's residuals.
    from energy_forecast.evaluation.risk import add_prediction_intervals, load_risk_policy

    latest_forecast_df = add_prediction_intervals(
        latest_forecast_df, backtest_result.predictions, load_risk_policy(settings.paths.root_dir)
    )
    archive_forecast(latest_forecast_df, predictions_dir)

    # 2. Persistencia dos artefatos de previsao e avaliacao em disco
    backtest_predictions_path = predictions_dir / "backtest_predictions.parquet"
    latest_forecast_path = predictions_dir / "latest_forecast_7d.parquet"
    horizon_csv_path = artifacts_dir / "backtest_horizon_metrics.csv"
    fold_csv_path = artifacts_dir / "backtest_fold_metrics.csv"
    feature_imp_csv_path = artifacts_dir / "feature_importance.csv"
    summary_json_path = artifacts_dir / "backtest_summary.json"

    backtest_result.predictions.to_parquet(backtest_predictions_path, index=False)
    latest_forecast_df.to_parquet(latest_forecast_path, index=False)
    backtest_result.horizon_metrics.to_csv(horizon_csv_path, index=False)
    backtest_result.fold_metrics.to_csv(fold_csv_path, index=False)
    feature_importance_df.to_csv(feature_imp_csv_path, index=False)

    # 3. Configuracao do servidor de rastreamento MLflow (SQLite + diretorio local de artefatos)
    tracking_uri = _resolve_sqlite_tracking_uri(settings)
    artifact_Location_uri = (artifacts_dir / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(settings.mlflow.experiment_name)
    if exp is None:
        experiment_id = client.create_experiment(
            name=settings.mlflow.experiment_name,
            artifact_location=artifact_Location_uri,
            tags={
                "project": settings.project.name,
                "subsystem": settings.forecasting.target_subsystem,
                "unit": settings.ingestion.target_unit,
                "data_source": "ONS - Operador Nacional do Sistema Eletrico",
            },
        )
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(settings.mlflow.experiment_name)

    latest_origin = pd.Timestamp(df["data_referencia"].max()).normalize()
    sample_infer_df = build_inference_features_for_origin(
        df=df,
        origin_date=latest_origin,
        horizon_days=settings.forecasting.horizon_days,
        lags=settings.forecasting.lags,
        rolling_windows=settings.forecasting.rolling_windows,
        country_code=settings.forecasting.country_holidays,
    )
    x_example = sample_infer_df[list(FEATURE_COLUMNS)]
    assert final_forecaster.model_ is not None
    y_example = final_forecaster.model_.predict(x_example)
    signature = infer_signature(x_example, y_example)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "subsystem": settings.forecasting.target_subsystem,
                "unit": settings.ingestion.target_unit,
                "validation_strategy": "expanding_window_walk_forward",
                "first_origin_date": backtest_result.summary.first_origin_date,
                "last_origin_date": backtest_result.summary.last_origin_date,
                "lgbm_beats_baseline_mae": str(backtest_result.summary.lgbm_beats_baseline_mae),
                "lgbm_beats_baseline_mape": str(backtest_result.summary.lgbm_beats_baseline_mape),
            }
        )

        # Parametros do modelo, features e protocolo de backtesting
        params_to_log: dict[str, Any] = {
            f"lgbm_{k}": v for k, v in final_forecaster.model_params.items()
        }
        params_to_log.update(
            {
                "horizon_days": settings.forecasting.horizon_days,
                "lags": ",".join(str(x) for x in settings.forecasting.lags),
                "rolling_windows": ",".join(str(x) for x in settings.forecasting.rolling_windows),
                "country_holidays": settings.forecasting.country_holidays,
                "min_train_days": settings.backtesting.min_train_days,
                "step_days": settings.backtesting.step_days,
                "num_origins": backtest_result.summary.num_origins,
                "total_evaluated_days": backtest_result.summary.total_evaluated_days,
            }
        )
        mlflow.log_params(params_to_log)

        # Metricas agregadas e estratificadas por horizonte (h = 1..7)
        metrics_to_log: dict[str, float] = {
            "mae_baseline": backtest_result.summary.mae_baseline,
            "mae_lgbm": backtest_result.summary.mae_lgbm,
            "mae_improvement_pct": backtest_result.summary.mae_improvement_pct,
            "mape_baseline": backtest_result.summary.mape_baseline,
            "mape_lgbm": backtest_result.summary.mape_lgbm,
            "mape_improvement_pct": backtest_result.summary.mape_improvement_pct,
        }
        for _, h_row in backtest_result.horizon_metrics.iterrows():
            h = int(h_row["horizon"])
            metrics_to_log[f"mae_baseline_h{h}"] = float(h_row["mae_baseline"])
            metrics_to_log[f"mae_lgbm_h{h}"] = float(h_row["mae_lgbm"])
            metrics_to_log[f"mae_improvement_pct_h{h}"] = float(h_row["mae_improvement_pct"])
            metrics_to_log[f"mape_baseline_h{h}"] = float(h_row["mape_baseline"])
            metrics_to_log[f"mape_lgbm_h{h}"] = float(h_row["mape_lgbm"])
            metrics_to_log[f"mape_improvement_pct_h{h}"] = float(h_row["mape_improvement_pct"])

        mlflow.log_metrics(metrics_to_log)

        # Registro do modelo LightGBM com assinatura e versionamento no Model Registry
        mlflow.lightgbm.log_model(
            lgb_model=final_forecaster.model_,
            name="model",
            signature=signature,
            input_example=x_example,
            registered_model_name=settings.mlflow.registered_model_name,
        )

        versions = client.search_model_versions(f"name='{settings.mlflow.registered_model_name}'")
        run_versions = [v.version for v in versions if v.run_id == run_id]
        model_version = str(run_versions[0]) if run_versions else "1"

        summary_payload = {
            "experiment_id": str(experiment_id),
            "experiment_name": settings.mlflow.experiment_name,
            "run_id": run_id,
            "registered_model_name": settings.mlflow.registered_model_name,
            "model_version": model_version,
            "summary": backtest_result.summary.to_dict(),
            "horizon_metrics": backtest_result.horizon_metrics.to_dict(orient="records"),
            "top_features": feature_importance_df.head(10).to_dict(orient="records"),
        }
        summary_json_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Registro de todos os artefatos de avaliacao e previsao na run
        mlflow.log_artifact(str(summary_json_path), artifact_path="evaluation")
        mlflow.log_artifact(str(horizon_csv_path), artifact_path="evaluation")
        mlflow.log_artifact(str(fold_csv_path), artifact_path="evaluation")
        mlflow.log_artifact(str(feature_imp_csv_path), artifact_path="evaluation")
        mlflow.log_artifact(str(backtest_predictions_path), artifact_path="predictions")
        mlflow.log_artifact(str(latest_forecast_path), artifact_path="predictions")

    logger.info(
        "Experimento registrado no MLflow: experiment=%s (id=%s), run_id=%s, modelo=%s v%s",
        settings.mlflow.experiment_name,
        experiment_id,
        run_id,
        settings.mlflow.registered_model_name,
        model_version,
    )

    return MLflowRegistrationResult(
        experiment_id=str(experiment_id),
        experiment_name=settings.mlflow.experiment_name,
        run_id=run_id,
        registered_model_name=settings.mlflow.registered_model_name,
        model_version=model_version,
        summary_json_path=summary_json_path,
        backtest_predictions_path=backtest_predictions_path,
        latest_forecast_path=latest_forecast_path,
    )
