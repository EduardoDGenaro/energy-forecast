"""Testes unitarios e de integracao para metricas, LightGBM e motor de backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.evaluation.backtesting import (
    generate_backtest_origins,
    run_walk_forward_backtest,
)
from energy_forecast.evaluation.metrics import (
    EvaluationError,
    calculate_improvement_pct,
    calculate_mae,
    calculate_mape,
)
from energy_forecast.forecasting.lightgbm_model import LightGBMForecaster, ModelTrainingError


def _make_seasonal_synthetic_series(num_days: int = 160) -> pd.DataFrame:
    """Cria uma serie sintetica com padrao semanal e tendência leve para testes."""
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    weekday_effect = np.array([1500.0, 1800.0, 1900.0, 1850.0, 1600.0, -1200.0, -2500.0])
    loads = [40000.0 + weekday_effect[dt.day_of_week] + (i * 5.0) for i, dt in enumerate(dates)]
    return pd.DataFrame({"data_referencia": dates, "carga_mwmed": loads})


def test_metrics_mae_mape_and_improvement_sign() -> None:
    """Verifica calculo exato de MAE, MAPE e sinal de melhoria ou degradacao."""
    y_true = [100.0, 200.0]
    y_pred_base = [110.0, 220.0]  # erros absolutos: 10, 20 -> MAE=15, MAPE=10%
    y_pred_good = [105.0, 190.0]  # erros absolutos: 5, 10 -> MAE=7.5, MAPE=5%
    y_pred_worse = [120.0, 240.0]  # erros absolutos: 20, 40 -> MAE=30, MAPE=20%

    mae_base = calculate_mae(y_true, y_pred_base)
    mae_good = calculate_mae(y_true, y_pred_good)
    mae_worse = calculate_mae(y_true, y_pred_worse)

    assert mae_base == pytest.approx(15.0)
    assert mae_good == pytest.approx(7.5)
    assert calculate_mape(y_true, y_pred_base) == pytest.approx(10.0)
    assert calculate_mape(y_true, y_pred_good) == pytest.approx(5.0)

    # Melhoria de 50% quando o erro cai pela metade
    assert calculate_improvement_pct(mae_base, mae_good) == pytest.approx(50.0)
    # Degradacao (-100%) quando o erro dobra em relacao ao baseline
    assert calculate_improvement_pct(mae_base, mae_worse) == pytest.approx(-100.0)


def test_mape_rejects_non_positive_actuals() -> None:
    """Garante erro explicito caso y_true contenha zero ou valores negativos."""
    with pytest.raises(EvaluationError, match="estritamente positivos"):
        calculate_mape([100.0, 0.0], [100.0, 10.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("calculate", [calculate_mae, calculate_mape])
def test_metrics_reject_non_finite_values(value: float, calculate) -> None:
    for actuals, predictions in (
        ([100.0, value], [100.0, 100.0]),
        ([100.0, 100.0], [100.0, value]),
    ):
        with pytest.raises(EvaluationError):
            calculate(actuals, predictions)


def test_lightgbm_forecaster_fit_predict_and_no_future_leakage() -> None:
    """Garante que o ajuste e a previsao do LightGBM sao imunes a dados futuros (> origin_date)."""
    df = _make_seasonal_synthetic_series(140)
    origin_date = pd.Timestamp("2024-04-30")
    fast_params = {
        "objective": "regression_l1",
        "n_estimators": 30,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 5,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }

    model_clean = LightGBMForecaster(model_params=fast_params)
    model_clean.fit(df, cutoff_date=origin_date)
    preds_clean = model_clean.predict_from_origin(df, origin_date=origin_date)
    imp_df = model_clean.get_feature_importance()

    assert len(preds_clean) == 7
    assert not imp_df.empty

    with pytest.raises(ModelTrainingError, match="anteceder o corte"):
        model_clean.predict_from_origin(df, origin_date=origin_date - pd.Timedelta(days=1))

    # Corrompe todas as observacoes posteriores a origin_date e re-treina do zero
    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["data_referencia"] > origin_date, "carga_mwmed"] = 9999999.0

    model_corrupted = LightGBMForecaster(model_params=fast_params)
    model_corrupted.fit(df_corrupted, cutoff_date=origin_date)
    preds_corrupted = model_corrupted.predict_from_origin(df_corrupted, origin_date=origin_date)

    pd.testing.assert_frame_equal(preds_clean, preds_corrupted)


def test_walk_forward_backtest_expanding_window_protocol() -> None:
    """Valida a execucao do backtesting temporal com multiplas origens sucessivas."""
    df = _make_seasonal_synthetic_series(160)
    fast_params = {
        "objective": "regression_l1",
        "n_estimators": 25,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 5,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }

    origins = generate_backtest_origins(
        df, min_train_days=90, step_days=7, horizon_days=7, max_origins=3
    )
    assert len(origins) == 3

    result = run_walk_forward_backtest(
        df=df,
        min_train_days=90,
        step_days=7,
        horizon_days=7,
        max_origins=3,
        lgbm_params=fast_params,
    )

    assert result.summary.num_origins == 3
    assert result.summary.total_evaluated_days == 21
    assert len(result.horizon_metrics) == 7
    assert len(result.fold_metrics) == 3
    assert result.fold_metrics["train_days"].is_monotonic_increasing


def test_real_backtest_artifacts_and_mlflow_outputs_exist() -> None:
    """Valida que os artefatos reais do backtesting e do MLflow foram gerados com integridade."""
    import json

    from energy_forecast.utils.config import load_settings

    settings = load_settings()
    summary_path = settings.paths.artifacts_dir / "backtest_summary.json"
    bt_preds_path = settings.paths.predictions_dir / "backtest_predictions.parquet"
    latest_path = settings.paths.predictions_dir / "latest_forecast_7d.parquet"

    assert summary_path.is_file()
    assert bt_preds_path.is_file()
    assert latest_path.is_file()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["registered_model_name"] == "lightgbm-se-daily-load"
    assert payload["summary"]["num_origins"] == 52
    assert payload["summary"]["total_evaluated_days"] == 364
    assert len(payload["horizon_metrics"]) == 7
